"""GLM-5.3-Flash vision tower — PyTorch sidecar.

Produces the 256 x 4096 image embeddings that the text engine splices in at
`<|image|>` (token 154854). The tower is dense, ~563M params, and runs once per
image, so a sidecar on the GPU is far cheaper than porting it to C.

Architecture is read off the checkpoint's own tensor names and vision_config:

    patch_embed.proj   Conv3d(3 -> 1024, k=(2,14,14), s=(2,14,14))   + bias
    24 x block         norm1 -> attn -> +res -> norm2 -> SwiGLU -> +res
                       attn: qkv(1024->3072, bias), q_norm/k_norm on head_dim,
                             2D RoPE, proj(1024->1024, bias)
    post_layernorm     RMSNorm(1024)
    downsample         Conv2d(1024 -> 4096, k=2, s=2) + bias   (spatial_merge=2)
    merger             proj(4096->4096) -> post_projection_norm(LayerNorm)
                       -> SwiGLU(4096 -> 10240 -> 4096)

Norm types are determined by which tensors exist: norm1/norm2/q_norm/k_norm/
post_layernorm ship weight only -> RMSNorm. merger.post_projection_norm ships
weight AND bias -> LayerNorm.

448 / 14 = 32x32 patches -> downsample 2x2 -> 16x16 = 256 tokens at 4096 dims,
which is exactly the text model's hidden size, so no extra projection is needed.

UNVERIFIED: the 2D RoPE convention (axial split and ordering) is not pinned down
by the checkpoint, and GLM-5.3 has no reference implementation to diff against.
See rope_2d() — this is the most likely place for a silent mismatch.
"""
import json, math, struct, glob, os
import torch
import torch.nn.functional as F

CKPT = "D:/GLM5.3-flash/GLM-5.3-Flash"


# ---------------------------------------------------------------- weights ---
def load_vision_weights(ckpt=CKPT, device="cuda", dtype=torch.float32):
    """Load only the shard(s) holding `model.visual.*`. Returns {name: tensor}."""
    index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    want = {k: v for k, v in index.items() if k.startswith("model.visual.")}
    shards = sorted(set(want.values()))
    out = {}
    for shard in shards:
        path = os.path.join(ckpt, shard)
        with open(path, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(hlen))
            base = 8 + hlen
            for name, meta in header.items():
                if name == "__metadata__" or not name.startswith("model.visual."):
                    continue
                s, e = meta["data_offsets"]
                fh.seek(base + s)
                raw = fh.read(e - s)
                t = torch.frombuffer(bytearray(raw), dtype=_dt(meta["dtype"]))
                out[name] = t.reshape(meta["shape"]).to(device=device, dtype=dtype)
    missing = set(want) - set(out)
    if missing:
        raise RuntimeError(f"{len(missing)} vision tensors missing, e.g. {sorted(missing)[:3]}")
    return out


def _dt(s):
    return {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
            "F64": torch.float64, "U8": torch.uint8, "I8": torch.int8}[s]


# ------------------------------------------------------------------ parts ---
def rms_norm(x, weight, eps=1e-5):
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * weight.float()).to(x.dtype)


def layer_norm(x, weight, bias, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def swiglu(x, gate_w, gate_b, up_w, up_b, down_w, down_b, limit=10.0):
    """Clamped SwiGLU, matching the text stack: gate clamped ABOVE only, up on both sides."""
    g = F.linear(x, gate_w, gate_b)
    u = F.linear(x, up_w, up_b)
    g = g.clamp(max=limit)
    u = u.clamp(min=-limit, max=limit)
    return F.linear(F.silu(g) * u, down_w, down_b)


def rope_2d(head_dim, h, w, device, theta=10000.0):
    """Parameter-free axial 2D RoPE over an h x w patch grid.

    GLM-5.3 dropped GLM-4V's learned interpolated position embedding, which is why
    the checkpoint carries no positional tensor at all. Half the head dims encode
    the row index, half the column index.

    UNVERIFIED: split order (row-half first) and the rotate-half convention are
    assumptions. If image understanding comes out subtly wrong while shapes and
    norms all check out, start here.
    """
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, 2, device=device).float() / half))
    rows = torch.arange(h, device=device).float()
    cols = torch.arange(w, device=device).float()
    fr = torch.outer(rows, freqs)                      # [h, half/2]
    fc = torch.outer(cols, freqs)                      # [w, half/2]
    fr = fr[:, None, :].expand(h, w, -1)
    fc = fc[None, :, :].expand(h, w, -1)
    ang = torch.cat([fr, fc], dim=-1).reshape(h * w, -1)   # [h*w, half]
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    """x: [tokens, heads, head_dim]. Rotates pairs (i, i+half)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[:, None, :]
    s = sin[:, None, :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


# ---------------------------------------------------------------- forward ---
@torch.no_grad()
def encode_image(pixel_values, W, cfg):
    """pixel_values: [3, H, W] float in [-1, 1] (or normalized per the processor).

    Returns [256, 4096] float32 — the embeddings to splice at <|image|>.
    """
    device = pixel_values.device
    P      = cfg["patch_size"]          # 14
    D      = cfg["hidden_size"]         # 1024
    heads  = cfg["num_heads"]           # 16
    hd     = D // heads                 # 64
    eps    = cfg["rms_norm_eps"]
    limit  = cfg.get("swiglu_limit", 10.0)
    merge  = cfg["spatial_merge_size"]  # 2
    tpatch = cfg["temporal_patch_size"] # 2

    # --- patch embed: a still image is replicated across the temporal patch ---
    x = pixel_values.unsqueeze(0)                       # [1,3,H,W]
    x = x.unsqueeze(2).repeat(1, 1, tpatch, 1, 1)       # [1,3,T,H,W]
    x = F.conv3d(x, W["model.visual.patch_embed.proj.weight"],
                 W["model.visual.patch_embed.proj.bias"],
                 stride=(tpatch, P, P))                 # [1,1024,1,h,w]
    _, _, _, gh, gw = x.shape
    x = x.flatten(3).squeeze(2).squeeze(0).transpose(0, 1)   # [gh*gw, 1024]

    cos, sin = rope_2d(hd, gh, gw, device)

    for i in range(cfg["depth"]):
        b = f"model.visual.blocks.{i}."
        h = rms_norm(x, W[b + "norm1.weight"], eps)

        qkv = F.linear(h, W[b + "attn.qkv.weight"], W[b + "attn.qkv.bias"])
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(-1, heads, hd); k = k.view(-1, heads, hd); v = v.view(-1, heads, hd)
        q = rms_norm(q, W[b + "attn.q_norm.weight"], eps)
        k = rms_norm(k, W[b + "attn.k_norm.weight"], eps)
        q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)

        # full bidirectional attention -- a ViT has no causal mask
        a = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0))
        a = a.squeeze(0).transpose(0, 1).reshape(-1, D)
        x = x + F.linear(a, W[b + "attn.proj.weight"], W[b + "attn.proj.bias"])

        h = rms_norm(x, W[b + "norm2.weight"], eps)
        x = x + swiglu(h,
                       W[b + "mlp.gate_proj.weight"], W[b + "mlp.gate_proj.bias"],
                       W[b + "mlp.up_proj.weight"],   W[b + "mlp.up_proj.bias"],
                       W[b + "mlp.down_proj.weight"], W[b + "mlp.down_proj.bias"],
                       limit)

    x = rms_norm(x, W["model.visual.post_layernorm.weight"], eps)

    # --- spatial merge: 2x2 stride-2 conv, 1024 -> 4096, halving the grid ---
    x = x.transpose(0, 1).reshape(1, D, gh, gw)
    x = F.conv2d(x, W["model.visual.downsample.weight"],
                 W["model.visual.downsample.bias"], stride=merge)
    out_dim = x.shape[1]
    x = x.flatten(2).squeeze(0).transpose(0, 1)          # [(gh/2)*(gw/2), 4096]

    # --- merger: proj -> LayerNorm -> SwiGLU ---
    m = "model.visual.merger."
    x = F.linear(x, W[m + "proj.weight"])
    x = layer_norm(x, W[m + "post_projection_norm.weight"],
                   W[m + "post_projection_norm.bias"], eps)
    g = F.linear(x, W[m + "gate_proj.weight"]).clamp(max=limit)
    u = F.linear(x, W[m + "up_proj.weight"]).clamp(min=-limit, max=limit)
    x = F.linear(F.silu(g) * u, W[m + "down_proj.weight"])
    return x.float()


def vision_config(ckpt=CKPT):
    return json.load(open(os.path.join(ckpt, "config.json")))["vision_config"]


if __name__ == "__main__":
    cfg = vision_config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    W = load_vision_weights(device=dev)
    n = sum(t.numel() for t in W.values())
    print(f"loaded {len(W)} vision tensors, {n/1e6:.1f}M params")

    img = torch.zeros(3, cfg["image_size"], cfg["image_size"], device=dev)
    emb = encode_image(img, W, cfg)
    print(f"embeddings: {tuple(emb.shape)}  dtype={emb.dtype}")
    exp = (cfg["image_size"] // cfg["patch_size"] // cfg["spatial_merge_size"]) ** 2
    assert emb.shape == (exp, cfg["out_hidden_size"]), f"expected ({exp}, {cfg['out_hidden_size']})"
    print(f"finite: {torch.isfinite(emb).all().item()}  "
          f"mean {emb.mean():+.4f}  std {emb.std():.4f}  "
          f"absmax {emb.abs().max():.4f}")
