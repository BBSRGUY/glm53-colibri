"""Vision sidecar for glm53: turns request images into the engine's IMG_EMB file.

The tower is loaded once and kept resident. It defaults to the CPU because the
engine may already hold most of the VRAM for its expert tier -- set
VISION_DEVICE=cuda if you have left room for it (~1.2 GB).

Contract with the engine (glm53.c, img_emb_load):
    int32 pos0 | int32 n | int32 dim | n*dim float32
`pos0` is ignored by the engine, which locates the placeholder run itself by
scanning for image_token_id; it is written only so the file is self-describing.
"""
import base64, io, os, struct, threading, urllib.request

_LOCK = threading.Lock()
_STATE = {"weights": None, "cfg": None, "device": None}

TOKENS_PER_IMAGE = None   # filled from config on first load


def _device():
    return os.environ.get("VISION_DEVICE", "cpu")


def resolve_ckpt(model_dir):
    """Where the BF16 vision weights live.

    The converted container stores the tower quantized under colibri's own layout,
    so the sidecar reads the ORIGINAL checkpoint instead -- it is one shard, BF16,
    and avoids a dequantisation path that would need its own validation.
    VISION_CKPT overrides; otherwise the model dir is tried (works when serving
    straight from an unconverted checkpoint).
    """
    return os.environ.get("VISION_CKPT") or model_dir


def available(model_dir):
    """True when this build can actually encode images."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    ckpt = resolve_ckpt(model_dir)
    return (os.path.exists(os.path.join(ckpt, "config.json")) and
            os.path.exists(os.path.join(ckpt, "model.safetensors.index.json")))


def _load(model_dir):
    global TOKENS_PER_IMAGE
    import vision_tower as V
    ckpt = resolve_ckpt(model_dir)
    cfg = V.vision_config(ckpt)
    W = V.load_vision_weights(ckpt, device=_device())
    TOKENS_PER_IMAGE = (cfg["image_size"] // cfg["patch_size"] // cfg["spatial_merge_size"]) ** 2
    _STATE.update(weights=W, cfg=cfg, device=_device())
    return W, cfg


def tokens_per_image(ckpt):
    if _STATE["cfg"] is None:
        with _LOCK:
            if _STATE["cfg"] is None:
                _load(ckpt)
    return TOKENS_PER_IMAGE


def _decode(src):
    """src is a data: URI, an http(s) URL, or a local path. Returns a PIL image."""
    from PIL import Image
    if src.startswith("data:"):
        _, _, payload = src.partition(",")
        return Image.open(io.BytesIO(base64.b64decode(payload)))
    if src.startswith("http://") or src.startswith("https://"):
        with urllib.request.urlopen(src, timeout=20) as r:
            return Image.open(io.BytesIO(r.read()))
    return Image.open(src)


def encode(images, ckpt, out_path):
    """Encode `images` (list of sources) and write out_path. Returns row count.

    Multiple images concatenate in order, which matches the engine binding rows
    to the placeholder run in prompt order.
    """
    import numpy as np, torch
    import vision_tower as V

    with _LOCK:
        if _STATE["weights"] is None:
            _load(ckpt)
        W, cfg = _STATE["weights"], _STATE["cfg"]
        size = cfg["image_size"]
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]

        rows = []
        for src in images:
            from PIL import Image
            im = _decode(src).convert("RGB").resize((size, size), Image.BICUBIC)
            t = torch.from_numpy(np.asarray(im).copy()).float().div_(255.0).permute(2, 0, 1)
            t = ((t - mean) / std).to(_STATE["device"])
            rows.append(V.encode_image(t, W, cfg).cpu())
        emb = torch.cat(rows, dim=0).contiguous()

    # Match the text embedding scale. The tower's raw output is ~40x the magnitude of
    # embed_tokens rows, and image rows are spliced into the SAME residual stream, so
    # left unscaled they swamp it and the model falls back on language priors.
    # VISION_TARGET_L2 is the mean L2 of the container's embed_tokens rows (measured
    # 0.604 for glm53-int4); VISION_SCALE=1 disables the rescale for A/B testing.
    # A CONSTANT divisor, not per-image normalisation. The tower distinguishes a black
    # frame from a white one largely by magnitude (L2 25 vs 47); normalising each image
    # to a fixed L2 erases precisely that signal -- measured, after it made the model
    # call a black image "white". 41.7 is the ratio of the tower's mean row L2 to the
    # container's embed_tokens mean row L2, so relative differences survive.
    div = float(os.environ.get("VISION_DIV", "41.7"))
    if div > 0:
        emb = emb / div

    n, dim = emb.shape
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<iii", 0, n, dim))
        f.write(emb.numpy().astype("<f4").tobytes())
    os.replace(tmp, out_path)      # atomic: the engine must never read a partial file
    return n


def clear(out_path):
    """Remove the file so the next turn is text-only."""
    try:
        os.remove(out_path)
    except FileNotFoundError:
        pass
