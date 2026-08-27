"""Oracle for a GLM-5.3 MLA/DSA layer, using transformers' GlmMoeDsaAttention.

The attention half is the REAL module from transformers -- GLM-5.3's MLA is GLM-5.2's
with mla_use_nope / qk_rope_head_dim=0, and the module tolerates the zero-width rope
split (verified by probe_mla.py). index_topk is set well above the sequence length so
the DSA indexer selects every key, which is the regime colibri's engine also treats as
dense causal attention.

The mHC wrapper and the dense SwiGLU around it are numpy, as in the other layer oracles.
"""
import os
import struct
import sys

import numpy as np
import torch
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention

import os as _os
D = int(_os.environ.get("D",32)); T = int(_os.environ.get("T","5")); M, ITERS = 4, 20
NH=int(_os.environ.get("NH",2)); QL=int(_os.environ.get("QL",16)); KL=int(_os.environ.get("KL",8))
NOPE=int(_os.environ.get("NOPE",16)); VH=int(_os.environ.get("VH",16)); INTER=int(_os.environ.get("INTER",40))
EPS, HC_EPS = 1e-5, 1e-6
SLIM = float(os.environ.get('SLIM', '10.0'))
N = 2 * M + M * M
QH = NOPE                       # qk_rope_head_dim = 0
ASCALE = QH ** -0.5

rng = np.random.default_rng(31337)
f32 = np.float32


def rnd(*shape, scale=0.05):
    return (rng.standard_normal(shape) * scale).astype(f32)


def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))).astype(f32)


def silu(z):
    z = np.asarray(z, np.float64)
    return (z / (1.0 + np.exp(-z))).astype(f32)


def bf16(a):
    u = np.asarray(a, '<f4').view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def rmsnorm(x, w, eps):
    ms = np.mean(np.asarray(x, np.float64) ** 2, axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(ms + eps)).astype(f32)) * w


W = {
    'in_ln': (1.0 + rnd(D, scale=0.1)).astype(f32),
    'post_ln': (1.0 + rnd(D, scale=0.1)).astype(f32),
    'a_fn': rnd(N, M * D), 'a_base': rnd(N, scale=0.1), 'a_scale': rnd(3, scale=0.5),
    'f_fn': rnd(N, M * D), 'f_base': rnd(N, scale=0.1), 'f_scale': rnd(3, scale=0.5),
    'q_a': rnd(QL, D, scale=0.2), 'q_a_ln': (1.0 + rnd(QL, scale=0.1)).astype(f32),
    'q_b': rnd(NH * QH, QL, scale=0.2),
    'kv_a': rnd(KL, D, scale=0.2), 'kv_a_ln': (1.0 + rnd(KL, scale=0.1)).astype(f32),
    'kv_b': rnd(NH * (NOPE + VH), KL, scale=0.2),
    'o': rnd(D, NH * VH, scale=0.2),
    'gate': rnd(INTER, D), 'up': rnd(INTER, D), 'down': rnd(D, INTER),
}
streams0 = rnd(T, M, D, scale=0.7)


def mhc(res_row, fn, scale, base):
    flat = res_row.reshape(-1).astype(np.float64)
    inv = 1.0 / np.sqrt((flat ** 2).mean() + EPS)
    mixes = (fn.astype(np.float64) @ flat) * inv
    pre = sigmoid(mixes[:M] * scale[0] + base[:M]) + HC_EPS
    post = 2.0 * sigmoid(mixes[M:2 * M] * scale[1] + base[M:2 * M])
    cl = mixes[2 * M:].reshape(M, M) * scale[2] + base[2 * M:].reshape(M, M)
    e = np.exp(cl - cl.max(axis=1, keepdims=True))
    comb = (e / e.sum(axis=1, keepdims=True)).astype(f32) + HC_EPS
    comb = comb / (comb.sum(axis=0, keepdims=True) + HC_EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(axis=1, keepdims=True) + HC_EPS)
        comb = comb / (comb.sum(axis=0, keepdims=True) + HC_EPS)
    collapsed = bf16((pre[:, None].astype(f32) * res_row).sum(axis=0))
    return post.astype(f32), comb.astype(f32), collapsed


def mhc_post(blk, res_row, post, comb):
    return bf16(post[:, None] * blk[None, :] + comb.T @ res_row)


cfg = GlmMoeDsaConfig(
    hidden_size=D, num_attention_heads=NH, num_key_value_heads=NH,
    q_lora_rank=QL, kv_lora_rank=KL, qk_nope_head_dim=NOPE, qk_rope_head_dim=0,
    v_head_dim=VH, num_hidden_layers=1,
    index_topk=4096, index_n_heads=2, index_head_dim=8,
    indexer_types=["full"], attention_bias=False,
)
# WITHOUT this the module is NON-CAUSAL: GlmMoeDsaAttention.forward only builds the
# causal/DSA mask when config._attn_implementation is in ("eager","sdpa"), and a
# standalone module leaves it None -- so attention_mask stays None and every query
# attends to every key. Verified: None matches a non-causal reference, "eager" matches
# a causal one.
cfg._attn_implementation = "eager"
attn = GlmMoeDsaAttention(cfg, 0).double().eval()
with torch.no_grad():
    attn.q_a_proj.weight.copy_(torch.from_numpy(W['q_a']).double())
    attn.q_a_layernorm.weight.copy_(torch.from_numpy(W['q_a_ln']).double())
    attn.q_b_proj.weight.copy_(torch.from_numpy(W['q_b']).double())
    attn.kv_a_proj_with_mqa.weight.copy_(torch.from_numpy(W['kv_a']).double())
    attn.kv_a_layernorm.weight.copy_(torch.from_numpy(W['kv_a_ln']).double())
    attn.kv_b_proj.weight.copy_(torch.from_numpy(W['kv_b']).double())
    attn.o_proj.weight.copy_(torch.from_numpy(W['o']).double())


def mla(x):
    xt = torch.from_numpy(np.ascontiguousarray(x)).double()[None]
    pos = torch.arange(len(x))[None]
    cos = torch.ones(1, len(x), 1, dtype=torch.float64)
    sin = torch.zeros(1, len(x), 1, dtype=torch.float64)
    with torch.no_grad():
        out = attn(xt, position_embeddings=(cos, sin), position_ids=pos, attention_mask=None)[0]
    return out[0].numpy().astype(f32)


def dense_mlp(x):
    g = np.minimum(x @ W['gate'].T, SLIM)
    u = np.clip(x @ W['up'].T, -SLIM, SLIM)
    return (silu(g) * u) @ W['down'].T


st = streams0.copy()
post = np.zeros((T, M), f32); comb = np.zeros((T, M, M), f32); coll = np.zeros((T, D), f32)
for s in range(T):
    post[s], comb[s], coll[s] = mhc(st[s], W['a_fn'], W['a_scale'], W['a_base'])
y = mla(rmsnorm(coll, W['in_ln'], EPS))
res = st.copy()
for s in range(T):
    st[s] = mhc_post(y[s], res[s], post[s], comb[s])
for s in range(T):
    post[s], comb[s], coll[s] = mhc(st[s], W['f_fn'], W['f_scale'], W['f_base'])
y = dense_mlp(rmsnorm(coll, W['post_ln'], EPS))
res = st.copy()
for s in range(T):
    st[s] = mhc_post(y[s], res[s], post[s], comb[s])

out = sys.argv[1] if len(sys.argv) > 1 else 'mla_fixture.bin'
with open(out, 'wb') as f:
    f.write(struct.pack('<10i4f', D, T, M, ITERS, NH, QL, KL, NOPE, VH, INTER,
                        EPS, HC_EPS, SLIM, ASCALE))
    for key in ('in_ln', 'post_ln', 'a_fn', 'a_base', 'a_scale', 'f_fn', 'f_base', 'f_scale',
                'q_a', 'q_a_ln', 'q_b', 'kv_a', 'kv_a_ln', 'kv_b', 'o',
                'gate', 'up', 'down'):
        f.write(np.ascontiguousarray(W[key], f32).tobytes())
    f.write(np.ascontiguousarray(streams0, f32).tobytes())
    f.write(np.ascontiguousarray(st, f32).tobytes())

print(f'wrote {out}: T={T} heads={NH} q_lora={QL} kv_lora={KL} nope={NOPE} v_head={VH}')
print(f'  attn_scale={ASCALE}  out range [{st.min():.5f}, {st.max():.5f}]  mean|out|={np.abs(st).mean():.5f}')
