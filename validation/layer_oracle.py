"""Numpy oracle for a whole GLM-5.3 decoder layer (KDA + dense MLP, mHC-wrapped).

Reproduces the block the engine runs for layers 0..2, where layer_types marks the
attention linear and first_k_dense_replace=3 makes the MLP dense:

    residual = streams                       [T, M, D]
    post,comb,collapsed = mhc(streams, attn) ; y = KDA(rmsnorm(collapsed, in_ln))
    streams = post (x) y + comb^T @ residual
    residual = streams
    post,comb,collapsed = mhc(streams, ffn)  ; y = SwiGLU_clamped(rmsnorm(collapsed, post_ln))
    streams = post (x) y + comb^T @ residual

The mHC half deliberately reproduces dsv4_mhc.h's bf16 rounding, so a mismatch here
means a real disagreement rather than the known 2^-8 float gap. KDA is the reference
already validated in kda_oracle.py.
"""
import struct
import sys

import numpy as np

H, HD, D, K, T = 4, 16, 24, 4, 5
M, INTER, ITERS = 4, 40, 20
import os
GATE_LB, EPS, HC_EPS = -5.0, 1e-5, 1e-6
SLIM = float(os.environ.get('SLIM','10.0'))
P, R, N = H * HD, HD, 2 * 4 + 4 * 4

rng = np.random.default_rng(777)
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
    'q': rnd(P, D), 'k': rnd(P, D), 'v': rnd(P, D), 'o': rnd(D, P),
    'cq': rnd(P, K), 'ck': rnd(P, K), 'cv': rnd(P, K),
    'fa': rnd(R, D), 'fb': rnd(P, R), 'ga': rnd(R, D), 'gb': rnd(P, R),
    'bp': rnd(H, D), 'dt': rnd(P, scale=0.2), 'A_log': rnd(H, scale=0.3),
    'onw': (1.0 + rnd(HD, scale=0.1)).astype(f32),
    'gate': rnd(INTER, D), 'up': rnd(INTER, D), 'down': rnd(D, INTER),
}
streams0 = rnd(T, M, D, scale=0.7)


def mhc(res_row, fn, scale, base):
    """res_row [M,D] -> post[M], comb[M,M], collapsed[D]; mirrors dsv4_mhc_pre."""
    flat = res_row.reshape(-1).astype(np.float64)
    inv = 1.0 / np.sqrt((flat ** 2).mean() + EPS)
    mixes = (fn.astype(np.float64) @ flat) * inv
    pre = sigmoid(mixes[:M] * scale[0] + base[:M]) + HC_EPS
    post = 2.0 * sigmoid(mixes[M:2 * M] * scale[1] + base[M:2 * M])
    cl = mixes[2 * M:].reshape(M, M) * scale[2] + base[2 * M:].reshape(M, M)
    e = np.exp(cl - cl.max(axis=1, keepdims=True))
    comb = (e / e.sum(axis=1, keepdims=True)).astype(f32) + HC_EPS
    comb = comb / (comb.sum(axis=0, keepdims=True) + HC_EPS)      # first col pass
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(axis=1, keepdims=True) + HC_EPS)
        comb = comb / (comb.sum(axis=0, keepdims=True) + HC_EPS)
    collapsed = bf16((pre[:, None].astype(f32) * res_row).sum(axis=0))
    return post.astype(f32), comb.astype(f32), collapsed


def mhc_post(blk, res_row, post, comb):
    """out[j,d] = post[j]*blk[d] + sum_i comb[i,j]*res_row[i,d]  (comb transposed)."""
    return bf16(post[:, None] * blk[None, :] + comb.T @ res_row)


def kda(x):
    qp, kp, vp = x @ W['q'].T, x @ W['k'].T, x @ W['v'].T
    outs = []
    for taps, arr in ((W['cq'], qp), (W['ck'], kp), (W['cv'], vp)):
        xp = np.concatenate([np.zeros((K - 1, P), f32), arr], axis=0)
        y = np.zeros_like(arr)
        for j in range(K):
            y += xp[j:j + len(arr)] * taps[:, j]
        outs.append(silu(y))
    q, k, v = outs
    g_low = (x @ W['fa'].T) @ W['fb'].T
    gfull = (x @ W['ga'].T) @ W['gb'].T
    beta = sigmoid(x @ W['bp'].T)
    A = np.exp(W['A_log'].astype(np.float64)).astype(f32)
    t = len(x)
    q, k, v = (a.reshape(t, H, HD) for a in (q, k, v))
    z = g_low.reshape(t, H, HD) + W['dt'].reshape(H, HD)
    alpha = np.exp(GATE_LB * sigmoid(A[None, :, None] * z)).astype(f32)
    qn = q / np.sqrt((q ** 2).sum(-1, keepdims=True) + 1e-6) * (HD ** -0.5)
    kn = k / np.sqrt((k ** 2).sum(-1, keepdims=True) + 1e-6)
    on = np.zeros((t, H, HD), f32)
    for h in range(H):
        S = np.zeros((HD, HD), f32)
        for i in range(t):
            S = S * alpha[i, h][:, None]
            kS = kn[i, h] @ S
            vt = (v[i, h] - kS) * beta[i, h]
            S = S + np.outer(kn[i, h], vt)
            on[i, h] = qn[i, h] @ S
    ms = np.mean(on.astype(np.float64) ** 2, axis=-1, keepdims=True)
    on = (on * (1.0 / np.sqrt(ms + EPS))).astype(f32) * W['onw']
    on = on * sigmoid(gfull.reshape(t, H, HD))
    return on.reshape(t, P) @ W['o'].T


def dense_mlp(x):
    g = np.minimum(x @ W['gate'].T, SLIM)                  # gate: upper bound only
    u = np.clip(x @ W['up'].T, -SLIM, SLIM)                # up: both sides
    return (silu(g) * u) @ W['down'].T


# ---- the assembled layer ----
st = streams0.copy()
# attention site
post = np.zeros((T, M), f32); comb = np.zeros((T, M, M), f32); coll = np.zeros((T, D), f32)
for s in range(T):
    post[s], comb[s], coll[s] = mhc(st[s], W['a_fn'], W['a_scale'], W['a_base'])
y = kda(rmsnorm(coll, W['in_ln'], EPS))
res = st.copy()
for s in range(T):
    st[s] = mhc_post(y[s], res[s], post[s], comb[s])
# FFN site
for s in range(T):
    post[s], comb[s], coll[s] = mhc(st[s], W['f_fn'], W['f_scale'], W['f_base'])
y = dense_mlp(rmsnorm(coll, W['post_ln'], EPS))
res = st.copy()
for s in range(T):
    st[s] = mhc_post(y[s], res[s], post[s], comb[s])

out = sys.argv[1] if len(sys.argv) > 1 else 'layer_fixture.bin'
with open(out, 'wb') as f:
    f.write(struct.pack('<8i4f', H, HD, D, K, T, M, INTER, ITERS,
                        GATE_LB, EPS, HC_EPS, SLIM))
    for key in ('in_ln', 'post_ln', 'a_fn', 'a_base', 'a_scale', 'f_fn', 'f_base', 'f_scale',
                'q', 'k', 'v', 'o', 'cq', 'ck', 'cv', 'fa', 'fb', 'ga', 'gb',
                'bp', 'dt', 'A_log', 'onw', 'gate', 'up', 'down'):
        f.write(np.ascontiguousarray(W[key], f32).tobytes())
    f.write(np.ascontiguousarray(streams0, f32).tobytes())
    f.write(np.ascontiguousarray(st, f32).tobytes())

print(f'wrote {out}: T={T} M={M} D={D} inter={INTER}')
print(f'  out range [{st.min():.5f}, {st.max():.5f}]  mean|out|={np.abs(st).mean():.5f}')
