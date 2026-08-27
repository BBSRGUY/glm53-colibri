"""Quantized-path oracle: a GLM-5.3 sparse MoE layer with int4-gs64 experts.

Mirrors the Phase-1 shipping recipe -- routed experts int4-gs64, every resident matrix
int8 -- and is built so a mismatch can only come from the ENGINE's quantized kernels:

  * weights are quantized with convert_fp8_to_int4.py's OWN functions (imported, not
    reimplemented), so the packing the engine reads is the packing the converter emits;
  * the expected output is computed from the DEQUANTIZED weights, so quantization error
    is common to both sides and cancels;
  * therefore engine-vs-oracle difference == (int4/int8 matmul kernel) - (dequant then
    f32 matmul). Anything above float noise is a kernel bug, not a rounding artifact.

Layout per quantized tensor: packed uint8 bytes, then f32 scales
(int8: O per-row scales; int4-gs64: O*ceil(I/64) group scales).
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join('..', 'colibri', 'c', 'tools'))
from convert_fp8_to_int4 import quant_int4_grouped, quant_int8   # noqa: E402

H, HD, D, K, T = 4, 16, 64, 4, 5
M, ITERS = 4, 20
E, TOPK, MI, NSH = 8, 2, 64, 1
GATE_LB, EPS, HC_EPS = -5.0, 1e-5, 1e-6
SLIM = float(os.environ.get('SLIM', '10.0'))
RSCALE, NORM_TOPK, GS = 2.5, 1, 64
P, R, N, SI = H * HD, HD, 2 * M + M * M, MI * NSH

rng = np.random.default_rng(90210)
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


# ---- quantize with the converter's own math, and give back the dequantized view ----
def q8(w):
    q, s = quant_int8(w.astype(f32), 8)
    deq = q.view(np.int8).reshape(w.shape).astype(f32) * s.reshape(-1, 1)
    return ('q8', q, s.astype(f32)), deq


def q4g(w):
    O, I = w.shape
    q, s = quant_int4_grouped(w.astype(f32), 4, GS)
    b = q.reshape(O, (I + 1) // 2)
    lo = (b & 0xF).astype(np.int32) - 8
    hi = (b >> 4).astype(np.int32) - 8
    d = np.zeros((O, I), f32)
    d[:, 0::2] = lo[:, :len(range(0, I, 2))]
    d[:, 1::2] = hi[:, :len(range(1, I, 2))]
    ng = (I + GS - 1) // GS
    deq = d * np.repeat(s.reshape(O, ng), GS, axis=1)[:, :I]
    return ('q4', q, s.astype(f32)), deq


RAW, DEQ = {}, {}


def add(name, w, how):
    enc, deq = how(w)
    RAW[name] = enc
    DEQ[name] = deq


W = {
    'in_ln': (1.0 + rnd(D, scale=0.1)).astype(f32),
    'post_ln': (1.0 + rnd(D, scale=0.1)).astype(f32),
    'a_fn': rnd(N, M * D), 'a_base': rnd(N, scale=0.1), 'a_scale': rnd(3, scale=0.5),
    'f_fn': rnd(N, M * D), 'f_base': rnd(N, scale=0.1), 'f_scale': rnd(3, scale=0.5),
    'cq': rnd(P, K), 'ck': rnd(P, K), 'cv': rnd(P, K),
    'fa': rnd(R, D), 'fb': rnd(P, R), 'ga': rnd(R, D), 'gb': rnd(P, R),
    'bp': rnd(H, D), 'dt': rnd(P, scale=0.2), 'A_log': rnd(H, scale=0.3),
    'onw': (1.0 + rnd(HD, scale=0.1)).astype(f32),
    'router': rnd(E, D, scale=0.3), 'rbias': rnd(E, scale=0.4),
}
# int8 resident matrices
for nm, shape in (('q', (P, D)), ('k', (P, D)), ('v', (P, D)), ('o', (D, P)),
                  ('shg', (SI, D)), ('shu', (SI, D)), ('shd', (D, SI))):
    add(nm, rnd(*shape, scale=0.2), q8)
# int4-gs64 routed experts
EXP_RAW, EXP_DEQ = [], []
for _ in range(E):
    trip_raw, trip_deq = [], []
    for shape in ((MI, D), (MI, D), (D, MI)):
        enc, deq = q4g(rnd(*shape, scale=0.2))
        trip_raw.append(enc); trip_deq.append(deq)
    EXP_RAW.append(trip_raw); EXP_DEQ.append(trip_deq)

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
    return post.astype(f32), comb.astype(f32), bf16((pre[:, None].astype(f32) * res_row).sum(axis=0))


def mhc_post(blk, res_row, post, comb):
    return bf16(post[:, None] * blk[None, :] + comb.T @ res_row)


def swiglu(x, wg, wu, wd):
    g = np.minimum(x @ wg.T, SLIM)
    u = np.clip(x @ wu.T, -SLIM, SLIM)
    return (silu(g) * u) @ wd.T


def kda(x):
    qp, kp, vp = x @ DEQ['q'].T, x @ DEQ['k'].T, x @ DEQ['v'].T
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
    return on.reshape(t, P) @ DEQ['o'].T


def moe(x):
    out = swiglu(x, DEQ['shg'], DEQ['shu'], DEQ['shd'])
    sig = sigmoid(x @ W['router'].T)
    choice = sig + W['rbias']
    for s in range(len(x)):
        order = np.argsort(-choice[s].astype(np.float64), kind='stable')[:TOPK]
        w = sig[s][order].astype(np.float64)
        if NORM_TOPK:
            w = w / (w.sum() + 1e-20)
        w = (w * RSCALE).astype(f32)
        for e, wk in zip(order, w):
            g, u, d = EXP_DEQ[e]
            out[s] += wk * swiglu(x[s:s + 1], g, u, d)[0]
    return out


st = streams0.copy()
post = np.zeros((T, M), f32); comb = np.zeros((T, M, M), f32); coll = np.zeros((T, D), f32)
for s in range(T):
    post[s], comb[s], coll[s] = mhc(st[s], W['a_fn'], W['a_scale'], W['a_base'])
y = kda(rmsnorm(coll, W['in_ln'], EPS))
res = st.copy()
for s in range(T):
    st[s] = mhc_post(y[s], res[s], post[s], comb[s])
for s in range(T):
    post[s], comb[s], coll[s] = mhc(st[s], W['f_fn'], W['f_scale'], W['f_base'])
y = moe(rmsnorm(coll, W['post_ln'], EPS))
res = st.copy()
for s in range(T):
    st[s] = mhc_post(y[s], res[s], post[s], comb[s])

out = sys.argv[1] if len(sys.argv) > 1 else 'moeq_fixture.bin'
with open(out, 'wb') as f:
    f.write(struct.pack('<13i5f', H, HD, D, K, T, M, ITERS, E, TOPK, MI, NSH, NORM_TOPK, GS,
                        GATE_LB, EPS, HC_EPS, SLIM, RSCALE))
    for key in ('in_ln', 'post_ln', 'a_fn', 'a_base', 'a_scale', 'f_fn', 'f_base', 'f_scale'):
        f.write(np.ascontiguousarray(W[key], f32).tobytes())
    for nm in ('q', 'k', 'v'):
        _, q, s = RAW[nm]; f.write(q.tobytes()); f.write(s.tobytes())
    _, q, s = RAW['o']; f.write(q.tobytes()); f.write(s.tobytes())
    for key in ('cq', 'ck', 'cv', 'fa', 'fb', 'ga', 'gb', 'bp', 'dt', 'A_log', 'onw',
                'router', 'rbias'):
        f.write(np.ascontiguousarray(W[key], f32).tobytes())
    for nm in ('shg', 'shu', 'shd'):
        _, q, s = RAW[nm]; f.write(q.tobytes()); f.write(s.tobytes())
    for trip in EXP_RAW:
        for _, q, s in trip:
            f.write(q.tobytes()); f.write(s.tobytes())
    f.write(np.ascontiguousarray(streams0, f32).tobytes())
    f.write(np.ascontiguousarray(st, f32).tobytes())

print(f'wrote {out}: experts int4-gs{GS}, resident int8, D={D} moe_inter={MI} E={E}')
print(f'  out range [{st.min():.5f}, {st.max():.5f}]  mean|out|={np.abs(st).mean():.5f}')
