"""Numpy oracle for glm53.c's kda_forward.

Adapted from colibri's own KDA reference (c/tools/k3_ref.py :: KdaRef.kda, itself
derived from Moonshot's modeling_kimi_linear.py) with exactly one substitution:
GLM-5.3 factors the output gate low-rank (g_a_proj @ g_b_proj, rank = head_dim)
where Kimi has a dense g_proj. The recurrence, the stateful conv, the L2/RMS norms
and the decay are taken over unchanged, so this is not a re-derivation of the C code.

Writes a fixture the engine consumes via GLM53_KDA_SELFTEST.
"""
import struct
import sys

import numpy as np

import os
H  = int(os.environ.get("H",4)); HD = int(os.environ.get("HD",16))
D  = int(os.environ.get("D",24)); K  = int(os.environ.get("K",4)); T = int(os.environ.get("T",6))          # small, but every path is exercised
GATE_LB, EPS = -5.0, 1e-5
P, R = H * HD, HD

rng = np.random.default_rng(20260827)
f32 = np.float32


def rnd(*shape, scale=0.05):
    return (rng.standard_normal(shape) * scale).astype(f32)


def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z.astype(np.float64)))).astype(f32)


def silu(z):
    return (z.astype(np.float64) / (1.0 + np.exp(-z.astype(np.float64)))).astype(f32)


W = {
    'q': rnd(P, D), 'k': rnd(P, D), 'v': rnd(P, D), 'o': rnd(D, P),
    'cq': rnd(P, K), 'ck': rnd(P, K), 'cv': rnd(P, K),
    'fa': rnd(R, D), 'fb': rnd(P, R), 'ga': rnd(R, D), 'gb': rnd(P, R),
    'bp': rnd(H, D), 'dt': rnd(P, scale=0.2), 'A_log': rnd(H, scale=0.3),
    'onw': (1.0 + rnd(HD, scale=0.1)).astype(f32),
}
x = rnd(T, D, scale=0.8)


def kda(x):
    qp, kp, vp = x @ W['q'].T, x @ W['k'].T, x @ W['v'].T
    outs = []
    for taps, arr in ((W['cq'], qp), (W['ck'], kp), (W['cv'], vp)):
        pad = np.zeros((K - 1, P), f32)
        xp = np.concatenate([pad, arr], axis=0)
        y = np.zeros_like(arr)
        for j in range(K):
            y += xp[j:j + T] * taps[:, j]
        outs.append(silu(y))
    q, k, v = outs

    g_low = (x @ W['fa'].T) @ W['fb'].T                      # decay LoRA
    gfull = (x @ W['ga'].T) @ W['gb'].T                      # GLM-5.3: gate is low-rank too
    beta = sigmoid(x @ W['bp'].T)
    A = np.exp(W['A_log'].astype(np.float64)).astype(f32)

    q, k, v = (a.reshape(T, H, HD) for a in (q, k, v))
    z = g_low.reshape(T, H, HD) + W['dt'].reshape(H, HD)
    alpha = np.exp(GATE_LB * sigmoid(A[None, :, None] * z)).astype(f32)
    qn = q / np.sqrt((q ** 2).sum(-1, keepdims=True) + 1e-6) * (HD ** -0.5)
    kn = k / np.sqrt((k ** 2).sum(-1, keepdims=True) + 1e-6)

    on = np.zeros((T, H, HD), f32)
    for h in range(H):
        S = np.zeros((HD, HD), f32)                          # S[k, v]
        for t in range(T):
            S = S * alpha[t, h][:, None]
            kS = kn[t, h] @ S
            vt = (v[t, h] - kS) * beta[t, h]
            S = S + np.outer(kn[t, h], vt)
            on[t, h] = qn[t, h] @ S
    ms = np.mean(on.astype(np.float64) ** 2, axis=-1, keepdims=True)
    on = (on * (1.0 / np.sqrt(ms + EPS))).astype(f32) * W['onw']
    on = on * sigmoid(gfull.reshape(T, H, HD))
    return on.reshape(T, P) @ W['o'].T


y = kda(x)

out = sys.argv[1] if len(sys.argv) > 1 else 'kda_fixture.bin'
with open(out, 'wb') as f:
    f.write(struct.pack('<5i2f', H, HD, D, K, T, GATE_LB, EPS))
    for key in ('q', 'k', 'v', 'o', 'cq', 'ck', 'cv', 'fa', 'fb', 'ga', 'gb',
                'bp', 'dt', 'A_log', 'onw'):
        f.write(np.ascontiguousarray(W[key], f32).tobytes())
    f.write(np.ascontiguousarray(x, f32).tobytes())
    f.write(np.ascontiguousarray(y, f32).tobytes())

print(f'wrote {out}: H={H} hd={HD} D={D} K={K} T={T}  P={P} R={R}')
print(f'  out range [{y.min():.5f}, {y.max():.5f}]  mean|out|={np.abs(y).mean():.5f}')
