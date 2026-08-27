"""Oracle for glm53.c's mHC path, using transformers' DeepseekV4HyperConnection.

GLM-5.3's hyper-connections are DeepSeek-V4's unchanged (llama.cpp's glm5next PR
subclasses DeepseekV4HyperConnection with a bare `pass`), and that module ships in
the installed transformers -- so this is a real third-party oracle, not a second
implementation by the same author.

Emits a little-endian binary fixture:
  [i32 M][i32 D][i32 S][i32 iters][f32 eps][f32 rms_eps]
  fn      f32 [(2+M)*M, M*D]
  base    f32 [(2+M)*M]
  scale   f32 [3]
  streams f32 [S, M, D]      (layer input, the residual)
  blkout  f32 [S, D]         (stand-in for the attention/MLP output)
  --- expected ---
  post      f32 [S, M]
  comb      f32 [S, M, M]
  collapsed f32 [S, D]
  updated   f32 [S, M, D]    streams after post (x) blkout + comb^T @ streams
  seeded    f32 [S, M, D]    broadcast of `embed` into M streams
  meaned    f32 [S, D]       mean-collapse of `updated`
  embed     f32 [S, D]
"""
import struct
import sys

import torch
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection

import os
M = int(os.environ.get("M",4)); D = int(os.environ.get("D",32)); S = int(os.environ.get("S",3))
ITERS, EPS, RMS_EPS = 20, 1e-6, 1e-5

torch.manual_seed(1234)


class Cfg:
    hc_mult = M
    hc_sinkhorn_iters = ITERS
    hc_eps = EPS
    rms_norm_eps = RMS_EPS
    hidden_size = D


hc = DeepseekV4HyperConnection(Cfg()).double()
mix = (2 + M) * M
with torch.no_grad():
    hc.fn.copy_(torch.randn(mix, M * D, dtype=torch.float64) * 0.05)
    hc.base.copy_(torch.randn(mix, dtype=torch.float64) * 0.1)
    hc.scale.copy_(torch.randn(3, dtype=torch.float64) * 0.5)

streams = torch.randn(1, S, M, D, dtype=torch.float64) * 0.7
blkout = torch.randn(1, S, D, dtype=torch.float64) * 0.3
embed = torch.randn(1, S, D, dtype=torch.float64) * 0.9

with torch.no_grad():
    post, comb, collapsed = hc(streams)
    # DeepseekV4DecoderLayer.forward, attention site:
    #   post.unsqueeze(-1) * blk.unsqueeze(-2) + matmul(comb.transpose(-1,-2), streams)
    updated = post.unsqueeze(-1) * blkout.unsqueeze(-2) + torch.matmul(
        comb.transpose(-1, -2), streams)
    # DeepseekV4Model.forward seeding, and llama.cpp's glm5next_hc_mean collapse
    seeded = embed.unsqueeze(2).expand(-1, -1, M, -1).contiguous()
    meaned = updated.mean(dim=2)


def w(f, t):
    f.write(t.detach().to(torch.float32).contiguous().numpy().tobytes())


out = sys.argv[1] if len(sys.argv) > 1 else 'mhc_fixture.bin'
with open(out, 'wb') as f:
    f.write(struct.pack('<4i2f', M, D, S, ITERS, EPS, RMS_EPS))
    for t in (hc.fn, hc.base, hc.scale, streams[0], blkout[0]):
        w(f, t)
    for t in (post[0], comb[0], collapsed[0], updated[0], seeded[0], meaned[0], embed[0]):
        w(f, t)

print(f'wrote {out}: M={M} D={D} S={S} iters={ITERS}')
print(f'  post     range [{post.min():.4f}, {post.max():.4f}]  (expect [0,2])')
print(f'  comb row sums   {comb[0, 0].sum(-1).tolist()}')
print(f'  comb col sums   {comb[0, 0].sum(-2).tolist()}   (Sinkhorn -> ~1)')
print(f'  collapsed[0,:4] {collapsed[0, 0, :4].tolist()}')
