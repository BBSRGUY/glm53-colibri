"""Full-stack numpy oracle for the tiny GLM-5.3 model, in colibri's TF harness format.

Composes the pieces validated individually earlier -- mHC (with its bf16 rounding),
KDA, dense-causal NoPE MLA, clamped SwiGLU, sigmoid/noaux_tc MoE -- into the whole
forward: embed -> broadcast to hc_mult streams -> 4 layers -> mean-collapse -> final
norm -> lm_head -> argmax per position.

Writes ref_glm53.json with prompt_ids / full_ids / tf_pred, consumed by
  REF=ref_glm53.json TF=1 SNAP=<converted> ./glm53

The engine runs QUANTIZED weights (experts int4-gs64, resident int8, and int8
activations in the expert path) while this runs f32, so exact agreement on every
position is not expected -- near-ties will flip. Broad agreement is the signal;
wholesale disagreement would mean a mis-wired stack.
"""
import json
import os
import sys

import numpy as np

SNAP = sys.argv[1] if len(sys.argv) > 1 else 'glm53_tiny'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'ref_glm53.json'

cfg = json.load(open(os.path.join(SNAP, 'config.json')))['text_config']
W = np.load(os.path.join(SNAP, '_oracle_weights.npy'), allow_pickle=True).item()

D = cfg['hidden_size']; NL = cfg['num_hidden_layers']; VOCAB = cfg['vocab_size']
NH = cfg['num_attention_heads']; QL = cfg['q_lora_rank']; KL = cfg['kv_lora_rank']
NOPE = cfg['qk_nope_head_dim']; VH = cfg['v_head_dim']
E = cfg['n_routed_experts']; TOPK = cfg['num_experts_per_tok']
MI = cfg['moe_intermediate_size']; NSH = cfg['n_shared_experts']
FIRST_DENSE = cfg['first_k_dense_replace']; DENSE_I = cfg['intermediate_size']
EPS = cfg['rms_norm_eps']; SLIM = cfg['swiglu_limit']; RSCALE = cfg['routed_scaling_factor']
NORM_TOPK = cfg['norm_topk_prob']
HC = cfg['hc_mult']; ITERS = cfg['hc_sinkhorn_iters']; HC_EPS = cfg['hc_eps']
LT = cfg['layer_types']
IDX_NH = cfg.get('index_n_heads', 0); IDX_HD = cfg.get('index_head_dim', 0)
IDX_TOPK = cfg.get('index_topk', 1 << 30); KPOOL = cfg.get('index_kpool', 0)
KP_ON = bool(cfg.get('index_kpool_compress', False))
KP_TAIL = bool(cfg.get('index_kpool_always_select_tail', False))
la = cfg['linear_attn_config']
KH, KHD, CK, GATE_LB = la['num_heads'], la['head_dim'], la['short_conv_kernel_size'], la['gate_lower_bound']
P, R = KH * KHD, KHD
LM = 'model.language_model.'
f32 = np.float32


def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))).astype(f32)


def silu(z):
    z = np.asarray(z, np.float64)
    return (z / (1.0 + np.exp(-z))).astype(f32)


def bf16(a):
    u = np.ascontiguousarray(a, '<f4').view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32).reshape(np.shape(a))


def rmsnorm(x, w):
    ms = np.mean(np.asarray(x, np.float64) ** 2, axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(ms + EPS)).astype(f32)) * w


def swiglu(x, wg, wu, wd):
    g = np.minimum(x @ wg.T, SLIM)
    u = np.clip(x @ wu.T, -SLIM, SLIM)
    return (silu(g) * u) @ wd.T


def mhc(res_row, fn, scale, base):
    flat = res_row.reshape(-1).astype(np.float64)
    inv = 1.0 / np.sqrt((flat ** 2).mean() + EPS)
    mixes = (fn.astype(np.float64) @ flat) * inv
    pre = sigmoid(mixes[:HC] * scale[0] + base[:HC]) + HC_EPS
    post = 2.0 * sigmoid(mixes[HC:2 * HC] * scale[1] + base[HC:2 * HC])
    cl = mixes[2 * HC:].reshape(HC, HC) * scale[2] + base[2 * HC:].reshape(HC, HC)
    e = np.exp(cl - cl.max(axis=1, keepdims=True))
    comb = (e / e.sum(axis=1, keepdims=True)).astype(f32) + HC_EPS
    comb = comb / (comb.sum(axis=0, keepdims=True) + HC_EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(axis=1, keepdims=True) + HC_EPS)
        comb = comb / (comb.sum(axis=0, keepdims=True) + HC_EPS)
    return post.astype(f32), comb.astype(f32), bf16((pre[:, None].astype(f32) * res_row).sum(axis=0))


def mhc_post(blk, res_row, post, comb):
    return bf16(post[:, None] * blk[None, :] + comb.T @ res_row)


def kda(li, x):
    p = f'{LM}layers.{li}.self_attn.'
    T = len(x)
    qp, kp, vp = (x @ W[p + f'{t}_proj.weight'].T for t in ('q', 'k', 'v'))
    outs = []
    for t, arr in (('q', qp), ('k', kp), ('v', vp)):
        taps = W[p + f'{t}_conv1d.weight'].reshape(P, CK)
        xp = np.concatenate([np.zeros((CK - 1, P), f32), arr], axis=0)
        y = np.zeros_like(arr)
        for j in range(CK):
            y += xp[j:j + T] * taps[:, j]
        outs.append(silu(y))
    q, k, v = outs
    g_low = (x @ W[p + 'f_a_proj.weight'].T) @ W[p + 'f_b_proj.weight'].T
    gfull = (x @ W[p + 'g_a_proj.weight'].T) @ W[p + 'g_b_proj.weight'].T
    beta = sigmoid(x @ W[p + 'b_proj.weight'].T)
    A = np.exp(W[p + 'A_log'].astype(np.float64)).astype(f32)
    onw = W[p + 'o_norm.weight']
    q, k, v = (a.reshape(T, KH, KHD) for a in (q, k, v))
    z = g_low.reshape(T, KH, KHD) + W[p + 'dt_bias'].reshape(KH, KHD)
    alpha = np.exp(GATE_LB * sigmoid(A[None, :, None] * z)).astype(f32)
    qn = q / np.sqrt((q ** 2).sum(-1, keepdims=True) + 1e-6) * (KHD ** -0.5)
    kn = k / np.sqrt((k ** 2).sum(-1, keepdims=True) + 1e-6)
    on = np.zeros((T, KH, KHD), f32)
    for h in range(KH):
        S = np.zeros((KHD, KHD), f32)
        for i in range(T):
            S = S * alpha[i, h][:, None]
            kS = kn[i, h] @ S
            vt = (v[i, h] - kS) * beta[i, h]
            S = S + np.outer(kn[i, h], vt)
            on[i, h] = qn[i, h] @ S
    ms = np.mean(on.astype(np.float64) ** 2, axis=-1, keepdims=True)
    on = (on * (1.0 / np.sqrt(ms + EPS))).astype(f32) * onw
    on = on * sigmoid(gfull.reshape(T, KH, KHD))
    return on.reshape(T, P) @ W[p + 'o_proj.weight'].T



def dsa_select(li, x, T):
    """GLM-5.3 k-pool DSA: which cells each query row may attend to.

    Pool p covers cells p*r .. p*r+r-1.  Its key is a per-channel convex mix
    softmax_over_members(gate + ape) . member keys.  Pools are scored with the indexer
    query, every cell inherits its pool's score, the incomplete tail is always visible,
    and the top (index_topk + r - 1) cells survive.  No RoPE anywhere -- GLM-5.3 is
    nope-only and the ape is the intra-pool ordering signal.
    """
    p = f'{LM}layers.{li}.self_attn.'
    q_ = p + 'indexer.'
    r = KPOOL
    qres = rmsnorm(x @ W[p + 'q_a_proj.weight'].T, W[p + 'q_a_layernorm.weight'])
    qi = (qres @ W[q_ + 'wq_b.weight'].T).reshape(T, IDX_NH, IDX_HD)
    kk = x @ W[q_ + 'wk.weight'].T                                   # [T, IDX_HD]
    mu = kk.mean(-1, keepdims=True); var = kk.var(-1, keepdims=True)  # LayerNorm, eps 1e-6
    kk = ((kk - mu) / np.sqrt(var + 1e-6)) * W[q_ + 'k_norm.weight'] + W[q_ + 'k_norm.bias']
    gg = x @ W[q_ + 'index_kpool_compress_gate'].T                    # [T, IDX_HD]
    ape = W[q_ + 'index_kpool_compress_ape']                          # [r, IDX_HD]
    wp = x @ W[q_ + 'weights_proj.weight'].T                          # [T, IDX_NH]

    keep = np.zeros((T, T), bool)
    for s in range(T):
        nk = s + 1
        if nk <= IDX_TOPK:                     # indexer inert: everything visible
            keep[s, :nk] = True
            continue
        npool = nk // r
        tail0 = npool * r
        sc = np.full(nk, -np.inf)
        if npool:
            gm = gg[:tail0].reshape(npool, r, IDX_HD) + ape[None, :, :]
            gm = gm - gm.max(axis=1, keepdims=True)
            e = np.exp(gm)
            w = e / e.sum(axis=1, keepdims=True)
            pool = (w * kk[:tail0].reshape(npool, r, IDX_HD)).sum(axis=1)   # [npool, IDX_HD]
            d0 = (qi[s] @ pool.T) * (IDX_HD ** -0.5)                        # [IDX_NH, npool]
            ps = (np.maximum(d0, 0) * wp[s][:, None]).sum(0) * (IDX_NH ** -0.5)
            sc[:tail0] = np.repeat(ps, r)
        sc[tail0:nk] = np.inf if KP_TAIL else 0.0
        want = min(nk, IDX_TOPK + r - 1)
        sel = np.argsort(-sc, kind='stable')[:want]
        keep[s, sel] = True
    return keep


def mla(li, x):
    p = f'{LM}layers.{li}.self_attn.'
    T = len(x)
    q = rmsnorm(x @ W[p + 'q_a_proj.weight'].T, W[p + 'q_a_layernorm.weight'])
    q = (q @ W[p + 'q_b_proj.weight'].T).reshape(T, NH, NOPE)
    ckv = rmsnorm(x @ W[p + 'kv_a_proj_with_mqa.weight'].T, W[p + 'kv_a_layernorm.weight'])
    kv = (ckv @ W[p + 'kv_b_proj.weight'].T).reshape(T, NH, NOPE + VH)
    k, v = kv[..., :NOPE], kv[..., NOPE:]
    keep_mask = dsa_select(li, x, T) if KP_ON else None
    out = np.zeros((T, NH, VH), f32)
    for h in range(NH):
        s = (q[:, h].astype(np.float64) @ k[:, h].astype(np.float64).T) * (NOPE ** -0.5)
        s = s + np.triu(np.full((T, T), -np.inf), 1)
        if keep_mask is not None:
            s = np.where(keep_mask, s, -np.inf)
        pr = np.exp(s - s.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        out[:, h] = (pr @ v[:, h].astype(np.float64)).astype(f32)
    return out.reshape(T, NH * VH) @ W[p + 'o_proj.weight'].T


def moe(li, x):
    p = f'{LM}layers.{li}.mlp.'
    out = swiglu(x, W[p + 'shared_experts.gate_proj.weight'],
                 W[p + 'shared_experts.up_proj.weight'],
                 W[p + 'shared_experts.down_proj.weight'])
    sig = sigmoid(x @ W[p + 'gate.weight'].T)
    choice = sig + W[p + 'gate.e_score_correction_bias']
    for s in range(len(x)):
        order = np.argsort(-choice[s].astype(np.float64), kind='stable')[:TOPK]
        w = sig[s][order].astype(np.float64)
        if NORM_TOPK:
            w = w / (w.sum() + 1e-20)
        w = (w * RSCALE).astype(f32)
        for e, wk in zip(order, w):
            q = f'{p}experts.{e}.'
            out[s] += wk * swiglu(x[s:s + 1], W[q + 'gate_proj.weight'],
                                  W[q + 'up_proj.weight'], W[q + 'down_proj.weight'])[0]
    return out


def forward(ids):
    T = len(ids)
    x = W[LM + 'embed_tokens.weight'][np.asarray(ids)].astype(f32)
    st = np.repeat(x[:, None, :], HC, axis=1).copy()            # broadcast seed
    for li in range(NL):
        p = f'{LM}layers.{li}.'
        post = np.zeros((T, HC), f32); comb = np.zeros((T, HC, HC), f32)
        coll = np.zeros((T, D), f32)
        for s in range(T):
            post[s], comb[s], coll[s] = mhc(st[s], W[p + 'hc_attn_fn'],
                                            W[p + 'hc_attn_scale'], W[p + 'hc_attn_base'])
        h = rmsnorm(coll, W[p + 'input_layernorm.weight'])
        y = kda(li, h) if LT[li] == 'linear_attention' else mla(li, h)
        res = st.copy()
        for s in range(T):
            st[s] = mhc_post(y[s], res[s], post[s], comb[s])
        for s in range(T):
            post[s], comb[s], coll[s] = mhc(st[s], W[p + 'hc_ffn_fn'],
                                            W[p + 'hc_ffn_scale'], W[p + 'hc_ffn_base'])
        h = rmsnorm(coll, W[p + 'post_attention_layernorm.weight'])
        if li < FIRST_DENSE:
            y = swiglu(h, W[p + 'mlp.gate_proj.weight'], W[p + 'mlp.up_proj.weight'],
                       W[p + 'mlp.down_proj.weight'])
        else:
            y = moe(li, h)
        res = st.copy()
        for s in range(T):
            st[s] = mhc_post(y[s], res[s], post[s], comb[s])
    x = st.mean(axis=1)                                          # mean-collapse
    x = rmsnorm(x, W[LM + 'norm.weight'])
    return x @ W['lm_head.weight'].T


rng = np.random.default_rng(7)
NT = int(os.environ.get('NT','6'))
prompt = rng.integers(0, VOCAB, size=NT).tolist()
full = prompt + rng.integers(0, VOCAB, size=NT).tolist()
logits = forward(full)
tf_pred = logits.argmax(-1).tolist()

json.dump({'prompt_ids': prompt, 'full_ids': full, 'tf_pred': tf_pred},
          open(OUT, 'w'))
np.save(OUT.replace('.json', '_logits.npy'), logits)
print(f'wrote {OUT}: {len(full)} positions, vocab={VOCAB}')
print(f'  full_ids  {full}')
print(f'  tf_pred   {tf_pred}')
print(f'  logit range [{logits.min():.4f}, {logits.max():.4f}]')
