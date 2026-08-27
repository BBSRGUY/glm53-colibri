"""Generate a tiny GLM-5.3-Flash checkpoint in real HF layout.

Deliberately the smallest model that still crosses every distinct path in glm53.c:

  layer 0: KDA  + DENSE mlp        (first_k_dense_replace = 1)
  layer 1: DSA  + sparse MoE
  layer 2: KDA  + sparse MoE
  layer 3: DSA  + sparse MoE

Tensor names and config nesting are the real ones -- model.language_model.* with a
text_config section -- so this exercises the prefix handling, the converter's classify()
routing, and the engine's loader together. Weights are BF16 (not FP8): the FP8 dequant
path was already covered in Phase 1 by a fixture cut from the real checkpoint.

No MTP block and no DSA indexer tensors: both are optional and the engine probes for
them, so leaving them out selects has_mtp=0 / has_dsa=0 and dense causal attention,
which is the regime the per-layer tests validated.
"""
import json
import os
import sys

import numpy as np
from safetensors.numpy import save_file

OUT = sys.argv[1] if len(sys.argv) > 1 else 'glm53_tiny'

D, NL, VOCAB = 64, None, 32
NH, QL, KL, NOPE, VH = 2, 16, 8, 16, 16
KDA_H, KDA_HD, CONV_K = 2, 32, 4
E, TOPK, MI, NSH, DENSE_I = 4, 2, 32, 1, 48
DSA = int(os.environ.get('DSA', '0'))
IDX_NH, IDX_HD, IDX_TOPK, KPOOL = 2, 16, int(os.environ.get('IDX_TOPK', '4')), 4
HC, HC_ITERS, HC_EPS = int(os.environ.get("HC","4")), 20, 1e-6
EPS, SLIM, RSCALE = 1e-5, 10.0, 2.5
LAYER_TYPES = os.environ.get('LT','linear_attention,deepseek_sparse_attention,linear_attention,deepseek_sparse_attention').split(',')
FIRST_DENSE = int(os.environ.get('FD','1'))
NL = len(LAYER_TYPES)
P, R, N = KDA_H * KDA_HD, KDA_HD, 2 * HC + HC * HC

rng = np.random.default_rng(20260827)


def rnd(*shape, scale=0.05):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def bf16(a):
    """Round f32 -> bf16 and keep it in a bf16-exact f32 array.

    The checkpoint stores bf16; the oracle must see the SAME values the engine loads,
    so round here and hand both sides the rounded weights.
    """
    u = np.ascontiguousarray(a, '<f4').view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32).reshape(a.shape)


T = {}          # bf16-rounded f32, for the oracle
LM = 'model.language_model.'


def put(name, a):
    T[name] = bf16(a)


put(LM + 'embed_tokens.weight', rnd(VOCAB, D, scale=0.3))
put(LM + 'norm.weight', 1.0 + rnd(D, scale=0.1))
put('lm_head.weight', rnd(VOCAB, D, scale=0.3))

for i in range(NL):
    p = f'{LM}layers.{i}.'
    put(p + 'input_layernorm.weight', 1.0 + rnd(D, scale=0.1))
    put(p + 'post_attention_layernorm.weight', 1.0 + rnd(D, scale=0.1))
    for br in (('attn','ffn') if HC>1 else ()):
        put(p + f'hc_{br}_fn', rnd(N, HC * D))
        put(p + f'hc_{br}_base', rnd(N, scale=0.1))
        put(p + f'hc_{br}_scale', rnd(3, scale=0.5))
    if LAYER_TYPES[i] == 'linear_attention':
        put(p + 'self_attn.q_proj.weight', rnd(P, D, scale=0.2))
        put(p + 'self_attn.k_proj.weight', rnd(P, D, scale=0.2))
        put(p + 'self_attn.v_proj.weight', rnd(P, D, scale=0.2))
        put(p + 'self_attn.o_proj.weight', rnd(D, P, scale=0.2))
        for t in ('q', 'k', 'v'):
            put(p + f'self_attn.{t}_conv1d.weight', rnd(P, 1, CONV_K))
        put(p + 'self_attn.f_a_proj.weight', rnd(R, D))
        put(p + 'self_attn.f_b_proj.weight', rnd(P, R))
        put(p + 'self_attn.g_a_proj.weight', rnd(R, D))
        put(p + 'self_attn.g_b_proj.weight', rnd(P, R))
        put(p + 'self_attn.b_proj.weight', rnd(KDA_H, D))
        put(p + 'self_attn.dt_bias', rnd(P, scale=0.2))
        put(p + 'self_attn.A_log', rnd(KDA_H, scale=0.3))
        put(p + 'self_attn.o_norm.weight', 1.0 + rnd(KDA_HD, scale=0.1))
    else:
        put(p + 'self_attn.q_a_proj.weight', rnd(QL, D, scale=0.2))
        put(p + 'self_attn.q_a_layernorm.weight', 1.0 + rnd(QL, scale=0.1))
        put(p + 'self_attn.q_b_proj.weight', rnd(NH * NOPE, QL, scale=0.2))
        put(p + 'self_attn.kv_a_proj_with_mqa.weight', rnd(KL, D, scale=0.2))
        put(p + 'self_attn.kv_a_layernorm.weight', 1.0 + rnd(KL, scale=0.1))
        put(p + 'self_attn.kv_b_proj.weight', rnd(NH * (NOPE + VH), KL, scale=0.2))
        put(p + 'self_attn.o_proj.weight', rnd(D, NH * VH, scale=0.2))
        if DSA:
            q = p + 'self_attn.indexer.'
            put(q + 'wq_b.weight', rnd(IDX_NH * IDX_HD, QL, scale=0.2))
            put(q + 'wk.weight', rnd(IDX_HD, D, scale=0.2))
            put(q + 'weights_proj.weight', rnd(IDX_NH, D, scale=0.3))
            put(q + 'k_norm.weight', 1.0 + rnd(IDX_HD, scale=0.1))
            put(q + 'k_norm.bias', rnd(IDX_HD, scale=0.05))
            put(q + 'index_kpool_compress_gate', rnd(IDX_HD, D, scale=0.3))
            put(q + 'index_kpool_compress_ape', rnd(KPOOL, IDX_HD, scale=0.5))
    if i < FIRST_DENSE:
        put(p + 'mlp.gate_proj.weight', rnd(DENSE_I, D, scale=0.2))
        put(p + 'mlp.up_proj.weight', rnd(DENSE_I, D, scale=0.2))
        put(p + 'mlp.down_proj.weight', rnd(D, DENSE_I, scale=0.2))
    else:
        put(p + 'mlp.gate.weight', rnd(E, D, scale=0.3))
        put(p + 'mlp.gate.e_score_correction_bias', rnd(E, scale=0.4))
        put(p + 'mlp.shared_experts.gate_proj.weight', rnd(MI * NSH, D, scale=0.2))
        put(p + 'mlp.shared_experts.up_proj.weight', rnd(MI * NSH, D, scale=0.2))
        put(p + 'mlp.shared_experts.down_proj.weight', rnd(D, MI * NSH, scale=0.2))
        for e in range(E):
            put(p + f'mlp.experts.{e}.gate_proj.weight', rnd(MI, D, scale=0.2))
            put(p + f'mlp.experts.{e}.up_proj.weight', rnd(MI, D, scale=0.2))
            put(p + f'mlp.experts.{e}.down_proj.weight', rnd(D, MI, scale=0.2))

# ---- optional MTP block at index NL (DeepSeek-V3 style draft head) ----
WANT_MTP = int(os.environ.get('MTP', '0'))
if WANT_MTP:
    p = f'{LM}layers.{NL}.'
    # NOTE: no hc_* here. The real checkpoint gives layer 45 norms, attention and a MoE
    # but no hyper-connections (hc_attn_base occurs 45x, input_layernorm 46x), so the
    # MTP block runs on the collapsed hidden state rather than the hc_mult streams.
    put(p + 'eh_proj.weight', rnd(D, 2 * D, scale=0.2))
    put(p + 'enorm.weight', 1.0 + rnd(D, scale=0.1))
    put(p + 'hnorm.weight', 1.0 + rnd(D, scale=0.1))
    put(p + 'shared_head.norm.weight', 1.0 + rnd(D, scale=0.1))
    put(p + 'input_layernorm.weight', 1.0 + rnd(D, scale=0.1))
    put(p + 'post_attention_layernorm.weight', 1.0 + rnd(D, scale=0.1))
    put(p + 'self_attn.q_a_proj.weight', rnd(QL, D, scale=0.2))
    put(p + 'self_attn.q_a_layernorm.weight', 1.0 + rnd(QL, scale=0.1))
    put(p + 'self_attn.q_b_proj.weight', rnd(NH * NOPE, QL, scale=0.2))
    put(p + 'self_attn.kv_a_proj_with_mqa.weight', rnd(KL, D, scale=0.2))
    put(p + 'self_attn.kv_a_layernorm.weight', 1.0 + rnd(KL, scale=0.1))
    put(p + 'self_attn.kv_b_proj.weight', rnd(NH * (NOPE + VH), KL, scale=0.2))
    put(p + 'self_attn.o_proj.weight', rnd(D, NH * VH, scale=0.2))
    put(p + 'mlp.gate.weight', rnd(E, D, scale=0.3))
    put(p + 'mlp.gate.e_score_correction_bias', rnd(E, scale=0.4))
    put(p + 'mlp.shared_experts.gate_proj.weight', rnd(MI * NSH, D, scale=0.2))
    put(p + 'mlp.shared_experts.up_proj.weight', rnd(MI * NSH, D, scale=0.2))
    put(p + 'mlp.shared_experts.down_proj.weight', rnd(D, MI * NSH, scale=0.2))
    for e in range(E):
        put(p + f'mlp.experts.{e}.gate_proj.weight', rnd(MI, D, scale=0.2))
        put(p + f'mlp.experts.{e}.up_proj.weight', rnd(MI, D, scale=0.2))
        put(p + f'mlp.experts.{e}.down_proj.weight', rnd(D, MI, scale=0.2))

os.makedirs(OUT, exist_ok=True)
# safetensors.numpy has no bf16 dtype; the values are already bf16-exact, and the
# converter reads whatever float dtype it finds, so store them as f32.
save_file({k: np.ascontiguousarray(v) for k, v in T.items()},
          os.path.join(OUT, 'model.safetensors'))

cfg = {
    'architectures': ['Glm5NextForConditionalGeneration'],
    'model_type': 'glm5_next',
    'text_config': {
        'model_type': 'glm5_next_text',
        'hidden_size': D, 'num_hidden_layers': NL, 'num_attention_heads': NH,
        'num_key_value_heads': NH, 'vocab_size': VOCAB,
        'intermediate_size': DENSE_I, 'moe_intermediate_size': MI,
        'n_routed_experts': E, 'num_experts_per_tok': TOPK, 'n_shared_experts': NSH,
        'first_k_dense_replace': FIRST_DENSE,
        'q_lora_rank': QL, 'kv_lora_rank': KL,
        'qk_nope_head_dim': NOPE, 'qk_rope_head_dim': 0, 'v_head_dim': VH,
        'mla_use_nope': True,
        'n_group': 1, 'topk_group': 1, 'norm_topk_prob': True,
        'topk_method': 'noaux_tc', 'scoring_func': 'sigmoid',
        'routed_scaling_factor': RSCALE, 'rms_norm_eps': EPS,
        'swiglu_limit': SLIM,
        'layer_types': LAYER_TYPES,
        'indexer_types': ['full'] * NL,
        'index_topk': IDX_TOPK, 'index_n_heads': IDX_NH, 'index_head_dim': IDX_HD,
        'index_kpool': KPOOL, 'index_kpool_compress': bool(DSA),
        'index_kpool_always_select_tail': True,
        'linear_attn_config': {
            'num_heads': KDA_H, 'head_dim': KDA_HD,
            'short_conv_kernel_size': CONV_K, 'gate_lower_bound': -5.0,
            'kda_layers': [i for i, t in enumerate(LAYER_TYPES) if t == 'linear_attention'],
            'full_attn_layers': [i for i, t in enumerate(LAYER_TYPES) if t != 'linear_attention'],
        },
        'mhc': HC > 1, 'hc_mult': HC, 'hc_sinkhorn_iters': HC_ITERS, 'hc_eps': HC_EPS,
        'num_nextn_predict_layers': int(os.environ.get('MTP', '0')),
        'eos_token_id': [VOCAB - 1],
        'tie_word_embeddings': False,
    },
}
json.dump(cfg, open(os.path.join(OUT, 'config.json'), 'w'), indent=1)
np.save(os.path.join(OUT, '_oracle_weights.npy'), T, allow_pickle=True)

print(f'wrote {OUT}/: {len(T)} tensors, {NL} layers '
      f'({LAYER_TYPES.count("linear_attention")} KDA / '
      f'{LAYER_TYPES.count("deepseek_sparse_attention")} DSA), '
      f'{FIRST_DENSE} dense + {NL-FIRST_DENSE} sparse MLP, hc_mult={HC}')
