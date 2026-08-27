"""Can transformers' GlmMoeDsaAttention run with GLM-5.3's NoPE settings?

GLM-5.3 sets mla_use_nope=true / qk_rope_head_dim=0, which GLM-5.2 never does. If the
module tolerates a zero-width rope split, it can serve as a third-party oracle for the
MLA layer; if not, the oracle has to be built from the projection sequence directly.
"""
import torch
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention

D, HEADS, T = 32, 2, 4

cfg = GlmMoeDsaConfig(
    hidden_size=D,
    num_attention_heads=HEADS,
    num_key_value_heads=HEADS,
    q_lora_rank=16,
    kv_lora_rank=8,
    qk_nope_head_dim=16,
    qk_rope_head_dim=0,          # <-- GLM-5.3 NoPE
    v_head_dim=16,
    num_hidden_layers=1,
    index_topk=1024,             # >= T, so the indexer selects everything
    index_n_heads=2,
    index_head_dim=8,
    indexer_types=["full"],
    attention_bias=False,
)
print('config built. qk_head_dim =', getattr(cfg, 'qk_head_dim', 'n/a'))

try:
    attn = GlmMoeDsaAttention(cfg, 0).double().eval()
    print('module constructed OK')
    print('  has indexer:', attn.indexer is not None)
    print('  scaling:', attn.scaling)
    x = torch.randn(1, T, D, dtype=torch.float64)
    pos = torch.arange(T)[None]
    rope_dim = cfg.qk_rope_head_dim
    cos = torch.ones(1, T, max(rope_dim, 1), dtype=torch.float64)
    sin = torch.zeros(1, T, max(rope_dim, 1), dtype=torch.float64)
    with torch.no_grad():
        out = attn(x, position_embeddings=(cos, sin), position_ids=pos, attention_mask=None)
    print('forward OK, out shape', out[0].shape)
except Exception as e:
    print(f'FAILED: {type(e).__name__}: {e}')
