# Findings

Bugs and architectural facts found while porting GLM-5.3-Flash to colibri. The
first section is the part worth reading if you maintain colibri: three of those
bugs affect GLM-5.2 users too.

---

## Upstream bugs (affect models other than GLM-5.3)

### 1. Resume overwrites already-converted output — silent data loss

`c/tools/convert_fp8_to_int4.py`. Resuming a pass that has newly-selected
shards restarts the output counter at 0, so the new shard writes over an
existing output file. The resume manifest records both source shards mapping to
the same output:

```
'model-00005-of-00062.safetensors': 'out-idx-00000.safetensors',
'model-00002-of-00062.safetensors': 'out-idx-00000.safetensors',
```

Layer 11's entire DSA indexer vanished from a container that had converted it
successfully minutes earlier. Nothing errored, and the container still loaded.

Not GLM-5.3-specific — any interrupted-then-resumed conversion can hit it.
Worked around here by clearing the pass's outputs and re-running from scratch;
the numbering bug itself is untouched.

### 2. `--mtp` shard selection is prefix-literal

Same file. The MTP pass selects shards with

```python
want = {v for k, v in wmap.items() if k.startswith(f"model.layers.{a.n_layers}.")}
```

The `--indexer` branch three lines below already goes through `layer_idx()`.
Any checkpoint that nests its layers (GLM-5.3 uses `model.language_model.`)
selects **zero** shards, and the whole draft head is silently absent from the
container. Fixed by routing both branches through `layer_idx()`.

### 3. `--indexer` excludes the MTP block

Same file. Selection used `0 <= layer_idx(k) < n_layers`, but a checkpoint with
`index_share_for_mtp_iteration` gives the MTP block its own indexer at layer
`n_layers`. Those tensors were claimed by no pass at all. Fixed to `<=`.

---

## GLM-5.3-specific bugs found in this port

### 4. Clamped SwiGLU was missing in three places

GLM-5.3's `text_config` sets `swiglu_limit: 10.0`; GLM-5.2 has no such key, so
the inherited code never clamped. The reference (shared with DeepSeek-V4) clamps
**asymmetrically**: the gate above only, the up projection on both sides.

Missing in `dense_mlp()`, `expert_ffn()`, and — found separately, and the worst
of the three — the shared expert's own inline `silu*up` inside `moe()`. The
shared expert fires on every token of every sparse layer.

### 5. Speculative decoding is unsound with KDA

Speculation feeds draft tokens through the model and keeps the verified prefix.
That works when history lives in a KV cache: rejected positions are overwritten.
GLM-5.3's 34 KDA layers carry a **recurrent** state with no positional index to
rewind, so a rejected draft permanently corrupts it.

Measured: with MTP enabled, greedy output changed for **every** `DRAFT` value;
with MTP off it was byte-identical across `DRAFT=0/1/2`. Reproduces with
`hc_mult=1`, so it is not the mHC path.

colibri's GLM-5.2 engine never hits this — every layer there is MLA. The engine
now refuses speculation when any KDA layer is present
(`COLI_GLM53_KDA_SPEC=1` to measure the drift anyway).

A real fix needs either snapshot/restore of `kstate` + conv windows per
speculative batch (~142 MB of state per step at full scale) or advancing KDA
only over verified tokens. Neither is cheap; GLM-5.3 forfeits MTP's speedup
until one lands.

### 6. MTP block has no hyper-connections — NULL deref and 4x heap overflow

The checkpoint gives layer 45 norms, MLA attention and its own 288-expert MoE
but **no `hc_*` tensors** (`hc_attn_base` occurs 45x, `input_layernorm` 46x).
Branching the mHC path on `c->hc_mult` alone sent the MTP head down it with
`l->hc_attn_fn == NULL`, and indexed its `[S,D]` buffer as `[S,M,D]` — a 4x
out-of-bounds read *and write* at `hc_mult=4`. Confirmed: **SIGSEGV**.

Fixed by gating on `l->hc_attn_fn` rather than the global.

### 7. DSA silently disabled for the entire model

`indexer_types` is `"full"` for all 45 layers, but only the 12 non-KDA blocks
carry indexer weights. The auto-probe requires weights on every layer whose
`idx_type` is set, so it failed on layer 0 (a KDA layer) and switched DSA off
model-wide — silently, because the fallback is dense attention, which is
*correct* below `index_topk` and merely wrong above it.

Caught by an A/B: the engine matched a dense oracle 31/32 while matching the
k-pool oracle only 24/32, with the misses landing exactly on the positions where
the two oracles disagree.

Fixed by clearing `idx_type` on linear-attention layers.

### 8. Converter and engine disagreed on the KDA side projections

`classify()` put every KDA projection in one quantized class, but
`kda_forward()` reads the small ones as f32 — the same split `kimi_k3.c` uses.
The engine refused the container outright:

```
f_a_proj.weight: tensor is U8/I8 -- not a float tensor
```

Fixed by splitting the class: `q/k/v` stay quantized (3.4B params), while
`f_a`/`f_b`/`g_a`/`g_b`/`b_proj` stay f32. Costs ~345 MB of resident RAM.

---

## The prefix, four times

`model.language_model.` bit four independent code paths, each written against
GLM-5.2 naming:

| # | Where | Symptom |
|---|---|---|
| 1 | `layer_idx()` in the converter | every layer tensor scored -1; MTP/indexer split collapsed |
| 2 | engine tensor-name macros | every tensor lookup missed |
| 3 | `_GLM_EXPERT` regex in `family_registry.py` | zero experts found; planner sized 188 GB as resident |
| 4 | `--mtp` shard selection | zero shards selected; draft head absent |

If you port another nested checkpoint, grep for `model.layers.` first.

---

## Architectural notes

* **k-pool DSA.** GLM-5.3's indexer scores *pools* of `index_kpool` (4) tokens,
  not single tokens. A pool key is a per-channel convex mix
  `softmax(gate + ape) · member keys`, so the cache must hold the gate logits of
  every token alongside its indexer key. transformers' `glm_moe_dsa` has zero
  k-pool support; this is a genuinely different indexer, not a tweak.
* **No RoPE anywhere.** `mla_use_nope: true`, `qk_rope_head_dim: 0`. The
  indexer has none either — the intra-pool `ape` is the ordering signal. The
  config still carries `indexer_rope_interleave: true`; the running reference
  wins.
* **KV geometry.** Only 11 of 45 layers hold a KV cache. KDA state is ~0.15 GB
  and **constant in context**. 128K context costs 4.5 GB of KV, not the ~18 GB
  a GLM-5.2-shaped geometry would predict.
* **Expert path is W4A8.** `qrow_i8` quantizes activations to int8;
  `dot_i4i8` does int4 x int8. `IDOT=0` selects exact f32 kernels. Measured on
  one synthetic layer, activation quantization adds ~1.6% mean relative error on
  top of weight quantization — which weight-only ablations do not capture.
