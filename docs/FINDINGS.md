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

### 9. Serve mode: ragged batches, and a state that is per-model

`run_serve_mux` routes **every** decode through `step_decode_batch`, which passes
per-row `kvs[]`/`positions[]` — even when only one slot is active. The KDA
dispatch refused any such call, so `coli serve` / `coli chat` died on the first
request with `engine_error`, while the direct path worked fine.

The refusal was too broad. A one-row batch is not ragged: one sequence, one
position, advancing the recurrent state by one token, which is exactly a
contiguous decode. Now gated on `S != 1`.

Two related hazards surfaced with it, both silent rather than loud:

* KDA state is per-**model** (`m->kstate[layer]`), not per-slot, so two
  concurrent sequences would advance the same state and corrupt each other with
  no error. `KV_SLOTS > 1` is now refused at startup.
* Nothing reset the state between requests, so each conversation inherited the
  previous one's. Reset now happens on a forward starting at position 0.

KV prefix reuse is disabled for KDA models: a recurrent state cannot be
reconstructed from a cached prefix, only replayed by feeding the tokens in
order. Resuming onto a state that never saw those tokens would be wrong.

### 10. Prefix reuse fed the model nothing, and the safeguard was fiction

Despite the note above, prefix reuse was not in fact disabled for KDA models.
The guard was `putenv("KV_PREFIX=0")`, and nothing in the engine reads a
`KV_PREFIX` environment variable -- a safeguard in appearance only, and a useful
thing to check for in any port: grep for the consumer, not the setter. The real
reuse happens in the serve submit path:

```c
int prefix=0;
if(!echo) while(prefix<sc->len && prefix<nt && sc->hist[prefix]==tmp[prefix]) prefix++;
```

A slot keeps the previous turn's token history. A repeated prompt matches it in
full, so `prefix == nt`, prefill starts at `pos_base = nt`, and **zero new tokens
are fed** -- which also means `kda_state_reset` never fires, because it is gated
on `pos_base == 0`. The model then answers from the previous request's recurrent
state, having never been shown the current prompt.

Sound for MLA, whose KV rows are position-addressed and self-contained. Unsound
for KDA, whose state is defined only by having been fed its tokens in order.

Observed on real weights, all from this one cause:

| Symptom | Mechanism |
|---|---|
| "I notice you sent an empty message" | literally true: zero new tokens fed |
| "my previous responses were cut off" | the state still held them |
| identical prompt, different answers at temperature 0 | output depends on the prior request |
| degenerate `" 3.2.1.1.1.1"` | polluted state from an earlier request |

Fixed by computing `has_kda` and skipping the prefix match entirely, so every
request re-prefills from position 0. Cross-slot KV adoption (`COLI_KV_SHARE=1`)
is refused for the same reason: it copies KV rows, and the recurrent state has
no rows to copy.

Cost: a full re-prefill per turn. At 0.48 tok/s that is expensive, and it is
what the README always claimed was happening.

**Why component tests cannot see it.** Every component test runs a single forward
pass, and this fault exists only on the *second* request. No single-request check
can detect it, however thorough -- which is why serve mode is now verified with a
request sequence whose answers must be reproducible.

### 11. `ARCH == "glm"` misses `glm53` -- both stop layers disarmed at once

`stop_policy()` in `openai_server.py` installs GLM's role-boundary stops only
when `ARCH == "glm"`. The new family registers as `id="glm53"`, so `StopFilter`
was built with an **empty** sequence tuple. Meanwhile the C engine deliberately
drops every stop but EOS in batched serve mode (#401) *on the assumption that
Python owns the role markers*. Neither side was watching, and `<|user|>` reached
the client as ordinary text, after which the model wrote both sides of the
conversation.

Fixed with `ARCH in ("glm", "glm53")`. This is a fifth instance of the prefix
table below, on an arch id rather than a tensor name.

Nine further sites gated the same way and silently skipped `glm53`: six in
`autotune.py`, two in `resource_plan.py`, one banner in `coli`. All are now fixed,
but not by aliasing `glm53` onto the GLM-5.2 path -- one of them needed different
logic entirely.

**The planner under-counted context state by 50%.** `_glm53_geometry()` sizes the
DSA indexer key cache *and* the k-pool gate logits, but gates both on
`_colibri_indexer_present`, which `resource_plan.py` only ever set for `glm`. The
flag was therefore never set for GLM-5.3 and that state was never planned:

| context | planned before | actual | unaccounted |
|---:|---:|---:|---:|
| 2,048 | 50.3 MB | 75.5 MB | +25 MB |
| 32,768 | 805 MB | 1,208 MB | +403 MB |
| 131,072 | 3.22 GB | 4.83 GB | **+1.61 GB** |

Simply enabling the existing probe for `glm53` would have produced `False`, for
two reasons that are both this port's recurring themes: it looks for
`model.layers.N...` (GLM-5.3 nests under `language_model.`), and it derives the
required layers from `indexer_types`, which is `"full"` on all 45 layers though
only 11 carry indexer weights -- the same misreading as finding #7. The `glm53`
branch uses `layer_types != "linear_attention"` and the nested prefix. Verified
against the real checkpoint index: the rule yields exactly `[3,7,...,43]`, the
11 main-stack indexers, with layer 45 (MTP) accounted separately.

`_ANALYSIS_CACHE_VERSION` was bumped 1 -> 2 with it. The cache signature is built
from shard and config `stat()` only, so without the bump every model analysed
before this fix would have kept the wrong flag indefinitely.

Registering a family also carries build obligations that colibri's registry test
enforces: an engine must appear in the `install` rule, `$(LIBEXECDIR)`, CI's
`ENGINES` list and the release artifact copy. `glm53` is now wired into all four,
and colibri's full Python suite runs green with these patches applied:
**699 passed, 0 failed**.


### 12. The GPU expert tier sizes its candidate list with a 2x over-estimate

`glm53.c` builds the VRAM candidate list as `budget / per-expert-estimate`, and the estimate
is roughly double the truth. On a 12 GB card:

```
[CUDA] tier staging: ... instead of 420 at once (10.6 GB)   <- 420 candidates = 10.6 GB budget
[CUDA] hot expert tier: 420/856 experts, VRAM 5.62 GB        <- those 420 occupied 5.62 GB
```

420 experts nominated for a 10.6 GB budget, consuming 5.62 GB: **25.2 MB assumed per expert,
13.4 MB actual**. VRAM stops half full regardless of what budget it is given, and no amount of
raising `CUDA_EXPERT_GB` fixes it, because that value *is* the numerator.

Proven by forcing the pin set far larger (`COLI_RAM_OVERCOMMIT=1 PIN=auto PIN_GB=11`, 856
experts pinned, 16 GB warm): VRAM still took exactly 420 and pushed the remaining 436 to RAM.
Host memory was not the constraint; the arithmetic was.

Worked around by declaring a budget larger than the card (`CUDA_EXPERT_GB=20` on a 12 GB GPU),
which yields ~790 candidates. The per-device capacity check still stops placement safely, at
**791 experts / 10.58 GB of 12.3 GB**. Worth +14% throughput.

A real fix belongs in the estimate rather than the env var; the figure appears to count host
and device copies of the same expert, which `CUDA_RELEASE_HOST` then frees.

### 13. Three configuration levers were worth 2x, and the docs said RAM was the only one

Not a defect, but the most useful operational finding here. Same hardware, same weights:

| step | tok/s |
|---|---:|
| defaults | 0.384 |
| `DIRECT=1 PIPE=1` | 0.487 |
| VRAM filled (#12) | 0.555 |
| dual-NVMe striping | **0.755** |

The diagnosis that made it possible: the engine reported **49% "I/O wait"** while the drive
was only **15% busy** at queue depth **0.9**. I/O wait means the compute thread is blocked; it
does not mean the device is saturated. The workload was latency-bound, not bandwidth-bound --
so `O_DIRECT` (skip a page cache that was copying 4 GB/token through RAM the machine lacks)
and read striping across two NVMe drives both paid, while adding GPU capacity barely did.

Caveat recorded honestly: the CUDA int4 kernels are not bit-identical to the CPU ones. With
791 experts resident on the GPU, "Hello" answers `'Hi! ...'` instead of `'Hi there! ...'`.
Deterministic across runs, still correct, but no longer the same tokens as the CPU path -- so
the component oracles, which are CPU-only, do not cover the GPU configuration.

### 14. Persisted KV resumed a stale conversation across restarts

The third door into the same fault as #10. `KVSAVE` defaults to 1, so the engine
writes `.coli_kv` and on the next start reports:

```
[KV] resumed conversation from disk: 1219 tokens in 0.2s (no re-prefill)
```

"no re-prefill" is sound for MLA -- the rows are position-addressed -- and
unsound for KDA, whose recurrent state never saw those tokens. Finding #10 closed
prefix reuse *within* a session and cross-slot adoption; this path restores state
*before* any request, so the fix did not cover it.

Symptom on real weights: a long code generation, ~2000 tokens in, spliced the
user's own earlier prompt text into the middle of a CSS declaration --

```css
    justify-content: center;
    align beautiful, good looking , royal, premium , glassmorphic design
```

-- and the model then noticed its own corruption and stopped. Short prompts never
reached the phantom region, which is why the earlier "Hello" tests came back clean
and this only appeared on a long generation.

Fixed operationally with `KVSAVE=0`. The principled fix is to refuse persistence
whenever a KDA layer is present, as prefix reuse now does.

### 15. Vision: one missing GELU made the model blind

GLM-5.3-Flash is natively multimodal. A sidecar (`vision/`) plus an engine
injection path now carry image embeddings end to end, and the model reads them.

**The bug.** The patch merger applies GELU between its LayerNorm and its SwiGLU:

```python
hidden = self.act1(self.post_projection_norm(self.proj(hidden)))   # act1 = GELU
return self.down_proj(self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden))
```

Nothing in the checkpoint hints at that activation -- tensor names give you `proj`,
`post_projection_norm`, `gate/up/down_proj` and no more. Omitting it did not merely
shift the numbers, it destroyed the tower's spatial resolution:

| | without GELU | with GELU |
|---|---|---|
| red-patch vs blue-patch, same image | 0.9836 | **0.6859** |
| mean row L2 vs `embed_tokens` (0.604) | 25.2 (42x) | **0.94** |

The 42x magnitude anomaly was a *symptom* of the missing activation, not a separate
problem -- which is why rescaling the embeddings to match the text distribution
achieved nothing, and why normalising each image to a fixed L2 made things worse by
erasing the magnitude channel the tower uses for brightness.

**How it was found.** By reading `Glm4vVisionPatchMerger` in transformers. GLM-5.3
has no reference implementation, and that was treated as final for far too long --
but the closely related **GLM-4V does**, and its merger transferred directly. When a
model has no reference, look for its nearest published relative before reasoning
from tensor names alone.

**Verified behaviour** (was wrong on all three before the fix):

```
black frame  -> "The image is completely black with no visible content."
white frame  -> "The image is completely blank and white."
red circle top-left      -> "top-left"
red circle bottom-right  -> "bottom-right"      <- answers DIFFER, and both correct
```

**Method note, kept because it nearly cost the whole result.** The first spatial test
passed and meant nothing: "which corner is the red circle?" answered "top left",
correctly -- then answered "top left" again with the image flipped. A four-way
question passed on a coin flip, and without that control this would have shipped as
working vision while being entirely blind. The cheapest possible check (black vs
white) should also have come before any tower archaeology.

**Confirmed correct along the way**, each by measurement rather than argument: pixel
preprocessing and the conv3d patch embed (patch-level cosine 0.023), pre-norm block
order (post-norm is degenerate -- cosine 1.0000, zero spread), the 2D RoPE convention
(rows-first beat no-RoPE and cols-first; the rotate-half expansion is provably
identical to the reference), `post_layernorm` placement after the blocks, and the
full-grid stride-2 downsample (equivalent to the reference's per-block reshape).

**Verified on real inputs.** A 3072x4080 phone photo of a Sudoku book: the model
read the printed caption off it -- *"page 7 from a book titled 1,000++ All HARD
Sudoku Puzzles by amazon.com/djape"* -- and counted six puzzles in a 2x3 layout,
correctly. Two images in one turn inject as 512 placeholders (prompt 539 tokens)
and both are described. Three consecutive turns alternating art / sudoku / art
each described their own image, with the two art turns byte-identical: no stale
image leaks between requests, which given findings #10 and #14 was the likeliest
place for a third stale-state bug.

**A 4 MB request cap blocked photographs entirely.** `MAX_BODY = 4 << 20` predates
vision; a 3072x4080 JPEG is ~8.4 MB base64 and the connection was dropped before
any handler ran. Now env-configurable (`COLI_MAX_BODY`) and raised to 32 MB
automatically when the tower is available, leaving the text-serving default alone.

**Video works too.** 8 frames sampled evenly from a 6 s clip, fed to the Conv3d as
real consecutive pairs rather than one frame replicated, giving 4 temporal steps x
256 = 1024 placeholders bound to `video_token_id` 154855:

```
prompt 1048 tokens (1024 video + 24 text), 287 s
-> "Mario ... red cap with the M emblem, red shirt, blue overalls, white gloves,
    brown shoes, and his signature mustache ... cartoonish video game-style landscape"
```

Every detail matches the clip. Position encoding stays 2D -- the reference repeats
the same (h, w) indices per temporal step rather than rotating a time axis -- and
attention runs across space *and* time, so frames can be compared.

Two bugs surfaced getting there. `cv2.VideoCapture` takes a file path only, so a
`data:` URI decoded zero frames while the image path (PIL, reads bytes) worked
fine; video sources are now spilled to a temp file. And the running server had
`vision_sidecar` already imported, so the fix appeared to do nothing until restart
-- a failure mode easily mistaken for a wrong fix.

**Motion IS understood, but only if the question is decomposed.** An earlier
version of this finding said motion was not perceived and blamed the architecture.
That explanation was wrong, and the correction is the more useful result.

The control: two clips from identical frames, one the exact reverse -- a black ball
travelling between a fixed red square (left) and a fixed blue square (right). Only
frame order differs.

| prompt | result |
|---|---|
| "which direction does it move?" | same answer to both clips -- **fails** |
| "where is the ball in Frame 1? in Frame 4?" | correct, and **inverted** for the reversed clip |
| staged: frame 1, then frame 4, then conclude | correct, including the direction |

```
LR (truth red -> blue):  Frame 1 near RED,  Frame 4 near BLUE
RL (truth blue -> red):  Frame 1 near BLUE, Frame 4 near RED
                         "the black ball moves FROM the BLUE square"
```

A coin flip gives the same answer twice; these invert. The temporal channel exists
and works -- the model just will not perform the two-frame comparison spontaneously.

**The tower was verified correct independently**, which is what made the diagnosis
possible. A reversed clip's last frame IS the original's first frame, and the
embeddings say exactly that:

```
LR step 0 vs RL step 0 (same index, different content)  cos 0.8315
LR step 0 vs RL step 3 (time-mirrored, same content)    cos 0.9297   <- higher
```

Two enabling changes. The engine now maps placeholder position -> row with an
explicit table instead of `pos0 + offset`, so the run need not be contiguous; the
server interleaves readable `Frame k:` labels between the per-frame blocks
(`1024 rows bound to 1024 placeholders spanning 10..1051 (18 gap tokens)`). The
labels alone did not fix the direct question -- decomposing it did -- but they are
what makes frames individually addressable.

Practical recipe: ask per frame, then conclude. `vision/probe_motion.py` reproduces
the failure, `vision/probe_motion_staged.py` the success.

**Also not covered:** accuracy degrades with two images (a layout the single-image
run counted correctly came back as "four puzzles in a 2x2 grid"), and there is still
no GLM-5.3 reference to compare numerics against.

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
