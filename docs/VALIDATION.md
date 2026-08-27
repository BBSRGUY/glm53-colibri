# Validation

Every number here was measured on the machine described in README.md. Where a
third-party oracle existed it was used; where none existed that is stated
plainly, because it changes how much the result is worth.

Run the component suite with `make test-glm53` from colibri's `c/` directory.

---

## Component suite (11 tests)

| Test | Oracle | Result |
|---|---|---|
| mHC `post` / `comb` | transformers `DeepseekV4HyperConnection` | 1.2e-07 / 6.0e-08 |
| mHC `collapsed` / `updated` | same | 3.7e-03 = exactly one bf16 ulp |
| mHC stream seed / mean-collapse | same | 0.0 / 1.2e-07 |
| KDA batch | colibri `k3_ref.py`, adapted | rel 2.1e-07 |
| KDA incremental vs batch | self-consistency | **0.000e+00 bit-identical** |
| Whole layer (KDA + dense) | numpy | 0.000e+00 |
| Whole layer, clamp binding | numpy | 0.000e+00 |
| Sparse MoE layer | numpy | 0.000e+00 |
| Sparse MoE, clamp binding | numpy | 0.000e+00 |
| MLA layer, S<=4 and S>4, both absorb modes | transformers `GlmMoeDsaAttention` | 6.8e-04 |
| int4-gs64 MoE, `IDOT=0` (W4A16) | dequant reference | 2.6e-06 |
| int4-gs64 MoE, `IDOT=1` (W4A8) | same | 2.1e-02 (activation quantization) |

**Independently sourced:** mHC and MLA only. The KDA, whole-layer, MoE and
k-pool oracles are the author's own numpy, so a shared misreading of the spec
would not show up. This is the main weakness of the suite.

### bf16 is the floor

mHC rounds to bf16 at both sites, so one ulp (~2^-8 = 3.9e-3) is the finest
difference the model can itself represent. Where a test reads 3.7e-03 rather
than 0, that was confirmed to be *exactly* rounding: `|oracle - bf16(oracle)|`
matched the observed error to the last digit at both D=32 (3.722e-03) and
D=4096 (7.807e-03).

### Negative controls

Every gate was checked to actually fail. Perturbing one expected value trips
each of them (`moe_neg` 1.88e-02, `layer_neg` 9.55e-03, k-pool selection
mismatch). A tolerance that also disarms its control is worthless.

### Production dimensions

The suite was re-run at real GLM-5.3 sizes (64 heads x head_dim 128, D=4096,
q_lora 1536, kv_lora 512, moe_inter 2048, topk 8). Three tolerances had been
calibrated on tiny fixtures and failed there — all three were *test* bugs, not
engine bugs:

1. absolute tolerance on bf16 outputs (one ulp scales with magnitude);
2. gate tolerance ignoring accumulation length (the Sinkhorn dot runs over
   16384 terms at D=4096 vs 128 at D=32);
3. MoE/whole-layer demanding bit-exactness, which only holds when engine and
   oracle agree in f32 *before* rounding.

All now gate at one bf16 ulp while still printing `max-rel`, so the tiny
fixtures continue to read `0.000e` and a regression under the gate stays visible.

---

## k-pool DSA indexer

Compared **selection sets** rather than final tokens, which removes both
quantization near-ties and top-k tie-breaking from the comparison:

| Container | Rows identical |
|---|---|
| everything int8 | 40 / 48 |
| f32 indexer, rest int8 | 39 / 48 |
| **everything f32** | **48 / 48 — exact** |

The middle row is the interesting one. Carrying the *indexer* at f32 changed
nothing; the mismatches came from the indexer's **input** drifting, because
every layer before it was still quantized. Only an all-f32 container isolates
the selection. `--all-f32` exists for exactly this.

Under a normally-quantized container the selection therefore *does* drift from
an f32 reference. That is upstream quantization, not a k-pool defect — but it
means long-context behaviour past `index_topk` is the least-tested part of the
stack on real weights.

---

## Whole stack

A 4-layer GLM-5.3 (2 KDA + 2 DSA, 1 dense + 3 MoE, mHC, MTP) generated in HF
layout, converted by the patched converter, loaded by the real engine, and
teacher-forced against a numpy oracle:

| Configuration | Positions matched |
|---|---|
| all-int8, exact f32 kernels | 11/12 |
| all-int8, W8A8 | 11/12 |
| int4-gs64 experts, exact | 10/12 |
| int4-gs64 experts, W4A8 (shipping) | 9/12 |

Single-layer bisect: KDA+dense 12/12, MLA+dense 11/12, KDA+MoE 12/12,
MLA+MoE 11/12 at int8. No component is broken; degradation tracks quantization
aggressiveness.

The persistent int8 mismatch was checked rather than assumed: it sits at a
position where the **oracle's own** top-1/top-2 margin is 0.00087, with engine
logits within 0.04 on values of ~5 (<1% relative). A genuine near-tie.

---

## Real weights

Converted checkpoint verified with `validation/verify_container.py`:

```
container: 76 shards, 76431 tensors, 179.3 GB
expected 38770 tensors: 38770 present, 0 MISSING, 0 unexpected
RESULT: container is complete and correctly formatted
```

Quantization error on 400 full-size expert tensors from real shard 6:

| | Predicted from a 244 MB slice | Measured on real shard |
|---|---|---|
| int4-gs128 mean rel. err | 0.1301 | **0.1301** |

---

## Serve mode (OpenAI endpoint)

The first version of this section verified serve mode with **one** request and
called it working. That was wrong, and the way it was wrong is worth recording:
a single request cannot detect a bug whose symptom is *the second* request
answering from the first one's state. Finding #10 was live the whole time.

What is checked now is a **sequence**, at `temperature=0`, where every answer is
required to be reproducible:

```
1  "Hello"                       -> 'Hi there! How can I help you today?'   [stop]
2  "Hello"                       -> 'Hi there! How can I help you today?'   [stop]
3  "What is the capital of..."   -> (correct)
4  "Hello"                       -> 'Hi there! How can I help you today?'   [stop]
```

All three "Hello" answers are byte-identical, including #4 across an
intervening unrelated prompt. Before the fix the same sequence produced three
*different* answers:

```
1  'It looks like your message got cut off. Could you let'        [length]
2  'It looks like your message may have been cut off. Could'      [length]
4  'I notice my previous responses were cut off or incomplete.'   [length]
```

Run 4 referred to "my previous responses" in a request whose entire content was
the word "Hello".

Two independent things had to be fixed for this to hold: prefix reuse (#10) and
the stop set (#11). The `[stop]` finish reason is part of the check -- before,
generation ran to the length cap because no stop token was armed.

A two-turn conversation was checked separately, since that is the path finding
#10 broke and the one the web UI uses:

```
user       "My favourite colour is teal."
assistant  "Teal is a lovely choice!"
user       "What colour did I just say I like?"
        -> 'You said your favourite colour is teal.'   [stop]
```

The history is genuinely consumed -- the answer is not recoverable from the last
turn alone. This costs a full re-prefill of the transcript on every turn.

---

## What is NOT validated

* **No end-to-end reference exists.** `Glm5NextForConditionalGeneration` is not
  in transformers, and no `eh_proj` implementation exists anywhere in it. The
  engine has never been compared against a full-model reference — only against
  per-component oracles and its own coherent output.
* **MTP numerics.** The draft head loads and runs; its *output* was never
  validated. colibri's `MTP_PRENORM` / `MTP_SWAP` switches show the concat order
  and norm placement were uncertain even for GLM-5.2. Because drafts are
  verified against the main model, wrong numerics cost acceptance rate, not
  correctness — so the right test is acceptance rate on real weights, which
  requires KDA speculation to work first.
* **Vision tower.** Not consumed. The 563.6M-parameter ViT converts into the
  container (0.62 GB) and is simply unused. Position encoding is
  parameter-free 2D RoPE (GLM-5.3 dropped GLM-4V's learned interpolated
  embedding, which is why no positional tensor exists); see README.
* **Long context.** Nothing was run past `index_topk` on real weights.
* **Scale.** Whole-stack tests are 4 layers at D=64. Nothing exercises expert
  streaming under cache pressure except the real run.
* **Long conversations.** One two-turn exchange is checked above. Nothing tests
  a transcript that keeps growing, which is where the per-turn re-prefill cost
  compounds. Streaming responses and concurrent requests are also untested (the
  latter refused by design).
* **Component tests cannot see request-to-request state.** Every test in the
  suite is a single forward pass. A bug that only appears on the second request
  is invisible to all eleven of them, which is how #10 survived to real weights.
