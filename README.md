# glm53-colibri

Run **GLM-5.3-Flash** (320B total / 18B active, MoE) on a consumer machine, on
CPU, by streaming experts from NVMe.

This adds a GLM-5.3 engine to [colibri](https://github.com/JustVugg/colibri),
which supports GLM-5.2 but not GLM-5.3. It is a **fork with additions**, not a
new engine — see [NOTICE](NOTICE) for exactly what is derived and from what.

Measured on the machine below: **0.48 tok/s**, 13.3 GB RSS, 168 GiB on disk.

> **Status: working, incompletely validated.** It generates coherent text on
> real weights and every component is checked against an oracle, but no
> full-model reference implementation exists for GLM-5.3 anywhere, so the
> assembled engine has never been compared end to end against one. Read
> [docs/VALIDATION.md](docs/VALIDATION.md) before trusting output.

---

## Why this needed writing

GLM-5.3 is not GLM-5.2 with different weights. Four things differ enough to
break the existing engine:

| | GLM-5.2 | GLM-5.3-Flash |
|---|---|---|
| Attention | 100% MLA | **34 KDA (linear) + 11 MLA/DSA** of 45 |
| Residual | single stream | **mHC**, 4 parallel streams with Sinkhorn mixing |
| DSA indexer | per-token scoring | **k-pool**: scores pools of 4 tokens |
| Activation | plain SwiGLU | **clamped** SwiGLU (`swiglu_limit: 10.0`) |
| Tensor names | `model.layers.N` | `model.language_model.layers.N` |

Two of those pieces already existed inside colibri and were reused: KDA from
its Kimi K3 engine, mHC verbatim from its DeepSeek-V4 engine. (llama.cpp's
glm5next PR independently reached the same two donors.) The k-pool indexer
existed nowhere runnable.

## The k-pool indexer

The main original contribution. GLM-5.3's lightning indexer scores **pools** of
`index_kpool` (4) consecutive tokens rather than individual tokens; a pool key
is a learned per-channel convex mix `softmax(gate + ape) · member keys`, so the
cache must hold the gate logits of every token alongside its indexer key.

`transformers`' `glm_moe_dsa` has **zero** k-pool support. llama.cpp
[PR #27752](https://github.com/ggml-org/llama.cpp/pull/27752) describes the
algorithm but is unmerged, text-only, and its author states it is *"not
numerically validated"* and *"never tested on real weights"* (they had 128 GB;
the checkpoint is 328 GB).

This implementation is validated **cell-set-exact** (48/48 rows) against a
reference on an all-f32 container, and runs on the real 320B weights.

## Bugs found

Eight, detailed in [docs/FINDINGS.md](docs/FINDINGS.md). **Three affect
colibri generally, not just GLM-5.3** — most seriously a resume path that
silently overwrites already-converted output, which cost an entire layer's DSA
indexer without erroring.

Also: unclamped SwiGLU in three places, a NULL-deref plus 4x heap overflow on
the MTP block (confirmed SIGSEGV), DSA silently disabled model-wide, and
speculative decoding being **unsound** with KDA's non-rewindable recurrent
state.

---

## Quick start

Needs ~500 GB free disk, a C toolchain (gcc/clang; MSVC will not build
colibri), and Python with `torch`, `transformers`, `safetensors`,
`huggingface_hub`.

```bash
# 1. colibri, at the commit this was forked from
git clone https://github.com/JustVugg/colibri && cd colibri
git checkout dd7df2c907494f0b9812a8bf972d56079f11791e

# 2. apply this work
cp /path/to/glm53-colibri/engine/glm53.c        c/
cp /path/to/glm53-colibri/engine/glm53_mhc.h    c/
cp /path/to/glm53-colibri/engine/tests/*.c      c/tests/
git apply /path/to/glm53-colibri/converter/*.patch /path/to/glm53-colibri/engine/*.patch

# 3. build
cd c && make glm53

# 4. component tests (no weights needed)
make test-glm53
```

Then the weights (~328 GB download, ~90 min conversion):

```bash
python scripts/download_glm53.py            # -> GLM-5.3-Flash/
bash   scripts/convert_glm53.sh             # -> glm53-int4/, ~168 GiB
python validation/verify_container.py glm53-int4    # pre-flight; run this
```

Chat:

```bash
python coli chat --model /path/to/glm53-int4 --ram 17 --ctx 2048 --cap 6
```

`--ram` matters: auto-detect is conservative and drops the expert cache hard.

---

## Quantization

Experts **int4-gs128**, everything resident int8, KDA side projections f32.

Group size was chosen by measurement, not intuition. Ablating on OLMoE-1B-7B
(n=200/task, acc_norm, harness reproducing colibri's published fp16 baseline of
57.0% exactly):

| scheme | bits/wt | mean | delta |
|---|---:|---:|---:|
| fp16 | 16.00 | 57.0% | — |
| **int4-g128** | 4.25 | 55.8% | **−1.2pp** |
| int4-g64 | 4.50 | 55.5% | −1.5pp |
| int4 per-row | 4.02 | 52.3% | −4.7pp |
| int3-g64 | 3.50 | 51.2% | −5.8pp |

**Grouping is what matters; group size is not.** g64 and g128 are inside noise
(~2pp standard error), while per-row costs 3.5pp. g128 is smaller and reads
4.49 GB/token instead of 4.76. int3-g64 is *worse than per-row int4* while also
being smaller — strictly dominated.

Caveat: this is a weight-only round trip. colibri's expert path is **W4A8** —
activations are quantized to int8 too, adding error these numbers do not
capture. Treat −1.2pp as a floor.

---

## Performance, honestly

Measured on: Intel 24-core, 31.6 GB RAM (~17.5 GB free), RTX 4080 Laptop 12 GB,
Crucial P3 Plus NVMe.

```
prefill 15 tokens in 26.87s | decode 24 tokens in 49.65s (0.48 tok/s)
expert hit rate 21.5% | RSS 13.33 GB
PROFILE: expert-disk 195.7s | expert-matmul 5.2s
```

**Disk service exceeds expert matmul 37:1.** The bottleneck is RAM, not the
GPU and not the CPU. At 32 GB the cache holds 6 of 288 experts per layer against
top-8 routing, so consecutive tokens route to largely different experts and
almost nothing is reused (322 experts loaded per token against a 336 baseline).

Raising `RAM_GB` from auto (14.8) to 17 and halving context took the hit rate
from 8.5% to 21.5% but throughput only from 0.43 to 0.48 tok/s — cache doubling
improved reuse *within* a token, not the number of distinct experts each token
needs.

**More RAM is the only real lever.** colibri's own figures put 64 GB+ with
~40 GB pinned at 2–4 tok/s. Untried here: dual-NVMe striping, which colibri
supports and which should help while disk-bound.

---

## Known limitations

* **Speculative decoding is disabled** and refuses to run. KDA's recurrent state
  cannot be rewound after a rejected draft, so speculation silently corrupts
  every later token. GLM-5.3 forfeits MTP's speedup until snapshot/restore or
  verified-only advance is implemented. (`COLI_GLM53_KDA_SPEC=1` to measure the
  drift; output is wrong.)
* **Vision is not implemented.** GLM-5.3 is natively multimodal; the 563.6M ViT
  converts into the container and is unused. Groundwork is in
  [docs/FINDINGS.md](docs/FINDINGS.md) — notably that its position encoding is
  parameter-free 2D RoPE (GLM-5.3 dropped GLM-4V's learned interpolated
  embedding, which is why the checkpoint has no positional tensor). Because a
  ViT is dense, cheap, runs once per image and fits in 12 GB VRAM, the cheapest
  path is a PyTorch sidecar producing 256 embeddings spliced at `<|image|>`.
* **Long context past `index_topk` (2048) is the least-tested path** on real
  weights.
* **MTP numerics unvalidated** — the head loads and runs, its output was never
  checked. Moot while speculation is off.

---

## Layout

```
engine/      glm53.c (forked from colibri.c), glm53_mhc.h, test, Makefile patch
converter/   patches: classify() routing, k-pool config, --all-f32, family registry
validation/  oracles, fixtures, container/plan checkers
scripts/     download and convert
docs/        FINDINGS.md, VALIDATION.md
```

## License

Apache 2.0, inherited from colibri. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
GLM-5.3-Flash weights are MIT, © Z.AI — not redistributed here.
