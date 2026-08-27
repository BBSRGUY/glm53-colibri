#!/usr/bin/env bash
# Convert the real GLM-5.3-Flash checkpoint to a colibri container.
#
# Three passes, as the converter's design requires: the main model, the MTP draft block
# (layer 45), and the DSA indexer. They write disjoint shards into one directory.
#
# Widths follow the Phase-1 recipe as corrected by the measurements:
#   * routed experts int4 at --group-size 128. The ablation put g128 at -1.2pp and g64 at
#     -1.5pp against fp16 -- inside noise -- while g128 is 9.7 GB smaller and reads 4.49
#     GB/token instead of 4.76. Grouping is what matters (per-row int4 measured -4.7pp);
#     the group SIZE does not.
#   * everything resident int8, except the KDA side projections (f_a/f_b/g_a/g_b/b_proj),
#     which classify() keeps f32 because kda_forward reads them f32. Costs ~345 MB.
#   * MTP head int8: colibri measured int4 draft heads collapsing to 0-4% acceptance.
#
# Expected: ~176 GB total, ~10.4 GB resident.
set -euo pipefail

SRC=${SRC:-D:/GLM5.3-flash/GLM-5.3-Flash}
DST=${DST:-D:/GLM5.3-flash/glm53-int4}
CONV=D:/GLM5.3-flash/colibri/c/tools/convert_fp8_to_int4.py
GS=${GS:-128}

common=(--indir "$SRC" --outdir "$DST" --n-layers 45 --group-size "$GS")

echo "=== pass 1/3: main model (experts int4-gs$GS, resident int8) ==="
python "$CONV" "${common[@]}" \
    --ebits 8 --xbits 4 --io-bits 8 \
    --shared-bits 8 --o-bits 8 --kvb-bits 8 --attn-bits 8 \
    --dmlp-bits 8 --kda-bits 8 --vis-bits 8 \
    --min-free-gb 30

echo "=== pass 2/3: MTP draft block (int8 -- int4 heads do not draft) ==="
python "$CONV" "${common[@]}" --mtp --ebits 8

echo "=== pass 3/3: DSA k-pool indexer ==="
python "$CONV" "${common[@]}" --indexer --ebits 8

echo "=== result ==="
du -sh "$DST"
ls -la "$DST" | head -20
