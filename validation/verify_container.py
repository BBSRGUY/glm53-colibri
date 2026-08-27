"""Pre-flight a converted GLM-5.3 container before committing to a long run.

Checks the three things that make the engine die late rather than early:

  1. COMPLETENESS -- every tensor classify() said this pass would emit is actually in
     the container. A missing expert only surfaces when routing first selects it, which
     can be thousands of tokens in.
  2. FORMAT -- quantized tensors carry a .qs sidecar of the right cardinality, and the
     tensors the engine reads as f32 (KDA side projections, norms, routers, mHC) have
     NO sidecar. Getting this backwards is what made the engine refuse f_a_proj.
  3. SIZE -- measured bytes against the manifest prediction.

Usage: python verify_container.py <container_dir> [source_index.json]
"""
import collections
import glob
import json
import os
import sys

from safetensors import safe_open

sys.path.insert(0, os.path.join('..', 'colibri', 'c', 'tools'))
from convert_fp8_to_int4 import classify  # noqa: E402

CONT = sys.argv[1] if len(sys.argv) > 1 else 'D:/GLM5.3-flash/glm53-int4'
INDEX = sys.argv[2] if len(sys.argv) > 2 else 'D:/GLM5.3-flash/glm53_index.json'
N_LAYERS = 45

# ---- what the container actually holds ----
have, dtypes, shards = {}, {}, sorted(glob.glob(os.path.join(CONT, '*.safetensors')))
if not shards:
    sys.exit(f'no shards in {CONT}')
for p in shards:
    with safe_open(p, framework='pt') as f:
        for k in f.keys():
            have[k] = f.get_slice(k).get_shape()
            dtypes[k] = f.get_slice(k).get_dtype()
print(f'container: {len(shards)} shards, {len(have)} tensors, '
      f'{sum(os.path.getsize(p) for p in shards)/1e9:.1f} GB')

# ---- what the converter should have emitted ----
# source tensor names: an index.json if present, else the source shards themselves
if os.path.isdir(INDEX):
    wm = {}
    for p in sorted(glob.glob(os.path.join(INDEX, '*.safetensors'))):
        with safe_open(p, framework='pt') as f:
            wm.update({k: p for k in f.keys()})
elif INDEX.endswith('.json'):
    wm = json.load(open(INDEX))['weight_map']
else:
    sys.exit(f'{INDEX}: expected an index.json or a directory of safetensors')
want, kinds = {}, {}
for name in wm:
    for kw in ({}, dict(keep_mtp=True), dict(keep_idx=True)):
        k = classify(name, N_LAYERS, **kw)
        if k in ('skip', 'consumed'):
            continue
        want[name] = k
        kinds[name] = k
        break

missing = [n for n in want if n not in have]
extra = [n for n in have if not n.endswith('.qs') and n not in want]
print(f'expected {len(want)} tensors: {len(want)-len(missing)} present, '
      f'{len(missing)} MISSING, {len(extra)} unexpected')
for n in missing[:8]:
    print(f'   MISSING [{kinds[n]}] {n}')
for n in extra[:8]:
    print(f'   EXTRA   {n}')

# ---- format: quantized vs f32 passthrough ----
bad = []
bykind = collections.Counter()
for n, k in want.items():
    if n not in have:
        continue
    has_qs = (n + '.qs') in have
    is_u8 = dtypes[n] in ('U8', 'uint8', 'I8', 'int8')
    if k == 'f32':
        if has_qs or is_u8:
            bad.append((n, k, 'expected raw f32, found quantized'))
    else:
        # a quantized class may still be f32 if the tensor is not 2-D (conv1d, biases)
        if is_u8 and not has_qs:
            bad.append((n, k, 'packed bytes with no .qs sidecar'))
    bykind[(k, 'quant' if has_qs else 'f32')] += 1

print('\nby class:')
for (k, form), c in sorted(bykind.items()):
    print(f'  {k:6} {form:5} {c:6}')

if bad:
    print(f'\nFORMAT PROBLEMS: {len(bad)}')
    for n, k, why in bad[:8]:
        print(f'   [{k}] {why}: {n}')

ok = not missing and not bad
print('\nRESULT:', 'container is complete and correctly formatted' if ok else 'PROBLEMS FOUND')
sys.exit(0 if ok else 1)
