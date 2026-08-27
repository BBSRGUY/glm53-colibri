"""Dry-run the patched converter's tensor routing against the REAL GLM-5.3-Flash manifest.

Validates two things before committing to a multi-day, 328 GB conversion:
  1. COVERAGE - every one of the 76,108 tensors is claimed by exactly one pass
     (main / --mtp / --indexer) or is a consumed FP8 scale sidecar. A tensor that
     no pass claims is a tensor the engine will look for and not find.
  2. SIZE - exact output bytes, computed from real shapes with the same packing
     math the converter uses, so the disk/RAM budget is measured, not guessed.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "colibri", "c", "tools"))
from convert_fp8_to_int4 import classify, layer_idx          # the patched module

N_LAYERS = 45                                                 # text_config.num_hidden_layers
import os
GS       = int(os.environ.get("GS","64"))                                                 # --group-size

# --- packed size, mirroring quant_int8 / quant_int4_grouped / quant_int3_g64 ---
def packed(shape, bits, ndim_ok):
    n = 1
    for d in shape: n *= d
    if not ndim_ok or bits >= 32: return n * 4                # f32 passthrough
    O, I = shape[0], n // shape[0]
    if bits == 3:
        ng = (I + 63) // 64
        return O * ng * 24 + O * ng * 4
    if bits <= 4:
        ng = (I + GS - 1) // GS
        return O * ((I + 1) // 2) + O * ng * 4                # nibbles + f32 group scales
    return n + O * 4                                          # int8 + f32 row scales

BITS = {"x": 4, "sh": 8, "o": 8, "kvb": 8, "attn": 8,
        "kda": 8, "dmlp": 8, "io": 8, "vis": 8, "q": 8}

# streamed from disk vs resident in RAM
STREAMED = {"x"}

def main():
    H = json.load(open(os.path.join(HERE, "headers.json")))
    passes = {"main": dict(), "mtp": dict(keep_mtp=True), "idx": dict(keep_idx=True)}
    claim, kinds = {}, {}
    for name in H:
        for pname, kw in passes.items():
            k = classify(name, N_LAYERS, **kw)
            if k in ("skip", "consumed"): continue
            claim.setdefault(name, []).append((pname, k))
            kinds.setdefault((pname, k), []).append(name)

    consumed = [n for n in H if classify(n, N_LAYERS) == "consumed"]
    unclaimed = [n for n in H if n not in claim and n not in consumed]
    dup = {n: v for n, v in claim.items() if len(v) > 1}

    print(f"tensors in checkpoint : {len(H)}")
    print(f"  consumed FP8 scales : {len(consumed)}")
    print(f"  claimed by a pass   : {len(claim)}")
    print(f"  UNCLAIMED           : {len(unclaimed)}   <-- must be 0")
    print(f"  claimed twice       : {len(dup)}   <-- must be 0")
    if unclaimed:
        for n in unclaimed[:15]: print("      !", n)
    if dup:
        for n, v in list(dup.items())[:15]: print("      !", n, v)

    print(f"\n{'pass':6} {'kind':6} {'tensors':>8} {'params':>14} {'out bytes':>14}")
    print("-" * 54)
    tot_res = tot_str = 0
    for (pname, k), names in sorted(kinds.items()):
        nparam = outb = 0
        for n in names:
            sh = H[n]["shape"]; nn = 1
            for d in sh: nn *= d
            nparam += nn
            b = BITS.get(k, 32) if k != "f32" else 32
            outb += packed(sh, b, len(sh) == 2)
        print(f"{pname:6} {k:6} {len(names):8} {nparam/1e9:11.3f} B {outb/1e9:11.3f} GB")
        (globals(), None)
        if k in STREAMED: tot_str += outb
        else:            tot_res += outb
    print("-" * 54)
    print(f"{'':13} {'RESIDENT (RAM)':>22} {tot_res/1e9:11.3f} GB")
    print(f"{'':13} {'STREAMED (disk)':>22} {tot_str/1e9:11.3f} GB")
    print(f"{'':13} {'TOTAL on disk':>22} {(tot_res+tot_str)/1e9:11.3f} GB")

main()
