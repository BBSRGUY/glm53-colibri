"""Fetch every safetensors shard header for GLM-5.3-Flash via HTTP range reads.

A safetensors file starts with u64-LE header length, then that many bytes of JSON
holding {name: {dtype, shape, data_offsets}} for every tensor in the shard. So the
full 76,108-tensor manifest WITH shapes and dtypes costs ~20 MB of range requests
instead of a 328 GB download. Cached to headers.json.
"""
import json, os, struct, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "headers.json")

def get(url, lo, hi):
    r = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
    return urllib.request.urlopen(r, timeout=120).read()

def header(shard):
    url = BASE + shard
    n = struct.unpack("<Q", get(url, 0, 7))[0]
    return json.loads(get(url, 8, 8 + n - 1))

def main():
    idx = json.load(open(os.path.join(HERE, "..", "glm53_index.json")))["weight_map"]
    shards = sorted(set(idx.values()))
    print(f"{len(shards)} shards, {len(idx)} tensors")
    out = {}
    with ThreadPoolExecutor(8) as ex:
        for i, (s, h) in enumerate(zip(shards, ex.map(header, shards)), 1):
            h.pop("__metadata__", None)
            out.update(h)
            sys.stdout.write(f"\r  {i}/{len(shards)}  ({len(out)} tensors)")
            sys.stdout.flush()
    print()
    json.dump(out, open(OUT, "w"))
    print(f"wrote {OUT}  ({len(out)} tensors, {os.path.getsize(OUT)/1e6:.1f} MB)")

main()
