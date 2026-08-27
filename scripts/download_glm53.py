"""Download zai-org/GLM-5.3-Flash (~328 GB) to D:/GLM5.3-flash/GLM-5.3-Flash.

Resumable: hf_hub caches by content hash, so re-running after an interruption fetches
only what is missing. Keeping the raw FP8 checkpoint (rather than streaming straight
through the converter) means the quantization recipe can be changed -- gs64 vs gs128,
mixed widths -- without re-downloading 328 GB.
"""
import os
import sys
import time

from huggingface_hub import snapshot_download

REPO = "zai-org/GLM-5.3-Flash"
DEST = "D:/GLM5.3-flash/GLM-5.3-Flash"

t0 = time.time()
path = snapshot_download(
    REPO,
    local_dir=DEST,
    allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model", "LICENSE"],
    max_workers=8,
    resume_download=True,
)
dt = time.time() - t0
total = sum(os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(DEST) for f in fs)
print(f"\nDONE {path}")
print(f"  {total/1e9:.1f} GB in {dt/60:.1f} min ({total/dt/1e6:.0f} MB/s)")
