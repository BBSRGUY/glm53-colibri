"""Compare the engine's k-pool cell selection against the oracle's, set by set.

This isolates the selection from everything downstream: no quantization near-ties, no
top-k tie-break ambiguity in the final token, just "did the indexer pick the same cells".
"""
import json
import os
import re
import subprocess
import sys

import numpy as np

SNAP_IN = 'g_in'
import os as _o; SNAP_OUT = _o.environ.get('SNAP_OUT','D:/GLM5.3-flash/work/g_out')
REF = 'D:/GLM5.3-flash/work/g_kp.json'
ENGINE = 'D:/GLM5.3-flash/colibri/c/glm53.exe'

# ---- engine selections ----
env = dict(os.environ, OMP_NUM_THREADS='1', DSA_DUMP='1', TF='1',
           REF=REF, SNAP=SNAP_OUT)
ARGS=_o.environ.get('ENGINE_ARGS','').split()
out = subprocess.run([ENGINE]+ARGS, env=env, capture_output=True, text=True,
                     cwd='D:/GLM5.3-flash/colibri/c').stderr
eng = {}
for m in re.finditer(r'\[DSA_SEL\] L(\d+) row(\d+) n=(\d+):([0-9 \-]*)', out):
    li, row, n = int(m.group(1)), int(m.group(2)), int(m.group(3))
    cells = [int(t) for t in m.group(4).split()]
    eng[(li, row)] = set(cells)

# ---- oracle selections ----
os.environ['NT'] = '16'
sys.argv = ['fullstack_oracle.py', SNAP_IN, 'tmp_ref.json']
g = {'__name__': '__main__', '__file__': 'fullstack_oracle.py'}
exec(compile(open('fullstack_oracle.py').read(), 'fullstack_oracle.py', 'exec'), g)

full = json.load(open('tmp_ref.json'))['full_ids']
T = len(full)
cap = {}
orig = g['mla']


def spy(li, x):
    cap[li] = x.copy()
    return orig(li, x)


g['mla'] = spy
g['forward'](full)

ok = bad = missing = 0
for (li, row), eset in sorted(eng.items()):
    if li not in cap:
        missing += 1
        continue
    sel = g['dsa_select'](li, cap[li], T)
    oset = set(np.nonzero(sel[row])[0].tolist())
    if eset == oset:
        ok += 1
    else:
        bad += 1
        if bad <= 5:
            print(f'  MISMATCH L{li} row{row}')
            print(f'    engine only: {sorted(eset - oset)}')
            print(f'    oracle only: {sorted(oset - eset)}')

print(f'\nk-pool cell selection: {ok} rows identical, {bad} differ'
      + (f', {missing} unpaired' if missing else ''))
print('RESULT:', 'MATCH' if bad == 0 and ok > 0 else 'MISMATCH')
sys.exit(0 if bad == 0 and ok > 0 else 1)
