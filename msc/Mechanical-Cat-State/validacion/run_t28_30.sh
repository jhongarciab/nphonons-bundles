#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=../.venv/bin/python
grid() { $PY -c "import numpy as np,sys;print(' '.join(f'{x:.6f}' for x in np.logspace(np.log10($1),np.log10($2),$3)))"; }
for G in $(grid 0.02 1 30); do out="tarea28_cache/G${G}.npz"; [ -f "$out" ] || echo "$G 2 $out"; done \
  | xargs -P 6 -L 1 bash -c "$PY tarea28_worker.py \$0 \$1 \$2 > \${2%.npz}.log 2>&1"
echo T28DONE
for a2 in 2 3 4; do for G in $(grid 0.005 1 12); do out="tarea29_cache/G${G}_a2${a2}.npz"; [ -f "$out" ] || echo "$G $a2 $out"; done; done \
  | xargs -P 6 -L 1 bash -c "$PY tarea27_worker.py \$0 \$1 \$2 > \${2%.npz}.log 2>&1"
echo T29DONE
for a2 in 5 4 3; do for cfg in "0.000 1.5e-4" "0.144 1.5e-4" "0.288 1.5e-4" "0.144 0"; do set -- $cfg; out="tarea30_cache/a2${a2}_dq$1_gm$2.npz"; [ -f "$out" ] || echo "$a2 $1 $2 $out"; done; done \
  | xargs -P 6 -L 1 bash -c "$PY tarea30_worker.py 0.13 \$0 \$1 \$2 \$3 > \${3%.npz}.log 2>&1"
echo DONE
