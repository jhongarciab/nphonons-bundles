#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=../.venv/bin/python
KS=$($PY -c "import numpy as np;print(' '.join(f'{x:.5f}' for x in np.logspace(-1,np.log10(20),10)))")
for k in $KS; do o="tarea29bis_gate_cache/k${k}_a24.npz"; [ -f "$o" ] || echo "$k 4 $o"; done \
  | xargs -P 3 -L 1 bash -c "$PY tarea29bis_gate_worker.py \$0 \$1 \$2 > \${2%.npz}.log 2>&1"
echo DONE
