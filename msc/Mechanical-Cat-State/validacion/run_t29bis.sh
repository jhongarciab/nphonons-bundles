#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=../.venv/bin/python
KS=$($PY -c "import numpy as np;print(' '.join(f'{x:.6f}' for x in np.logspace(np.log10(0.05),np.log10(50),12)))")
for a2 in 2 4 6; do for m in A B; do for k in $KS; do
  o="tarea29bis_cache/${m}_a2${a2}_k${k}.npz"; [ -f "$o" ] || echo "$m $a2 $k $o"
done; done; done | xargs -P 3 -L 1 bash -c "$PY tarea29bis_worker.py \$0 \$1 \$2 \$3 > \${3%.npz}.log 2>&1"
echo DONE
