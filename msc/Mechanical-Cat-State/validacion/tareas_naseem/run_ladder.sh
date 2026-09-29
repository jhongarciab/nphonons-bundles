#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=../.venv/bin/python
GS=$($PY -c "import numpy as np;print(' '.join(f'{x:.6f}' for x in np.logspace(np.log10(0.005),0,16)))")
{
for G in $GS; do echo "0 $G min 20"; done
for v in min s1 s2 s3 s4 s4L s5 s5L no_dq no_dmlamb no_oneph no_nonres no_kerrp; do for G in $GS; do echo "2 $G $v 20"; done; done
G9=$(echo $GS | awk '{print $10}')
for v in min s5; do echo "2 $G9 $v 26"; done; echo "0 $G9 min 26"
} | while read a2 G v N; do o="ladder_cache/a2${a2}_G${G}_${v}_N${N}.npz"; [ -f "$o" ] || echo "$a2 $G $v $N $o"; done \
 | xargs -P 6 -L 1 bash -c "$PY ladder_worker.py \$0 \$1 \$2 \$3 \$4 > \${4%.npz}.log 2>&1"
echo DONE
