#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=../.venv/bin/python
G23=$($PY -c "import numpy as np;print(' '.join(f'{x:.6f}' for x in np.logspace(-2,np.log10(3),10)))")
G28=$($PY -c "import numpy as np;print(' '.join(f'{x:.6f}' for x in np.logspace(np.log10(0.02),0,30)))")
{
for a2 in 2 3; do for f in 0.1 0.2 0.5 0.75 1 1.5 2; do for T in 10 20 50; do echo "37w $a2 $f $T 0 tarea37_cache/a2${a2}_f${f}_T${T}.npz"; done; done; done
echo "37w 2 0.1 50 26 tarea37_cache/a2_2_hot_Nplus.npz"; echo "37w 3 0.1 50 30 tarea37_cache/a2_3_hot_Nplus.npz"
for a2 in 2 3; do for pt in "2 10" "0.5 20" "0.1 50"; do set -- $pt; echo "37c $a2 $1 $2 0 tarea37_cache/chk_a2${a2}_f$1_T$2.npz"; done; done
for N in 20 26; do
  for G in $G23 0.02986 0.1296 0.5184 2.0736; do echo "38a $G $N 0 0 audit_cache/base_G${G}_N${N}.npz"; done
  for G in 0.01 0.03 0.07 0.13 0.24 0.45 0.84 1.6 3.0 $G28; do echo "38a $G $N 0.048 0.144 audit_cache/res_G${G}_N${N}.npz"; done
done
} | while read t a b c d o; do [ -f "$o" ] || echo "$t $a $b $c $d $o"; done \
 | xargs -P 5 -L 1 bash -c 'case $0 in 37w) ../.venv/bin/python tarea37_worker.py $1 $2 $3 $4 $5 > ${5%.npz}.log 2>&1;; 37c) ../.venv/bin/python tarea37_check.py $1 $2 $3 $5 > ${5%.npz}.log 2>&1;; 38a) ../.venv/bin/python audit_worker.py $1 $2 $3 $4 2 $5 > ${5%.npz}.log 2>&1;; esac'
echo DONE
