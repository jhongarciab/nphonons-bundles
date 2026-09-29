#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=../.venv/bin/python
NQS=$($PY -c "import numpy as np;print(' '.join(f'{x:.6g}' for x in np.logspace(-4,np.log10(0.3),10)))")
{
for G in 0.13 0.3 1.0; do echo "35 $G 20 1 tarea35_cache/G${G}_N20.npz"; done
for G in 0.13 0.3 1.0; do echo "35 $G 26 0 tarea35_cache/G${G}_N26.npz"; done
echo "35 0.13 26 1 tarea35_cache/G0.13_N26dyn.npz"
for G in 0.13 0.3 1.0; do echo "35 $G 32 0 tarea35_cache/G${G}_N32.npz"; done
for a2 in 2 3; do for q in $NQS; do echo "36 $a2 $q 0 tarea36_cache/a2${a2}_nq${q}_N0.npz"; done; done
for a2 in 2 3; do echo "36 $a2 0.3 $([ $a2 = 2 ] && echo 26 || echo 30) tarea36_cache/a2${a2}_nq0.3_Nplus.npz"; done
} | while read t a b c o; do [ -f "$o" ] || echo "$t $a $b $c $o"; done \
 | xargs -P 6 -L 1 bash -c 'if [ $0 = 35 ]; then ../.venv/bin/python tarea35_worker.py $1 $2 $3 $4 > ${4%.npz}.log 2>&1; else ../.venv/bin/python tarea36_worker.py $1 $2 $3 $4 > ${4%.npz}.log 2>&1; fi'
echo DONE
