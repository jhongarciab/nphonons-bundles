#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python
WPS=$($PY -c "import numpy as np;print(' '.join(f'{x:.6f}' for x in np.linspace(11.95,12.01,15)))")
{
echo "39 12.0 22 300 cache39/wp12.0_N22.npz"; echo "39 11.98 22 300 cache39/wp11.98_N22.npz"; echo "39 11.98 26 60 cache39/wp11.98_N26.npz"
for w in $WPS; do echo "40 $w 0 0 cache40/wp$w.npz"; done
} | while read t a b c o; do [ -f "$o" ] || echo "$t $a $b $c $o"; done \
 | xargs -P 5 -L 1 bash -c 'if [ $0 = 39 ]; then .venv/bin/python tarea39_worker.py $1 $2 $3 $4 > ${4%.npz}.log 2>&1; else .venv/bin/python tarea40_worker.py $1 $4 > ${4%.npz}.log 2>&1; fi'
echo DONE
