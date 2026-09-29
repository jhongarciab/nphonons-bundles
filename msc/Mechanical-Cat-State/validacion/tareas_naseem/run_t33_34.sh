#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=../.venv/bin/python
{
for G in 0.03 0.08 0.13 0.3 1.0; do echo "33 $G 0 tarea33_cache/G${G}_N20.npz"; done
echo "33 0.13 26 tarea33_cache/G0.13_N26.npz"
for nq in 0 0.1 0.3 0.6; do echo "34 $nq 0 tarea34_cache/nq${nq}_N20.npz"; done
echo "34 0.6 26 tarea34_cache/nq0.6_N26.npz"
} | while read t a n o; do [ -f "$o" ] || echo "$t $a $n $o"; done \
 | xargs -P 3 -L 1 bash -c "$PY tarea\$0_worker.py \$1 \$2 \$3 > \${3%.npz}.log 2>&1"
echo DONE
