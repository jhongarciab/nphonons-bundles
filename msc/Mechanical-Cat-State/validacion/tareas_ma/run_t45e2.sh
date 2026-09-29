#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for s in -4 -5 -6 -8 -10; do [ -f cache45e/r0.03_s$s.npz ] || echo "0.03 $s cache45e/r0.03_s$s.npz"; done \
 | xargs -P 5 -L 1 bash -c '.venv/bin/python tarea45e_worker.py $0 $1 $2 > ${2%.npz}.log 2>&1'
echo DONE_E2
