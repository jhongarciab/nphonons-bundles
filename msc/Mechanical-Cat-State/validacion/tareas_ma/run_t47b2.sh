#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for r2 in 0.5 0.2; do for s in 6 4 3 2 1.5 1 0.5 0 -0.5 -1 -1.5 -2 -3 -4 -6; do echo "$r2 $s"; done; done | while read a b; do o=cache45e/r${a}_s${b}.npz; [ -f $o ] || [ -f cache45e/r${a}_s${b}.log ] || echo "$a $b $o"; done \
 | xargs -P 4 -L 1 bash -c '.venv/bin/python tarea45e_worker.py $0 $1 $2 > ${2%.npz}.log 2>&1'
echo DONE_B2
