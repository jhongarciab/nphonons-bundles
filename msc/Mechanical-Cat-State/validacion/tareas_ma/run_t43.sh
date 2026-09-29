#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
for kf in 0 3 1 0.3 0.1; do echo "$kf"; done | while read kf; do [ -f cache43/kf${kf}_N16.npz ] || echo "$kf"; done \
 | xargs -P 2 -L 1 bash -c '.venv/bin/python tarea43_worker.py $0 16 cache43/kf$0_N16.npz > cache43/kf$0_N16.log 2>&1'
echo DONE43
