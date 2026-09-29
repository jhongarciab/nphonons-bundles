#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
{ echo "nogz_pair 5"; echo "nogz_pair 12"; echo "nocounter_nogz_pair 12"; } | while read v g; do [ -f cache45d/${v}_gz$g.npz ] || echo "$v $g"; done \
 | xargs -P 3 -L 1 bash -c '.venv/bin/python tarea45d_worker.py 6 0.05 $1 22 $0 cache45d/$0_gz$1.npz > cache45d/$0_gz$1.log 2>&1'
echo DONE_D2
