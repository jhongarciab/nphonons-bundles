#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
{
for gz in 0.12028 0.25039 0.5 0.99913; do for dq in -0.144000 -0.072000 0.000000 0.072000 0.144000; do
  out="tarea24_cache/dm0.048_gz${gz}_dq${dq}.npz"
  [ -f "$out" ] || echo "$gz 0.048000 $dq $out"
done; done
} | xargs -P 6 -L 1 bash -c '../.venv/bin/python tarea24_worker.py $0 3 $1 $2 $3 >/dev/null 2>&1'
echo DONE
