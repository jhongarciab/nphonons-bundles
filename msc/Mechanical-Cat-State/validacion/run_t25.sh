#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
{
for a2 in 5 4 3 2 1; do for gz in 0.12028 0.25039 0.5 0.99913; do
  out="tarea25_cache/gz${gz}_a2${a2}.npz"
  [ -f "$out" ] || echo "$gz $a2 $out"
done; done
} | xargs -P 4 -L 1 bash -c '../.venv/bin/python tarea25_ab_worker.py $0 $1 0.048 0.144 $2 > ${2%.npz}.log 2>&1'
echo DONE
