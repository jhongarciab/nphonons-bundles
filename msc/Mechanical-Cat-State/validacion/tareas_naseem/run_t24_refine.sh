#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for pair in "0.12028 -0.144000" "0.25039 0.072000" "0.99913 0.144000"; do
  set -- $pair
  for dm in 0.036000 0.048000 0.060000 0.072000; do
    out="tarea24_cache/gz$1_dm${dm}_dq$2.npz"
    [ -f "$out" ] || echo "$1 $dm $2 $out"
  done
done | xargs -P 6 -L 1 bash -c '../.venv/bin/python tarea24_worker.py $0 3 $1 $2 $3 >/dev/null 2>&1'
echo DONE
