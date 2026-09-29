#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
{
for dm in 0.042000 0.045000 0.048000 0.051000 0.054000; do echo "0.5 $dm 0.144000 tarea24_cache/dm${dm}_dq0.144000_ref.npz"; done
for pair in "0.12028 -0.144000" "0.25039 0.072000" "0.99913 0.144000"; do
  set -- $pair
  for dm in 0.045000 0.051000; do echo "$1 $dm $2 tarea24_cache/gz$1_dm${dm}_dq$2.npz"; done
done
} | while read a b c d; do [ -f "$d" ] || echo "$a $b $c $d"; done | xargs -P 6 -L 1 bash -c '../.venv/bin/python tarea24_worker.py $0 3 $1 $2 $3 >/dev/null 2>&1'
echo DONE
