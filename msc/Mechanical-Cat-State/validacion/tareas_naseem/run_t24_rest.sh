#!/bin/bash
# retoma rejillas 5x5 de Tarea 24; un proceso fresco por celda, 4 en paralelo
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for gz in 0.12028 0.25039 0.99913; do
  for dm in -0.192000 -0.120000 -0.048000 0.024000 0.096000; do
    for dq in -0.144000 -0.072000 0.000000 0.072000 0.144000; do
      out="tarea24_cache/gz${gz}_dm${dm}_dq${dq}.npz"
      [ -f "$out" ] || echo "$gz $dm $dq $out"
    done
  done
done | xargs -P 4 -L 1 bash -c '../.venv/bin/python tarea24_worker.py $0 3 $1 $2 $3 >/dev/null 2>&1'
echo DONE
