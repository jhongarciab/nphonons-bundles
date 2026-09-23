#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=../.venv/bin/python
for G in 0.01 0.03 0.07 0.13 0.24 0.45 0.84 1.6 3.0; do
  out="tarea27_cache/G${G}_a22.npz"; [ -f "$out" ] || echo "$G $out"
done | xargs -P 5 -L 1 bash -c "$PY tarea27_worker.py \$0 2 \$1 > \${1%.npz}.log 2>&1"
echo T27DONE
for a2 in 5 4 3; do for G in 0.13 2.07; do for tol in "1e-12 1e-10" "1e-14 1e-12" "1e-15 1e-13"; do
  set -- $tol; out="tarea26_cache/G${G}_a2${a2}_r$2.npz"
  [ -f "$out" ] || echo "$G $a2 $1 $2 $out"
done; done; done | xargs -P 6 -L 1 bash -c "$PY -c 'import sys;sys.exit(0)'; gz=\$($PY -c \"import modelo_comun as m;print(m.gz_scale_from_Gamma2(\$0))\"); $PY tarea26_worker.py \$gz \$1 \$2 \$3 10 \$4 > \${4%.npz}.log 2>&1"
echo DONE
