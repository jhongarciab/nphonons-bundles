#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
{ echo "3.0 32 res_G3.0_N32"; echo "1.6 32 res_G1.6_N32"; echo "3.0 26 res_G3.0_N26p"; } | while read G N n; do [ -f audit_cache/$n.npz ] || echo "$G $N $n"; done \
 | xargs -P 2 -L 1 bash -c '../.venv/bin/python audit_worker.py $0 $1 0.048 0.144 2 audit_cache/$2.npz > audit_cache/$2.log 2>&1'
echo DONE
