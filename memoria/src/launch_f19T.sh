#!/bin/bash
# F19b: modelo completo (dxx=-9) con puertos reales a T=500 y T=8000 (rtol 1e-9), para la Fig. 6.
cd "$(dirname "$0")"; mkdir -p ../data
export F14_TOL=1e-9 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
for T in 500 8000; do
  setsid nohup python3 f19_full_real_ports.py -9 $T > ../data/f19_full_dxxm9_T$T.log 2>&1 &
done
