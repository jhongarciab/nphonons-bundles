#!/bin/bash
# Datos para las figuras M7 (ruido y puertos) y M8 (biexciton): reducido 1e-9, y completo dxx=-3,-12 a T=2000.
cd "$(dirname "$0")"; source /home/jhon/venv_fem/bin/activate 2>/dev/null; D=../data
export F9_TOL=1e-9 F9_NC=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for c in 0.25 0.5 0.75; do setsid nohup python3 f17_real_load_reduced.py real 10 2000 100 $c 0 real 10 > $D/secV_chi${c}_real.log 2>&1 & done
for k in 2 3; do setsid nohup python3 f17_real_load_reduced.py real 10 2000 100 1.0 0 real $k > $D/secV_kout${k}_real.log 2>&1 & done
python3 biex_eff_scan.py > $D/biex_eff_scan.log 2>&1 &
export F14_TOL=1e-9 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for x in -3 -12; do setsid nohup python3 f19_full_real_ports.py $x > $D/f19_full_dxxm${x#-}.log 2>&1 & done
