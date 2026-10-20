#!/bin/bash
# Barrido operacional F17c con tolerancia 1e-9 (Ncut=2) para la figura M2.
cd "$(dirname "$0")"; source /home/jhon/venv_fem/bin/activate 2>/dev/null; D=../data/memory_figs
mkdir -p $D/old_tol1e6; cp -n $D/operational_T*.json $D/old_tol1e6/ 2>/dev/null
for T in 0 50 100 200 350 500 750 1000 1500 2000 3000 5000 8000; do
 F9_TOL=1e-9 F9_NC=2 nohup python3 f17_real_load_reduced.py real 10 $T 100 1.0 0 real 10 > $D/operational_T$T.json 2>/dev/null &
done
wait
