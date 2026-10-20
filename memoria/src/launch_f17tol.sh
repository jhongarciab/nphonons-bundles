#!/bin/bash
cd "$(dirname "$0")"; source /home/jhon/venv_fem/bin/activate 2>/dev/null; D=../data
for T in 500 2000 8000; do
 F9_TOL=1e-9 F9_NC=2 nohup python3 f17_real_load_reduced.py real 10 $T 100 1.0 0 real 10 > $D/f17tol9_ro_real10_kout10_T$T.log 2>&1 &
done
F9_TOL=1e-9 F9_NC=3 nohup python3 f17_real_load_reduced.py real 10 2000 100 1.0 0 real 10 > $D/f17tol9_nc3_ro_real10_kout10_T2000.log 2>&1 &
F9_TOL=1e-9 F9_NC=2 nohup python3 f17_real_load_reduced.py ideal 1 2000 100 1.0 0 ideal 10 > $D/f17tol9_ideal_T2000.log 2>&1 &
wait
