#!/bin/bash
cd "$(dirname "$0")"; source /home/jhon/venv_fem/bin/activate 2>/dev/null; D=../data
for S in 1.0 0.7 0.5 0.33 0.2 0.12; do
 F9_TOL=1e-9 nohup python3 f18_reeval.py $S > $D/f18tol9_reeval_S$S.log 2>&1 &
done
wait
for S in 1.0 0.7 0.5 0.33 0.2 0.12; do
 F9_TOL=1e-9 F18_WARM=$D/old_tol1e6_f18/f18_lambda_S$S.json nohup python3 f18_lambda_scan.py $S > $D/f18tol9_opt_S$S.log 2>&1 &
done
wait
