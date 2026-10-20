#!/bin/bash
# Rejilla uniforme de S (paso 0.05) para el panel de deformacion: S nuevos y repeticion de 0.25 y 0.4 con otro arranque (la mejor F_search se elige al dibujar).
cd "$(dirname "$0")"; source /home/jhon/venv_fem/bin/activate 2>/dev/null; D=../data
export F9_TOL=1e-9 F9_NC=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $D/f18_prev; cp -n $D/f18_lambda_S0.25.json $D/f18_lambda_S0.4.json $D/f18_prev/ 
cp $D/f18_lambda_S0.3.json $D/f18_warm_for_0.25.json; cp $D/f18_lambda_S0.5.json $D/f18_warm_for_0.4.json
Q=$(mktemp)
for p in "0.35 0.3" "0.45 0.5" "0.55 0.5" "0.65 0.7" "0.75 0.7" "0.85 0.9" "0.95 1.0"; do set -- $p
  echo "F18_WARM=$D/f18_lambda_S$2.json python3 f18_lambda_scan.py $1 > $D/f18dense_opt_S$1.log 2>&1" >> $Q; done
echo "F18_WARM=$D/f18_warm_for_0.25.json python3 f18_lambda_scan.py 0.25 > $D/f18dense_opt_S0.25b.log 2>&1" >> $Q
echo "F18_WARM=$D/f18_warm_for_0.4.json python3 f18_lambda_scan.py 0.4 > $D/f18dense_opt_S0.4b.log 2>&1" >> $Q
cat $Q | xargs -P 9 -I{} bash -c "{}"
echo done > $D/strain2_done.flag
