#!/bin/bash
# Barridos densos para Figs 6-9: puertos/ruido (reducido, 3 T), deformacion S (pulsos reoptimizados), biexciton (completo). Cola con xargs -P12, lo largo primero.
cd "$(dirname "$0")"; source /home/jhon/venv_fem/bin/activate 2>/dev/null; D=../data
export F9_TOL=1e-9 F9_NC=2 F14_TOL=1e-9 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Q=$(mktemp)
for x in -2 -4 -5 -7 -8 -10 -15; do echo "python3 f19_full_real_ports.py $x > $D/f19_full_dxxm${x#-}.log 2>&1" >> $Q; done
for p in "0.1 0.12" "0.15 0.12" "0.25 0.2" "0.3 0.33" "0.4 0.33" "0.6 0.5" "0.8 0.7" "0.9 1.0"; do set -- $p
  echo "F18_WARM=$D/f18_lambda_S$2.json python3 f18_lambda_scan.py $1 > $D/f18dense_opt_S$1.log 2>&1" >> $Q; done
for T in 500 2000 8000; do
  for k in 1 1.5 2 3 4 5 6 8 10 13 16 20 25 30; do echo "python3 f17_real_load_reduced.py real $k $T 100 1.0 0 ideal 10 > $D/dn_kin${k}_T$T.log 2>&1" >> $Q; done
  for k in 1.5 2 3 4 5 7 10 14 20 30; do echo "python3 f17_real_load_reduced.py real 10 $T 100 1.0 0 real $k > $D/dn_kout${k}_T$T.log 2>&1" >> $Q; done
  for c in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do echo "python3 f17_real_load_reduced.py real 10 $T 100 $c 0 real 10 > $D/dn_chi${c}_T$T.log 2>&1" >> $Q; done
done
cat $Q | xargs -P 12 -I{} bash -c "{}"
echo done > $D/dense_done.flag
