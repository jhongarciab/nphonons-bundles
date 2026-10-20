#!/bin/bash
# Fig. 7 con rejillas equiespaciadas (20-21 puntos por panel y espera), rtol 1e-9, modelo reducido.
cd "$(dirname "$0")"; source /home/jhon/venv_fem/bin/activate 2>/dev/null; D=../data
export F9_TOL=1e-9 F9_NC=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Q=$(mktemp)
for T in 2000 500 8000; do
  for k in $(seq 1.5 1.5 30); do echo "python3 f17_real_load_reduced.py real $k $T 100 1.0 0 ideal 10 > $D/dn3_kin${k}_T$T.log 2>&1" >> $Q; done
  for k in $(seq 1.5 1.5 30); do echo "python3 f17_real_load_reduced.py real 10 $T 100 1.0 0 real $k > $D/dn3_kout${k}_T$T.log 2>&1" >> $Q; done
  for c in $(seq 0 0.05 1); do echo "python3 f17_real_load_reduced.py real 10 $T 100 $c 0 real 10 > $D/dn3_chi${c}_T$T.log 2>&1" >> $Q; done
done
cat $Q | xargs -P 10 -I{} bash -c "{}"
echo done > $D/dn3_done.flag
