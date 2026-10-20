#!/bin/bash
# (1) Tolerancia a respuesta Stark desigual: ruido comun (chi=1) con amplitudes a=(1+d,1,1) [A] o (1,1+d,1-d) [B]; T=2000 y 8000, d=0..0.3 paso 0.025.
# (2) Rejilla de chi en T=1000 y 4000 para el contorno F_h(chi,T). Baja prioridad (nice) para no frenar las optimizaciones de S.
cd "$(dirname "$0")"; source /home/jhon/venv_fem/bin/activate 2>/dev/null; D=../data
export F9_TOL=1e-9 F9_NC=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Q=$(mktemp)
for T in 2000 8000; do for k in $(seq 0 12); do d=$(python3 -c "print(f'{0.025*$k:.3f}')"); for p in A B; do
  echo "F9_STARK=$p:$d nice -n 19 python3 f17_real_load_reduced.py real 10 $T 100 1.0 0 real 10 > $D/stark_${p}_${d}_T$T.log 2>&1" >> $Q; done; done; done
for T in 1000 4000; do for k in $(seq 0 17); do c=$(python3 -c "print(f'{$k/17:.4f}')")
  echo "nice -n 19 python3 f17_real_load_reduced.py real 10 $T 100 $c 0 real 10 > $D/dn4_chi${c}_T$T.log 2>&1" >> $Q; done; done
cat $Q | xargs -P 6 -I{} bash -c "{}"
echo done > $D/stark_done.flag
