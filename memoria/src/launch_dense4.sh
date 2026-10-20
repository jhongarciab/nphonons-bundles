#!/bin/bash
# Fig. 7 (a-c): 18 puntos equiespaciados por panel y espera (kin, kout = 1.5k, k=1..18; chi = k/17), rtol 1e-9, modelo reducido. Omite lo ya calculado.
cd "$(dirname "$0")"; source /home/jhon/venv_fem/bin/activate 2>/dev/null; D=../data
export F9_TOL=1e-9 F9_NC=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
Q=$(mktemp)
add() { [ -s "$D/$2" ] && grep -q mineig "$D/$2" || echo "$1 > $D/$2 2>&1" >> $Q; }
for T in 2000 500 8000; do
  for k in $(seq 1 18); do v=$(python3 -c "print(f'{1.5*$k:.1f}')")
    add "python3 f17_real_load_reduced.py real $v $T 100 1.0 0 ideal 10" "dn3_kin${v}_T$T.log"
    add "python3 f17_real_load_reduced.py real 10 $T 100 1.0 0 real $v" "dn3_kout${v}_T$T.log"; done
  for k in $(seq 0 17); do c=$(python3 -c "print(f'{$k/17:.4f}')")
    add "python3 f17_real_load_reduced.py real 10 $T 100 $c 0 real 10" "dn4_chi${c}_T$T.log"; done
done
wc -l < $Q > $D/dn4_queue_size.txt
cat $Q | xargs -P 12 -I{} bash -c "{}"
echo done > $D/dn4_done.flag
