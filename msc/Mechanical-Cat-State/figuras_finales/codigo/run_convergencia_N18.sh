#!/bin/bash
# Convergencia en N con filtro: κ₂/κ = 0.25, x = 6.86, N = 18 (N_f = 2). Tiempo real y memoria máxima con /usr/bin/time.
cd "$(dirname "$0")/.."
PY=/home/jhon/GitHub/trabajo-grado/msc/Mechanical-Cat-State/.venv_qutip5/bin/python
export OMP_NUM_THREADS=6
L=data/filtro_completo/log_termico_ma6.txt
echo "== $(date '+%F %T') gx=0.0535714 N=18 x=6.86" >> $L
/usr/bin/time -f "TIEMPO_REAL=%e s  MEMORIA_MAX=%M kB" $PY -W ignore codigo/calc_filtro_completo.py 0.0535714 6.0 0.42 0.03 0.3 4 18 2 --gam=6e-07 --x=6.86 >> $L 2>&1
echo "== fin $(date '+%F %T')" >> $L
echo FIN > data/filtro_completo/FIN_convergencia_N18
