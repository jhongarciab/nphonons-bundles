#!/bin/bash
# Modo vecino de γ_pf (completo plano, N = 22, κ₂/κ = 0.25): T = 0 y x = 6.86, en paralelo. Tiempos reales en el log.
cd "$(dirname "$0")/.."
PY=/home/jhon/GitHub/trabajo-grado/msc/Mechanical-Cat-State/.venv_qutip5/bin/python
export OMP_NUM_THREADS=3
L=data/filtro_completo/log_plano_modos.txt
( echo "== T=0"; time $PY -W ignore codigo/plano_modos.py ) >> $L.0 2>&1 &
( echo "== x=6.86"; time $PY -W ignore codigo/plano_modos.py 6.86 ) >> $L.1 2>&1 &
wait; cat $L.0 $L.1 > $L; rm $L.0 $L.1; echo FIN > data/filtro_completo/FIN_plano_modos
