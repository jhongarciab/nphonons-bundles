#!/bin/bash
# Baño plano (completo, Nf = 1), κ₂/κ = 0.25, x = 6.86: (1) apagar n_q o n_m por separado; (2) barrido en N = 18, 22, 26 (térmico y T = 0).
cd "$(dirname "$0")/.."
PY=/home/jhon/GitHub/trabajo-grado/msc/Mechanical-Cat-State/.venv_qutip5/bin/python
export OMP_NUM_THREADS=2
L=data/filtro_completo/log_plano_control.txt
run() { ( time env $1 $PY -W ignore codigo/calc_filtro_completo.py 0.0535714 6.0 0.42 0.03 0.3 4 $2 1 --variante=plano --gam=6e-07 $3 ) >> $L 2>&1; }
( run NQ_OFF=1 22 --x=6.86; run NM_OFF=1 22 --x=6.86 ) &
( run A=1 18 --x=6.86; run A=1 18 ""; run A=1 26 "" ) &
( run A=1 26 --x=6.86 ) &
wait; echo FIN > data/filtro_completo/FIN_plano_control
