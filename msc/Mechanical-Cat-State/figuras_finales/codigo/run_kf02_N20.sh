#!/bin/bash
# Pareja térmica + T = 0 del completo con filtro a κ_f/ω = 0.2 (κ_f = 1.2), N = 20, (κ₂/κ, x) = (0.25, 6.86). Ver data/preregistro_kf0.2.md. Una corrida a la vez.
cd "$(dirname "$0")/.."
PY=/home/jhon/GitHub/trabajo-grado/msc/Mechanical-Cat-State/.venv_qutip5/bin/python
export OMP_NUM_THREADS=6
L=data/filtro_completo/log_termico_ma7.txt
run() {  # argumentos extra
  echo "== $(date '+%F %T') kf=1.2 N=20 $1" >> $L
  /usr/bin/time -f "TIEMPO_REAL=%e s  MEMORIA_MAX=%M kB" $PY -W ignore codigo/calc_filtro_completo.py 0.0535714 6.0 0.42 0.03 1.2 4 20 2 --gam=6e-07 $1 >> $L 2>&1
  echo "== fin $(date '+%F %T')" >> $L
}
run --x=6.86
run ""
echo FIN > data/filtro_completo/FIN_kf02_N20
