#!/bin/bash
# Fase 6: (1) independencia de x del cociente completo/efectivo: κ₂/κ = 0.05, 0.1 a x = 9 y κ₂/κ = 0.4 a x = 9.5 (piso esperado 1–3%);
# (2) barrido en N con filtro: κ₂/κ = 0.25, x = 6.86, N = 24. Una corrida a la vez (memoria). Tiempo real y memoria máxima por corrida en el log.
cd "$(dirname "$0")/.."
PY=/home/jhon/GitHub/trabajo-grado/msc/Mechanical-Cat-State/.venv_qutip5/bin/python
export OMP_NUM_THREADS=6
G=6e-07; L=data/filtro_completo/log_termico_ma4.txt
run() {  # gx N x
  echo "== $(date '+%F %T') gx=$1 N=$2 x=$3" >> $L
  /usr/bin/time -f "TIEMPO_REAL=%e s  MEMORIA_MAX=%M kB" $PY -W ignore codigo/calc_filtro_completo.py $1 6.0 0.42 0.03 0.3 4 $2 2 --gam=$G --x=$3 >> $L 2>&1
  echo "== fin $(date '+%F %T')" >> $L
}
run 0.0239579 22 9.0; run 0.0338816 22 9.0; run 0.0677631 22 9.5; run 0.0535714 24 6.86
echo FIN > data/filtro_completo/FIN_validacion_termica4
