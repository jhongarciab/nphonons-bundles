#!/bin/bash
# Fase 3, secuencial (dos corridas en paralelo agotaron la memoria de 14 GB). Espera a la corrida (0.1, x = 6.86) ya en marcha.
cd "$(dirname "$0")/.."
PY=/home/jhon/GitHub/trabajo-grado/msc/Mechanical-Cat-State/.venv_qutip5/bin/python
export OMP_NUM_THREADS=6
G=6e-07; L=data/filtro_completo/log_termico_ma3.txt
while pgrep -f "[c]alc_filtro_completo.py 0.0338816" >/dev/null; do sleep 20; done
run() { ( time $PY -W ignore codigo/calc_filtro_completo.py $1 6.0 0.42 0.03 0.3 4 22 2 --gam=$G $2 ) >> $L 2>&1; }
run 0.0338816 ""; run 0.0677631 --x=6.86; run 0.0677631 ""
echo FIN > data/filtro_completo/FIN_validacion_termica3
