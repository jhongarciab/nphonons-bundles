#!/bin/bash
# Validación térmica, fase 3: κ₂/κ = 0.1 y 0.4 a x = 6.86 (y T = 0 para el piso). Modelo completo, N = 22, unidades de Ma.
# Dos cadenas en paralelo (una por κ₂/κ), tolerancia normal (la fase 2 mostró que no importa).
cd "$(dirname "$0")/.."
PY=/home/jhon/GitHub/trabajo-grado/msc/Mechanical-Cat-State/.venv_qutip5/bin/python
export OMP_NUM_THREADS=3
G=6e-07
L=data/filtro_completo/log_termico_ma3.txt
cadena() {  # gx
  ( time $PY -W ignore codigo/calc_filtro_completo.py $1 6.0 0.42 0.03 0.3 4 22 2 --gam=$G --x=6.86 ) >> $L 2>&1
  ( time $PY -W ignore codigo/calc_filtro_completo.py $1 6.0 0.42 0.03 0.3 4 22 2 --gam=$G ) >> $L 2>&1
}
cadena 0.0338816 & cadena 0.0677631 & wait
echo FIN > data/filtro_completo/FIN_validacion_termica3
