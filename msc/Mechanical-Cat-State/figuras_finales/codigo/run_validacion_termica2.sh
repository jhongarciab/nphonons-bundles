#!/bin/bash
# Validación térmica, segunda fase: separar κ₂/κ de x. Modelo completo, N = 22, unidades de Ma, secuencial.
cd "$(dirname "$0")/.."
PY=/home/jhon/GitHub/trabajo-grado/msc/Mechanical-Cat-State/.venv_qutip5/bin/python
export OMP_NUM_THREADS=6
G=6e-07
L=data/filtro_completo/log_termico_ma2.txt
( time $PY -W ignore codigo/calc_filtro_completo.py 0.0535714 6.0 0.42 0.03 0.3 4 22 2 --gam=$G --x=6.86 ) >> $L 2>&1                      # 1) (0.25, x = 6.86)
( time TOL_ESTRICTA=1 $PY -W ignore codigo/calc_filtro_completo.py 0.0535714 6.0 0.42 0.03 0.3 4 22 2 --gam=$G --x=10.1 ) >> $L 2>&1     # 2a) isla, tolerancia estricta
( time TOL_ESTRICTA=1 $PY -W ignore codigo/calc_filtro_completo.py 0.0535714 6.0 0.42 0.03 0.3 4 22 2 --gam=$G ) >> $L 2>&1              # 2b) isla T = 0, estricta
echo FIN > data/filtro_completo/FIN_validacion_termica2
