#!/bin/bash
# Validación térmica con el modelo completo (N = 22, secuencial por memoria). Espera a la corrida en curso.
cd "$(dirname "$0")/.."
PY=/home/jhon/GitHub/trabajo-grado/msc/Mechanical-Cat-State/verificacion_independiente/.venv/bin/python
export OMP_NUM_THREADS=6
until [ -f data/filtro_completo/FIN_termico ]; do sleep 60; done
$PY -W ignore codigo/calc_filtro_completo.py 0.79859571 200.0 14.0 1.0 10.0 4 22 2 --gam=2e-05 --x=6.86 >> data/filtro_completo/log_termico.txt 2>&1   # punto 2, térmico
$PY -W ignore codigo/calc_filtro_completo.py 1.7857143 200.0 14.0 1.0 10.0 4 22 2 --gam=2e-05 >> data/filtro_completo/log_termico.txt 2>&1              # punto 1, T = 0
$PY -W ignore codigo/calc_filtro_completo.py 0.79859571 200.0 14.0 1.0 10.0 4 22 2 --gam=2e-05 >> data/filtro_completo/log_termico.txt 2>&1             # punto 2, T = 0
echo FIN > data/filtro_completo/FIN_validacion_termica
