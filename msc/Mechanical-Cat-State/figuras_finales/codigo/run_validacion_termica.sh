#!/bin/bash
# Validación térmica con el modelo completo, N = 22, en UNIDADES DE Ma (κ = 0.03, ω = 6): las corridas en κ = 1 (ω = 200)
# dieron un modo de pozo no convergido y hermiticidad fuera de tolerancia (tolerancia absoluta del integrador frente a
# frecuencias 200× mayores). Secuencial por memoria (~11.6 GB por corrida).
cd "$(dirname "$0")/.."
PY=/home/jhon/GitHub/trabajo-grado/msc/Mechanical-Cat-State/.venv_qutip5/bin/python
export OMP_NUM_THREADS=6
G=6e-07   # γ = 2e-5 κ
$PY -W ignore codigo/calc_filtro_completo.py 0.0535714 6.0 0.42 0.03 0.3 4 22 2 --gam=$G --x=10.1 >> data/filtro_completo/log_termico_ma.txt 2>&1   # isla, térmico
$PY -W ignore codigo/calc_filtro_completo.py 0.0535714 6.0 0.42 0.03 0.3 4 22 2 --gam=$G >> data/filtro_completo/log_termico_ma.txt 2>&1            # isla, T = 0
$PY -W ignore codigo/calc_filtro_completo.py 0.0239579 6.0 0.42 0.03 0.3 4 22 2 --gam=$G --x=6.86 >> data/filtro_completo/log_termico_ma.txt 2>&1   # κ₂/κ = 0.05, térmico
$PY -W ignore codigo/calc_filtro_completo.py 0.0239579 6.0 0.42 0.03 0.3 4 22 2 --gam=$G >> data/filtro_completo/log_termico_ma.txt 2>&1            # κ₂/κ = 0.05, T = 0
echo FIN > data/filtro_completo/FIN_validacion_termica
