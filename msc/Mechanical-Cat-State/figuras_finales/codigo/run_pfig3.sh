#!/bin/bash
# Fig. 3 principal (puntos del modelo completo, 3 en paralelo x 2 hilos) y P2 (N=22, 5 en paralelo x 1 hilo)
cd "$(dirname "$0")/.."        # se ejecuta desde figuras_finales/
PY=/home/jhon/GitHub/trabajo-grado/msc/Mechanical-Cat-State/verificacion_independiente/.venv/bin/python
( OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 xargs -P 3 -L 1 sh -c "$PY -W ignore codigo/calc_pfig3.py \$0 \$1 \$2 \$3 \$4 >> data/pfig3/log.txt 2>&1" < data/pfig3/trabajos.txt; echo FIN > data/pfig3/FIN ) &
( OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 xargs -P 5 -L 1 sh -c "$PY -W ignore codigo/calc_principal_fig2.py \$0 \$1 \$2 \$3 \$4 \$5 \$6 \$7 >> data/pfig2/log_n22.txt 2>&1" < data/pfig2/trabajos_n22.txt; echo FIN > data/pfig2/FIN_n22 ) &
wait
