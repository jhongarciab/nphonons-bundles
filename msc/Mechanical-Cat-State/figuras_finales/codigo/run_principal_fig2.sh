#!/bin/bash
# Figura principal 2: puntos de data/pfig2/orden.txt (8 en paralelo, 1 hilo); los cacheados se saltan.
cd "$(dirname "$0")/.."        # se ejecuta desde figuras_finales/
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/jhon/GitHub/trabajo-grado/msc/Mechanical-Cat-State/.venv_qutip5/bin/python
xargs -P 8 -L 1 sh -c "$PY -W ignore codigo/calc_principal_fig2.py \$0 \$1 \$2 \$3 \$4 \$5 \$6 \$7 >> data/pfig2/log.txt 2>&1" < ${1:-data/pfig2/orden.txt}
echo FIN > data/pfig2/FIN
