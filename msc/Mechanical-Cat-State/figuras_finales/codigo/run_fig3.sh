#!/bin/bash
# Fig. 3: lanza los puntos de data/fig3/trabajos.txt (5 en paralelo, 2 hilos); los cacheados se saltan.
cd "$(dirname "$0")/.."        # se ejecuta desde figuras_finales/
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=/home/jhon/GitHub/trabajo-grado/msc/Mechanical-Cat-State/verificacion_independiente/.venv/bin/python
xargs -P 5 -L 1 sh -c "$PY codigo/calc_fig3.py \$0 \$1 \$2 \$3 \$4 >> data/fig3/log.txt 2>&1" < ${1:-data/fig3/trabajos.txt}
echo FIN > data/fig3/FIN
