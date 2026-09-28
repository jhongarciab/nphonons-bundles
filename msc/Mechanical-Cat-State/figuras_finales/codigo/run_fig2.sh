#!/bin/bash
# Fig. 2: lanza todos los puntos de data/fig2/trabajos.txt (10 en paralelo, 1 hilo cada uno); los ya cacheados se saltan.
cd "$(dirname "$0")/.."        # se ejecuta desde figuras_finales/
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/jhon/GitHub/trabajo-grado/msc/Mechanical-Cat-State/verificacion_independiente/.venv/bin/python
xargs -P 10 -L 1 sh -c "$PY codigo/calc_fig2.py \$0 \$1 \$2 \$3 \$4 >> data/fig2/log.txt 2>&1" < ${1:-data/fig2/trabajos.txt}
echo FIN > data/fig2/FIN
