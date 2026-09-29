#!/bin/bash
# Fig. 4: puntos Floquet de data/fig4/trabajos_fl.txt (6 en paralelo, 2 hilos); los cacheados se saltan.
cd "$(dirname "$0")/.."        # se ejecuta desde figuras_finales/
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=/home/jhon/GitHub/trabajo-grado/msc/Mechanical-Cat-State/.venv_qutip5/bin/python
xargs -P 6 -L 1 sh -c "$PY codigo/calc_fig4.py \$0 \$1 \$2 \$3 \$4 >> data/fig4/log_fl.txt 2>&1" < ${1:-data/fig4/trabajos_fl.txt}
echo FIN > data/fig4/FIN
