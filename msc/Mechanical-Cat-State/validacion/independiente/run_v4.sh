#!/bin/bash
# V4: puntos (gx w gz/κ N n_periodos_evolucion). Comunes con T42: (0.05,6,5), (0.05,6,12). Resto nuevos.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cat <<'L' | xargs -P 5 -L 1 sh -c '.venv/bin/python v4_espectral.py $0 $1 $2 $3 $4 > res_v4/log_$0_$1_$2_N$3.txt 2>&1'
0.05 6 12 28 0
0.15 6 4 22 40000
0.1 7 4 22 100000
0.05 6 5 22 0
0.05 6 12 22 0
0.05 5 4 22 0
0.05 7 10 22 0
0.03 6 4 22 0
0.08 5 10 22 0
0.12 7 7 22 0
0.1 4 10 22 0
L
