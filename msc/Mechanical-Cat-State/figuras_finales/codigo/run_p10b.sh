#!/bin/bash
# P10 (segunda fase): F = modelo completo con filtro (calc_filtro_completo), P = baño plano (calc_pfig3). 3 en paralelo.
cd "$(dirname "$0")/.."
PY=/home/jhon/GitHub/trabajo-grado/msc/Mechanical-Cat-State/verificacion_independiente/.venv/bin/python
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
while read tipo resto; do
  if [ "$tipo" = F ]; then echo "$PY -W ignore codigo/calc_filtro_completo.py $resto"; else echo "$PY -W ignore codigo/calc_pfig3.py $resto"; fi
done < data/p10b_trabajos.txt | xargs -P 3 -I{} sh -c "{} >> data/filtro_completo/log_p10b.txt 2>&1"
echo FIN > data/filtro_completo/FIN_p10b
