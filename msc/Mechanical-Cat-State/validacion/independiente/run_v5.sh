#!/bin/bash
# V5(b): Floquet con drive, |α|²=2 (Tarea 43), filtro de 2 niveles a 2w
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
P=.venv/bin/python
$P v5_filtro.py floquet 0 16 1 2 res_v5/fl_plano_N16.npz > res_v5/log_fl_plano_N16.txt 2>&1 &
$P v5_filtro.py floquet 0 22 1 2 res_v5/fl_plano_N22.npz > res_v5/log_fl_plano_N22.txt 2>&1 &
$P v5_filtro.py floquet 1 16 2 2 res_v5/fl_kf1_N16.npz > res_v5/log_fl_kf1_N16.txt 2>&1 &
$P v5_filtro.py floquet 0.3 16 2 2 res_v5/fl_kf0.3_N16.npz > res_v5/log_fl_kf0.3_N16.txt 2>&1 &
wait
echo FIN > res_v5/FIN
