#!/bin/bash
# V2: corridas de producción (marco de laboratorio, Floquet sobre T_p = 2π/wp)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
P=.venv/bin/python
( $P v2_lab.py 12.00 22 300 res_v2/wp12.0000_N22.npz ; $P v2_lab.py 11.98 28 60 res_v2/wp11.9800_N28.npz ) > res_v2/log_A.txt 2>&1 &
$P v2_lab.py 11.98 22 300 res_v2/wp11.9800_N22.npz > res_v2/log_B.txt 2>&1 &
( for wp in 11.960 11.965 11.970 11.975 11.9775 11.9825 11.985 11.990 11.995; do
    $P v2_lab.py $wp 22 60 res_v2/barr_wp${wp}_N22.npz; done ) > res_v2/log_C.txt 2>&1 &
( for wp in 11.9785 11.9795 11.9800 11.9806 11.9815 12.000 12.005; do
    $P v2_lab.py $wp 22 60 res_v2/barr_wp${wp}_N22.npz; done ) > res_v2/log_D.txt 2>&1 &
wait
