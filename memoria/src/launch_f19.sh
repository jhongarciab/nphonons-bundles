#!/bin/bash
# F19: modelo completo con biexciton y ambos puertos reales (rtol 1e-9). Control proyectado = reducido.
cd "$(dirname "$0")"; mkdir -p ../data
export F14_TOL=1e-9 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
F16_PROJ=1 setsid nohup python3 f19_full_real_ports.py -9 > ../data/f19_proj_dxxm9.log 2>&1 &
setsid nohup python3 f19_full_real_ports.py -9 > ../data/f19_full_dxxm9.log 2>&1 &
setsid nohup python3 f19_full_real_ports.py -6 > ../data/f19_full_dxxm6.log 2>&1 &
