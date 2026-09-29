#!/bin/bash
# V4b: g_x=0.05, ω=6, g_z/κ=4 con |α|² = 2, 4, 6 (Ω = |α|²G), método espectral
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
AL2=2 .venv/bin/python v4_espectral.py 0.05 6 4 22 0 res_v4/b_al2_N22.npz > res_v4/logb_al2_N22.txt 2>&1 &
AL2=4 .venv/bin/python v4_espectral.py 0.05 6 4 22 0 res_v4/b_al4_N22.npz > res_v4/logb_al4_N22.txt 2>&1 &
AL2=6 .venv/bin/python v4_espectral.py 0.05 6 4 26 0 res_v4/b_al6_N26.npz > res_v4/logb_al6_N26.txt 2>&1 &
AL2=6 .venv/bin/python v4_espectral.py 0.05 6 4 32 0 res_v4/b_al6_N32.npz > res_v4/logb_al6_N32.txt 2>&1 &
wait
