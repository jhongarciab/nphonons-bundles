#!/bin/bash
# Rejilla uniforme de esperas para M2 (T = 400..7600 paso 400), rtol 1e-9, Ncut=2.
cd "$(dirname "$0")"; source /home/jhon/venv_fem/bin/activate 2>/dev/null; D=../data/memory_figs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 F9_TOL=1e-9 F9_NC=2
seq 400 400 7600 | xargs -P 6 -I{} bash -c "python3 f17_real_load_reduced.py real 10 {} 100 1.0 0 real 10 > $D/operational_T{}.json 2>/dev/null"
echo done > $D/opsweep2_done.flag
