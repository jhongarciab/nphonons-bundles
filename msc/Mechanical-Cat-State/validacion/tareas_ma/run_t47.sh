#!/bin/bash
cd "$(dirname "$0")"
PY=.venv/bin/python
( export OMP_NUM_THREADS=2
  { for G in 3.0 1.0 0.5; do for N in 32 26; do echo "$G $N"; done; done; } | while read G N; do [ -f cache47a/G${G}_N$N.npz ] || echo "$G $N"; done \
   | xargs -P 2 -L 1 bash -c '.venv/bin/python tarea35_worker.py $0 $1 1 cache47a/G$0_N$1.npz > cache47a/G$0_N$1.log 2>&1'
  echo DONE_A ) > run_t47a.log 2>&1 &
( export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  { for gzk in 5 12; do echo "c $gzk x"; done
    for r2 in 0.05 0.2 0.5; do for s in -6 -4 -3 -2 -1.5 -1 -0.5 0 0.5 1 1.5 2 3 4 6; do echo "e $r2 $s"; done; done; } \
  | while read t a b; do if [ $t = c ]; then o=cache47c/nocounter_retuned_gz$a.npz; [ -f $o ] || echo "c $a $b $o"; else o=cache45e/r${a}_s${b}.npz; [ -f $o ] || echo "e $a $b $o"; fi; done \
  | xargs -P 2 -L 1 bash -c 'if [ $0 = c ]; then .venv/bin/python tarea45d_worker.py 6 0.05 $1 22 nocounter_retuned $3 > ${3%.npz}.log 2>&1; else .venv/bin/python tarea45e_worker.py $1 $2 $3 > ${3%.npz}.log 2>&1; fi'
  echo DONE_B ) > run_t47b.log 2>&1 &
wait
