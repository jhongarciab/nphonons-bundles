#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2
{ for G in 3.0 1.0 0.5; do for N in 32 26; do echo "$G $N"; done; done; } | while read G N; do [ -f tarea47a_cache/G${G}_N$N.npz ] || echo "$G $N"; done \
 | xargs -P 2 -L 1 bash -c '../.venv/bin/python tarea35_worker.py $0 $1 1 tarea47a_cache/G$0_N$1.npz > tarea47a_cache/G$0_N$1.log 2>&1'
echo DONE_A
