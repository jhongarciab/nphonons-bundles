#!/bin/bash

cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
{
for gzk in 2 3 5 8 12; do echo "42 6 0.05 $gzk 22 cache42/a_gx0.05_gz$gzk.npz"; echo "42 6 0.2 $gzk 22 cache42/b_gx0.2_gz$gzk.npz"; done
for gx in 0.03 0.05 0.1 0.2; do for w in 4 6 8; do echo "42 $w $gx 5 22 cache42/c_w${w}_gx${gx}.npz"; done; done
echo "42 6 0.05 12 28 cache42/a_gx0.05_gz12_N28.npz"
for w in 12.02 12.03 12.04 12.05 12.06; do echo "40 $w 0 0 0 cache40/wp${w}0000.npz"; done
for k1 in 0 0.001; do for g in 0.05 0.1 0.3 1 3; do for nth in 0 0.001 0.01 0.1; do for a2 in 2 4 6; do echo "44 $g $a2 $nth $k1 ../validacion/gautier_cache/g${g}_a${a2}_n${nth}_k${k1}.npz"; done; done; done; done
} | while read t a b c d o; do [ -f "$o" ] || echo "$t $a $b $c $d $o"; done \
 | xargs -P 6 -L 1 bash -c 'case $0 in 42) .venv/bin/python tarea42_worker.py $1 $2 $3 $4 $5 > ${5%.npz}.log 2>&1;; 40) .venv/bin/python tarea40_worker.py $1 $5 > ${5%.npz}.log 2>&1;; 44) (cd ../validacion && ../.venv/bin/python gautier_worker.py $1 $2 $3 $4 ${5#../validacion/} > ${5%.npz}.log 2>&1);; esac'
echo DONE
