#!/bin/bash
cd "$(dirname "$0")"
PY=.venv/bin/python
while ! grep -q DONE ../validacion/run_audit2.log 2>/dev/null; do sleep 30; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# ---- Stage B: (d) y (e), 4 en paralelo
{
for v in nocounter nogz_pair; do for gzk in 5 12; do echo "d $v $gzk"; done; done; echo "d nocounter_nogz_pair 12"
for r2 in 1 0.3 0.1 0.03; do for s in -3 -2 -1.5 -1 -0.5 0 0.5 1 1.5 2 3 4 6; do echo "e $r2 $s"; done; done
} | while read t a b; do
  if [ $t = d ]; then o=cache45d/${a}_gz${b}.npz; [ -f $o ] || echo "d $a $b $o"; else o=cache45e/r${a}_s${b}.npz; [ -f $o ] || echo "e $a $b $o"; fi
done | xargs -P 4 -L 1 bash -c 'if [ $0 = d ]; then .venv/bin/python tarea45d_worker.py 6 0.05 $2 22 $1 $3 > ${3%.npz}.log 2>&1; else .venv/bin/python tarea45e_worker.py $1 $2 $3 > ${3%.npz}.log 2>&1; fi'
echo STAGE_B
# ---- Stage C: audit N=38 (heavy, solo)
export OMP_NUM_THREADS=6
[ -f ../validacion/audit_cache/res_G3.0_N38.npz ] || (cd ../validacion && ../.venv/bin/python audit_worker.py 3.0 38 0.048 0.144 2 audit_cache/res_G3.0_N38.npz > audit_cache/res_G3.0_N38.log 2>&1)
echo STAGE_C
# ---- Stage D: (b) pequeño (N=10, Nf=2,3) en paralelo
export OMP_NUM_THREADS=2
{ for kf in 0.3 1; do for nf in 2 3; do echo "$kf $nf"; done; done; } | while read kf nf; do [ -f cache45b/kf${kf}_N10_Nf${nf}.npz ] || echo "$kf $nf"; done \
 | xargs -P 4 -L 1 bash -c '.venv/bin/python tarea43_worker.py $0 10 cache45b/kf$0_N10_Nf$1.npz $1 noevo > cache45b/kf$0_N10_Nf$1.log 2>&1'
echo STAGE_D
# ---- Stage E: (c) N=20 y (b) Nf=4, uno a la vez (8.5 GB cada uno)
export OMP_NUM_THREADS=6
[ -f cache45b/kf0.3_N20_Nf2.npz ] || .venv/bin/python tarea43_worker.py 0.3 20 cache45b/kf0.3_N20_Nf2.npz 2 noevo > cache45b/kf0.3_N20_Nf2.log 2>&1
for kf in 0.3 1; do [ -f cache45b/kf${kf}_N10_Nf4.npz ] || .venv/bin/python tarea43_worker.py $kf 10 cache45b/kf${kf}_N10_Nf4.npz 4 noevo > cache45b/kf${kf}_N10_Nf4.log 2>&1; done
echo DONE45
