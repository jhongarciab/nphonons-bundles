# -*- coding: utf-8 -*-
"""Uso: gautier_worker.py <g2/kappa> <alpha2> <nth> <kappa1/kappa> <outfile>"""
import sys, numpy as np, modelo_gautier as mg, modelo_ladder as ml
g2, a2, nth, k1, out = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
L, N = mg.build(g2, a2, nth, k1); r = ml.analizar(L, N, a2)
np.savez(out, g2=g2, alpha2=a2, nth=nth, kappa1=k1, N=N, **{k: v for k, v in r.items() if k not in ('wcode',)})
print(f"OK g2={g2} a2={a2} nth={nth} k1={k1} pf={r['gamma_pf']:.3e} bf={r['gamma_bf']:.3e} gap={r['gap_rob']:.4f} uniq={r['v_unico']} herm={r['v_herm']:.1e}")
