# -*- coding: utf-8 -*-
"""Uso: ladder_worker.py <alpha2> <Gamma2/kappa> <variante> <N> <outfile>"""
import sys, numpy as np, modelo_ladder as ml
a2, G2, var, N, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
r = ml.analizar(ml.build(G2, a2, N, var), N, a2)
np.savez(out, alpha2=a2, Gamma2=G2, var=var, N=N, **r)
print(f"OK a2={a2} G2={G2:.4f} {var} N={N} rob={r['gap_rob']:.4e} simple={r['gap_simple']:.4e} Im={r['gap_im']:+.2e} "
      f"pf={r['gamma_pf']:.3e} bf={r['gamma_bf']:.3e} uniq={r['v_unico']} herm={r['v_herm']:.1e} mineig={r['v_min_eig']:.1e}")
