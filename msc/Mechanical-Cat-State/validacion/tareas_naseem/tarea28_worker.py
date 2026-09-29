# -*- coding: utf-8 -*-
"""Tarea 28: 12 autovalores mas lentos (Re asc) del completo + overlaps P,n,a,sz,sm,sm a^2,sp a^2.
Uso: python tarea28_worker.py <Gamma2/kappa> <alpha2> <outfile>   (delta_m,Delta_q)=(0.048,0.144)"""
import sys, numpy as np
import modelo_comun as mc
G2, alpha2, outfile = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
NM = int(sys.argv[4]) if len(sys.argv) > 4 else 12
U, p = mc.full_propagator(mc.gz_scale_from_Gamma2(G2), alpha2, 0.048, 0.144)
ev, evec = U.eigenstates()
m = mc.modos_ordenados(ev, evec, p['Nb'], True, mc.T_r, n=NM, qubit_ops=True)
keys = ['ov_P', 'ov_n', 'ov_a', 'ov_sz', 'ov_sm', 'ov_sma2', 'ov_spa2']
np.savez(outfile, Gamma2=p['Gamma2'], alpha2=alpha2, lam=np.array([x['lam'] for x in m]),
         ov=np.array([[x[k] for k in keys] for x in m]), ov_keys=np.array(keys))
print(f"OK G2={p['Gamma2']:.4f}", " ".join(f"{x['lam']:.3g}" for x in m[:6]))
