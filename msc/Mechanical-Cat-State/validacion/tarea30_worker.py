# -*- coding: utf-8 -*-
"""Tarea 30: gamma_bf con rtol=1e-13. Uso: python tarea30_worker.py <Gamma2/kappa> <alpha2> <Delta_q> <gam_m> <outfile>"""
import sys, numpy as np
import modelo_comun as mc
G2, alpha2, Dq, gm, outfile = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
U, p = mc.full_propagator(mc.gz_scale_from_Gamma2(G2), alpha2, 0.048, Dq, 1e-15, 1e-13, gam_m=gm)
ev, evec = U.eigenstates()
m = mc.modos_ordenados(ev, evec, p['Nb'], True, mc.T_r)
bf = mc.bf_de_modos(m)
pf = max(m[1:4], key=lambda x: x['ov_P'])
np.savez(outfile, Gamma2=p['Gamma2'], alpha2=alpha2, Delta_q=Dq, gam_m=gm, gamma_bf=bf['lam'].real,
         im_bf=bf['lam'].imag, gamma_pf=pf['lam'].real, one_minus_absmu=1 - abs(bf['mu']))
print(f"OK G2={p['Gamma2']:.3f} a2={alpha2} Dq={Dq} gm={gm:g} gbf={bf['lam'].real:.5e} gpf={pf['lam'].real:.4e}")
