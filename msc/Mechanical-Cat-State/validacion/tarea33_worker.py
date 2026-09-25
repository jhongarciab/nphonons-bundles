# -*- coding: utf-8 -*-
"""Tarea 33: brecha en un Gamma2: Floquet completo (atol=1e-12,rtol=1e-10) vs paso (5) explicito vs efectivo bosonico.
Uso: tarea33_worker.py <Gamma2/kappa> <Nb|0> <outfile>   (alpha2=2, (dm,Dq)=(0.048,0.144))"""
import sys, numpy as np, modelo_comun as mc, modelo_ladder as ml
G2, Nb, out = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; Nb = None if Nb == 0 else Nb
a2 = 2.0; gzs = mc.gz_scale_from_Gamma2(G2)
U, p = mc.full_propagator(gzs, a2, 0.048, 0.144, 1e-12, 1e-10, Nb=Nb); N = p['Nb']
ev, evec = U.eigenstates()
m = mc.modos_ordenados(ev, evec, N, True, mc.T_r)
gap_fl, im_fl = m[4]['lam'].real, m[4]['lam'].imag
pf = max(m[1:4], key=lambda x: x['ov_P'])['lam'].real; bf = max(m[1:4], key=lambda x: x['ov_a'])['lam'].real
r5 = ml.analizar(ml.build(G2, a2, N, 's5'), N, a2)
L, _ = mc.effective_liouvillian(gzs, a2, 0.144, Nb=N); ee, eve = L.eigenstates(sparse=False)
me = mc.modos_ordenados(ee, eve, N, False, None)
np.savez(out, Gamma2=G2, N=N, gap_floquet=gap_fl, im_floquet=im_fl, pf_floquet=pf, bf_floquet=bf,
         gap_s5=r5['gap_rob'], gap_s5_simple=r5['gap_simple'], im_s5=r5['gap_im'], pf_s5=r5['gamma_pf'], bf_s5=r5['gamma_bf'],
         gap_eff=me[4]['lam'].real, im_eff=me[4]['lam'].imag)
print(f"OK G2={G2} N={N} floquet gap={gap_fl:.4e} Im={im_fl:+.2e} | s5 gap={r5['gap_rob']:.4e} | eff gap={me[4]['lam'].real:.4e}")
