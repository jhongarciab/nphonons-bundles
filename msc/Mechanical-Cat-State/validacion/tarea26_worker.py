# -*- coding: utf-8 -*-
"""Tarea 26: precision de gamma_bf. Uso: python tarea26_worker.py <gz_scale> <alpha2> <atol> <rtol> <M> <outfile>"""
import sys, numpy as np
from qutip import Qobj
import modelo_comun as mc

gz_scale, alpha2, atol, rtol, M = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
outfile = sys.argv[6]
U, p = mc.full_propagator(gz_scale, alpha2, 0.048, 0.144, atol, rtol)
Nb = p['Nb']

ev, evec = U.eigenstates()
mod1 = mc.modos_ordenados(ev, evec, Nb, True, mc.T_r)
bf1 = mc.bf_de_modos(mod1)

UM = Qobj(np.linalg.matrix_power(U.full(), M), dims=U.dims)
evM, evecM = UM.eigenstates()
modM = mc.modos_ordenados(evM, evecM, Nb, True, M * mc.T_r)
bfM = mc.bf_de_modos(modM)

def campos(m):
    return (m['lam'].real, m['lam'].imag, abs(m['mu']), m['ov_a']) if m else (np.nan,) * 4
g1, i1, mu1, oa1 = campos(bf1); gM, iM, muM, oaM = campos(bfM)
np.savez(outfile, gz_scale=gz_scale, alpha2=alpha2, atol=atol, rtol=rtol, M=M, Gamma2=p['Gamma2'],
         gamma_bf_1=g1, im_1=i1, absmu_1=mu1, gamma_bf_M=gM, im_M=iM, absmu_M=muM,
         one_minus_absmu_1=1 - mu1, one_minus_absmu_M=1 - muM,
         lam_top_1=np.array([m['lam'] for m in mod1]))
print(f"OK G2={p['Gamma2']:.3f} a2={alpha2} tol=({atol:g},{rtol:g}) gbf(1T)={g1:.4e} gbf(M={M})={gM:.4e}")
