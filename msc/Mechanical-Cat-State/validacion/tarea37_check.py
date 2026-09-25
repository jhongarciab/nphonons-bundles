# -*- coding: utf-8 -*-
"""Tarea 37 (verificacion del bit-flip): |alpha>|g> propagado por U^n (potencias por cuadrados, sin autovalores)
y ajuste de la tasa de decaimiento de <sgn x>. Uso: tarea37_check.py <alpha2> <f GHz> <T mK> <outfile>"""
import sys, numpy as np, modelo_comun as mc
from qutip import coherent, basis, tensor
a2, f, T, out = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
h_kT = 6.62607015e-34 * f * 1e9 / (1.380649e-23 * T * 1e-3); nq, nm = 1 / np.expm1(2 * h_kT), 1 / np.expm1(h_kT)
U, p = mc.full_propagator(mc.gz_scale_from_Gamma2(0.13), a2, 0.048, 0.144, 1e-15, 1e-13, nq=nq, nm=nm, qcons=True); N = p['Nb']
ev, evec = U.eigenstates(); m = mc.modos_ordenados(ev, evec, N, True, mc.T_r)
gbf = max(m[1:4], key=lambda x: x['ov_a'])['lam'].real
xw, xv = np.linalg.eigh(np.diag(np.sqrt(np.arange(1, N)), 1) + np.diag(np.sqrt(np.arange(1, N)), -1))
Sg = np.kron(np.eye(2), xv @ np.diag(np.sign(xw)) @ xv.T)
ket = tensor(basis(2, 1), coherent(N, np.sqrt(a2))).full()[:, 0]; d = len(ket); v = np.outer(ket, ket.conj()).ravel(order='F')
Pw = [U.full()]
for k in range(1, 36): Pw.append(Pw[-1] @ Pw[-1])       # U^(2^k)
ts, ss = [0.0], [np.real(np.trace(Sg @ v.reshape(d, d, order='F')))]; t = 0.0
for k in range(36):
    for rep in range(3):
        v = Pw[k] @ v; t += 2**k * mc.T_r; ts.append(t); ss.append(np.real(np.trace(Sg @ v.reshape(d, d, order='F'))))
ts, ss = np.array(ts), np.array(ss); s0 = ss[0]
mk = (ss < 0.9 * s0) & (ss > 0.15 * s0)
rate = -np.polyfit(ts[mk], np.log(ss[mk]), 1)[0] if mk.sum() >= 3 else np.nan
np.savez(out, alpha2=a2, f=f, T=T, nq=nq, nm=nm, gamma_bf_spectral=gbf, rate_fit=rate, n_fit=int(mk.sum()), t=ts, s=ss)
print(f"OK a2={a2} f={f} T={T} nq={nq:.3g} nm={nm:.3g} gamma_bf spectral={gbf:.4e} rate fit <sgn x>={rate:.4e} (n={int(mk.sum())}) s0={s0:.4f}")
