# -*- coding: utf-8 -*-
"""Tarea 37: banos termicos consistentes. Uso: tarea37_worker.py <alpha2> <f_m GHz> <T mK> <Nb|0> <outfile>
Gamma2/kappa=0.13, (dm,Dq)=(0.048,0.144). nq=1/(exp(h 2f/kT)-1), nm=1/(exp(h f/kT)-1); D[s-] con (nq+1)kappa,
D[s+] con nq kappa; D[b] con (nm+1)gamma_m, D[b^dag] con nm gamma_m (gamma_m/kappa fijo)."""
import sys, numpy as np, modelo_comun as mc
from qutip import vector_to_operator, coherent
a2, f, T, Nb, out = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]; Nb = None if Nb == 0 else Nb
h_kT = 6.62607015e-34 * f * 1e9 / (1.380649e-23 * T * 1e-3)
nq, nm = 1 / np.expm1(2 * h_kT), 1 / np.expm1(h_kT)
U, p = mc.full_propagator(mc.gz_scale_from_Gamma2(0.13), a2, 0.048, 0.144, 1e-15, 1e-13, nq=nq, Nb=Nb, nm=nm, qcons=True); N = p['Nb']
ev, evec = U.eigenstates(); i0 = int(np.argmin(abs(ev - 1)))
X = vector_to_operator(evec[i0]); rho = X.full() / np.trace(X.full())
Pf = np.kron(np.eye(2), np.diag((-1.0) ** np.arange(N))); rho = (rho + Pf @ rho @ Pf) / 2
herm = np.linalg.norm(rho - rho.conj().T); rho = (rho + rho.conj().T) / 2
trerr = abs(np.trace(rho) - 1); mineig = np.linalg.eigvalsh(rho).min()
a = np.kron(np.eye(2), np.diag(np.sqrt(np.arange(1, N)), 1))
rc = np.einsum('iaib->ab', rho.reshape(2, N, 2, N))
kets = [coherent(N, s * np.sqrt(a2)).full()[:, 0] for s in (1, -1)]; Q, _ = np.linalg.qr(np.array(kets).T)
pcode = np.real(np.trace(Q @ Q.conj().T @ rc))
m = mc.modos_ordenados(ev, evec, N, True, mc.T_r)
pf = max(m[1:4], key=lambda x: x['ov_P'])['lam'].real; bf = max(m[1:4], key=lambda x: x['ov_a'])['lam'].real
np.savez(out, alpha2=a2, f=f, T=T, nq=nq, nm=nm, N=N, gamma_pf=pf, gamma_bf=bf, eta=pf / bf, p_code=pcode,
         a2=abs(np.trace(rho @ a @ a)), purity=np.real(np.trace(rho @ rho)), gap=m[4]['lam'].real,
         trace_err=trerr, herm=herm, min_eig=mineig, lam=np.array([x['lam'] for x in m[:8]]))
print(f"OK a2={a2} f={f} T={T} nq={nq:.3g} nm={nm:.3g} N={N} pf={pf:.3e} bf={bf:.3e} eta={pf/bf:.3g} pcode={pcode:.4f} |a2|={abs(np.trace(rho@a@a)):.4f} herm={herm:.1e}")
