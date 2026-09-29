# -*- coding: utf-8 -*-
"""Tarea 36: umbral termico. Uso: tarea36_worker.py <alpha2> <nq> <Nb|0> <outfile>  (Gamma2/kappa=0.13)
Propagador (atol,rtol)=(1e-15,1e-13). Estacionario: autovector con mu~1, simetrizado en paridad (quita el modo
impar de bit-flip, 1-mu~4e-9) y hermitizado; se reporta el error de hermiticidad ANTES de hermitizar."""
import sys, numpy as np, modelo_comun as mc
from qutip import vector_to_operator, coherent
a2, nq, Nb, out = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; Nb = None if Nb == 0 else Nb
U, p = mc.full_propagator(mc.gz_scale_from_Gamma2(0.13), a2, 0.048, 0.144, 1e-15, 1e-13, nq=nq, Nb=Nb); N = p['Nb']
ev, evec = U.eigenstates()
i0 = int(np.argmin(abs(ev - 1)))
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
np.savez(out, alpha2=a2, nq=nq, N=N, gamma_pf=pf, gamma_bf=bf, eta=pf / bf, p_code=pcode, a2=abs(np.trace(rho @ a @ a)),
         purity=np.real(np.trace(rho @ rho)), pexc=np.real(np.trace(rho @ np.kron(np.diag([1.0, 0.0]), np.eye(N)))),
         gap=m[4]['lam'].real, trace_err=trerr, herm=herm, min_eig=mineig, mu0=ev[i0],
         lam=np.array([x['lam'] for x in m[:8]]))
print(f"OK a2={a2} nq={nq:.3g} N={N} pf={pf:.3e} bf={bf:.3e} eta={pf/bf:.3g} pcode={pcode:.4f} |a2|={abs(np.trace(rho@a@a)):.4f} herm={herm:.1e} mineig={mineig:.1e}")
