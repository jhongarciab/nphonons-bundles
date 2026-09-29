# -*- coding: utf-8 -*-
"""Tarea 34: qubit termico kappa*nq*D[s+] en el modelo completo (Floquet). Gamma2/kappa=0.13, alpha2=2.
Uso: tarea34_worker.py <nq> <Nb|0> <outfile>.  Propagador con (atol,rtol)=(1e-14,1e-12) para que el estado
estacionario cumpla las tolerancias de validacion."""
import sys, numpy as np, modelo_comun as mc
from qutip import vector_to_operator, tensor, qeye, destroy, sigmap, sigmam, coherent, basis
nq, Nb, out = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; Nb = None if Nb == 0 else Nb
G2, a2 = 0.13, 2.0
U, p = mc.full_propagator(mc.gz_scale_from_Gamma2(G2), a2, 0.048, 0.144, 1e-14, 1e-12, nq=nq, Nb=Nb); N = p['Nb']
ev, evec = U.eigenstates()
i0 = int(np.argmin(abs(ev - 1)))
X = vector_to_operator(evec[i0]); rho = X.full() / np.trace(X.full())
Pfull = np.kron(np.eye(2), np.diag((-1.0) ** np.arange(N)))     # orden (qubit x oscilador) como modelo_comun
rho = (rho + Pfull @ rho @ Pfull) / 2                             # quita contaminacion del modo impar (bit-flip)
herm = np.linalg.norm(rho - rho.conj().T); rho = (rho + rho.conj().T) / 2
trerr = abs(np.trace(rho) - 1); mineig = np.linalg.eigvalsh(rho).min()
a = np.kron(np.eye(2), np.diag(np.sqrt(np.arange(1, N)), 1)); sz_exc = np.kron(np.diag([1.0, 0.0]), np.eye(N))
purity = np.real(np.trace(rho @ rho)); a2exp = abs(np.trace(rho @ a @ a)); pexc = np.real(np.trace(rho @ sz_exc))
rc = rho.reshape(2, N, 2, N); rcav = np.einsum('iaib->ab', rc); pur_cav = np.real(np.trace(rcav @ rcav))
kets = [coherent(N, s * np.sqrt(a2)).full()[:, 0] for s in (1, -1)]
Q, _ = np.linalg.qr(np.array(kets).T); pcode = np.real(np.trace(Q @ Q.conj().T @ rcav))
m = mc.modos_ordenados(ev, evec, N, True, mc.T_r)
pf = max(m[1:4], key=lambda x: x['ov_P'])['lam'].real; bf = max(m[1:4], key=lambda x: x['ov_a'])['lam'].real
np.savez(out, nq=nq, N=N, purity=purity, purity_cav=pur_cav, a2=a2exp, pexc=pexc, p_code=pcode, gamma_pf=pf, gamma_bf=bf,
         gap=m[4]['lam'].real, gap_im=m[4]['lam'].imag, trace_err=trerr, herm=herm, min_eig=mineig,
         mu0=ev[i0], lam=np.array([x['lam'] for x in m[:8]]))
print(f"OK nq={nq} N={N} purity={purity:.4f} pur_cav={pur_cav:.4f} |<a2>|={a2exp:.4f} pexc={pexc:.4f} pcode={pcode:.4f} pf={pf:.3e} bf={bf:.3e} gap={m[4]['lam'].real:.4e} | tr={trerr:.1e} herm={herm:.1e} mineig={mineig:.1e}")
