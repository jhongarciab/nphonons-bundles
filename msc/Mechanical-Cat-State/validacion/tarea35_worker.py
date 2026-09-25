# -*- coding: utf-8 -*-
"""Tarea 35: cuadruplete de Floquet. Uso: tarea35_worker.py <Gamma2/kappa> <N> <do_dinamica 0|1> <outfile>
alpha2=2, (delta_m,Delta_q)=(0.048,0.144), Floquet (atol,rtol)=(1e-12,1e-10).
(a) 40 modos mas lentos con overlaps; (b) Im/omega_r; (c) dinamica estroboscopica de retorno al codigo desde
tres estados fuera del codigo + pesos modales (V^-1 = autovectores izquierdos)."""
import sys, numpy as np, scipy.linalg as sl
from qutip import basis, coherent, tensor, fock
import modelo_comun as mc, modelo_ladder as ml

G2, N, do_c, out = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
a2 = 2.0; al = np.sqrt(a2)
U, p = mc.full_propagator(mc.gz_scale_from_Gamma2(G2), a2, 0.048, 0.144, 1e-12, 1e-10, Nb=N)
ev, evec = U.eigenstates()
m = mc.modos_ordenados(ev, evec, N, True, mc.T_r, n=40, qubit_ops=True)
keys = ['ov_P', 'ov_n', 'ov_a', 'ov_sz', 'ov_sm', 'ov_sma2', 'ov_spa2']
lam = np.array([x['lam'] for x in m]); ov = np.array([[x[k] for k in keys] for x in m])
# cuadruplete: 4 modos mas lentos (k>=4) con |Im|>2
cand = [k for k in range(4, 40) if abs(lam[k].imag) > 2.0]
quad = cand[:4]
res = dict(Gamma2=G2, N=N, lam=lam, ov=ov, quad=np.array(quad), gap=lam[4].real, gap_im=lam[4].imag,
           omega_r=mc.om_m, T_r=mc.T_r)
r5 = ml.analizar(ml.build(G2, a2, N, 's5'), N, a2); res['gap_s5'] = r5['gap_rob']
if do_c:
    Umat = U.full(); d = int(round(np.sqrt(Umat.shape[0]))); dq = 2
    # vectores propios (todos) y sus normas
    V = np.column_stack([e.full().ravel() for e in evec])
    xnorm = np.array([np.linalg.norm(V[:, k]) for k in range(V.shape[1])])   # = ||X_k||_F (columnas de vec)
    lam_all = -np.log(ev.astype(complex)) / mc.T_r
    ordr = np.argsort(lam_all.real)                      # mismo orden que modos_ordenados
    code_idx = set(ordr[:4]); quad_idx = set(ordr[k] for k in quad)
    kets = [coherent(N, s * al).full()[:, 0] for s in (1, -1)]
    Q, _ = np.linalg.qr(np.array(kets).T)
    Pi = np.kron(np.diag([0.0, 1.0]), Q @ Q.conj().T)      # orden (qubit x oscilador): qubit en basis(2,1)=base
    inicial = {'coh': tensor(basis(2, 1), coherent(N, 1.3 * al)), 'fock': tensor(basis(2, 1), fock(N, 4)),
               'exc': tensor(basis(2, 0), coherent(N, al))}
    U20 = np.linalg.matrix_power(Umat, 20); nsteps = 1500; t = np.arange(nsteps + 1) * 20 * mc.T_r
    for name, ket in inicial.items():
        k0 = ket.full()[:, 0]; rho0 = np.outer(k0, k0.conj()); v0 = rho0.ravel(order='F')
        c = np.linalg.solve(V, v0); recon = np.linalg.norm(V @ c - v0) / np.linalg.norm(v0)
        w = np.abs(c) * xnorm; w[list(code_idx)] = 0; tot = w.sum()
        frac_quad = w[list(quad_idx)].sum() / tot
        # peso por modo (orden ascendente por Re) de los 12 primeros no-codigo + cuadruplete
        w_ord = np.array([w[ordr[k]] for k in range(40)]) / tot
        v = v0.copy(); leak = np.empty(nsteps + 1)
        for s in range(nsteps + 1):
            rho = v.reshape(d, d, order='F'); leak[s] = 1 - np.real(np.trace(Pi @ rho)); v = U20 @ v
        res[f'{name}_leak'] = leak; res[f'{name}_frac_quad'] = frac_quad; res[f'{name}_w'] = w_ord; res[f'{name}_recon'] = recon
    res['t'] = t
    # suelo estacionario
    i0 = int(np.argmin(abs(ev - 1))); X0 = evec[i0].full().ravel().reshape(d, d, order='F'); X0 = X0 / np.trace(X0)
    res['leak_inf'] = 1 - np.real(np.trace(Pi @ X0))
np.savez(out, **res)
print(f"OK G2={G2} N={N} gap={lam[4].real:.5f} Im={lam[4].imag:+.2f} quad={[(round(lam[k].real,4), round(lam[k].imag,2)) for k in quad]} s5={r5['gap_rob']:.5f}")
