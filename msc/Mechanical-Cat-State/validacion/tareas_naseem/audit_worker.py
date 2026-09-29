# -*- coding: utf-8 -*-
"""Tarea 38: espectro Floquet completo con peso de borde de Fock por modo.
Uso: audit_worker.py <Gamma2/kappa> <N> <delta_m> <Delta_q> <alpha2> <outfile>   (Floquet atol=1e-12,rtol=1e-10)
Guarda los 60 modos mas lentos (Re asc): lam, overlaps (P,n,a), w_edge = fraccion de ||X||_F^2 con algun indice de
oscilador n > N-6."""
import sys, numpy as np, modelo_comun as mc
G2, N, dm, dq, a2, out = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
U, p = mc.full_propagator(mc.gz_scale_from_Gamma2(G2), a2, dm, dq, 1e-12, 1e-10, Nb=N)
ev, evec = U.eigenstates(); K = 60
lam_all = -np.log(ev.astype(complex)) / mc.T_r; o = np.argsort(lam_all.real)[:K]
m = mc.modos_ordenados(ev, evec, N, True, mc.T_r, n=K)
edge = []; prof = []
for idx, i in enumerate(o):
    X = evec[i].full().ravel().reshape(2 * N, 2 * N, order='F').reshape(2, N, 2, N)   # (q, n, q', n')
    core = np.linalg.norm(X[:, :N - 6, :, :N - 6]) ** 2; edge.append(1 - core / np.linalg.norm(X) ** 2)
    if idx < 14: prof.append((np.abs(X)**2).sum(axis=(0, 2, 3)) / np.linalg.norm(X)**2 + (np.abs(X)**2).sum(axis=(0, 1, 2)) / np.linalg.norm(X)**2)   # perfil en n (fila + columna)
np.savez(out, Gamma2=G2, N=N, dm=dm, dq=dq, alpha2=a2, lam=np.array([x['lam'] for x in m]),
         ov=np.array([[x['ov_P'], x['ov_n'], x['ov_a']] for x in m]), w_edge=np.array(edge), prof=np.array(prof))
print(f"OK G2={G2} N={N} dm={dm} dq={dq} gap5={m[4]['lam']:.4f} w_edge[4..8]={np.round(edge[4:9],4)}")
