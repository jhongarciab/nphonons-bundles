"""Nueva Fig. 2(d), opcional: gato par transitorio. Ma, wp = 11.98, wq = 12, Ω = 0.06, N = 22, marco de laboratorio.
Propagador de un período y evolución estroboscópica (t = nT_p) desde |0>|g> hasta Γt = 30 (Γ = κ/2 = 0.015).
Fidelidad con el gato par polarónico |C+> ∝ D(d)(|α> + |-α>), d = gz/w, α = 2i (qubit trazado).
Guarda data/principal_fig2d.npz: t, F(t), P_c(t), paridad(t) y ρ en el máximo de F. Uso: python calc_principal_fig2d.py [--rerun]"""
import os, sys
import numpy as np
import qutip as qt
import comun as C

f = os.path.join(C.DATA, 'principal_fig2d.npz')
if not os.path.exists(f) or '--rerun' in sys.argv:
    N, W, GX, GZ, KAP, OM, WP, WQ = 22, 6.0, -0.3 * np.sin(np.pi / 4), 0.3 * np.cos(np.pi / 4), 0.03, 0.06, 11.98, 12.0
    U, tprop = C.propagador(N, W, WQ, GX, GZ, OM, WP, KAP)
    Tp = 2 * np.pi / WP; d = GZ / W; al = 2j
    Dd = qt.displace(N, d)
    cp = (Dd * (qt.coherent(N, al) + qt.coherent(N, -al))).unit()
    Fop = qt.tensor(cp.proj(), qt.qeye(2)).full().reshape(-1, order='F').conj()
    Pc = C.proyector_codigo(N, al, d).reshape(-1, order='F').conj()
    Par = qt.tensor((1j * np.pi * qt.num(N)).expm(), qt.qeye(2)).full().reshape(-1, order='F').conj()
    v = qt.ket2dm(qt.tensor(qt.basis(N, 0), qt.basis(2, 1))).full().reshape(-1, order='F')
    nmax = int(30 / 0.015 / Tp)
    ts, F, P, Q, estados = [], [], [], [], {}
    for n in range(nmax + 1):
        ts.append(n * Tp); F.append(np.real(Fop @ v)); P.append(np.real(Pc @ v)); Q.append(np.real(Par @ v))
        if n % 5 == 0:
            estados[n] = v.copy()
        v = U @ v
    F = np.array(F); k = int(np.argmax(F)); k5 = 5 * (k // 5)
    M = estados[k5].reshape(2 * N, 2 * N, order='F')
    np.savez(f, t=np.array(ts), F=F, Pc=np.array(P), par=np.array(Q), kmax=k5, rho=M, val=np.array(C.validar(M)), tprop=tprop)
z = np.load(f)
print(f"F máx = {z['F'][int(z['kmax'])]:.4f} en Γt = {0.015 * z['t'][int(z['kmax'])]:.2f}; P_c = {z['Pc'][int(z['kmax'])]:.5f}; paridad = {z['par'][int(z['kmax'])]:.4f}; val = {z['val']}")
