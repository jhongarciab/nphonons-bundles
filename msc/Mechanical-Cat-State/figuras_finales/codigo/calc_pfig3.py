"""Figura principal 3 (cálculo pesado): tasa de confinamiento DINÁMICA (C9) y γ_pf en puntos del modelo completo.

Mismos puntos y convención que calc_fig3.py (Tarea 42: wq = wp = 2(w - 4gx²/(3w)), κ = 0.03, Ω = |α|²G).
Propagador de un período T_p en el laboratorio y descomposición espectral completa U = R Λ R⁻¹.
Dinámica exacta a tiempos estroboscópicos: P_c(n) = Σ_k λ_kⁿ c_k p_k, con c = R⁻¹ vec(ρ0) y p_k = Tr[Π_c R_k]
(código fijo polarónico). Estados iniciales: |0>|g> y D(d)|1.3α>|g>. Rejilla de tiempos logarítmica hasta
t_max = 60/κ₂ (o 2e3 como mínimo). El ajuste del retorno se hace en principal_fig3.py.
Guarda data/pfig3/<clave>.npz: t, Pc_din (2 × nt), P_c estacionario, γ_pf, α_eff², validaciones.
Uso: python calc_pfig3.py gx w gz_sobre_kappa al2 N [--rerun]
"""
import sys, os
import numpy as np
import qutip as qt
import comun as C

KAP = 0.03


def punto(gx, w, gzk, al2, N, rerun=False):
    f = os.path.join(C.DATA, 'pfig3', f'gx{gx}_w{w}_gz{gzk}_al{al2}_N{N}.npz')
    if os.path.exists(f) and not rerun:
        return dict(np.load(f))
    gz = gzk * KAP; G = 2 * gx * gz / w; Om = al2 * G
    wp = 2 * (w - 4 * gx**2 / (3 * w)); Tp = 2 * np.pi / wp
    kap2 = 4 * G**2 / KAP
    U, tprop = C.propagador(N, w, wp, gx, gz, Om, wp, KAP)
    lam, R = np.linalg.eig(U)
    Linv = np.linalg.inv(R)
    D = 2 * N; d = gz / w; al = np.sqrt(complex(al2))
    Pc_op = C.proyector_codigo(N, al, d)
    p = np.array([np.trace(Pc_op @ R[:, k].reshape(D, D, order='F')) for k in range(D * D)])
    # estado estacionario y γ_pf (modo de paridad) como en calc_fig3
    k0 = np.argmin(abs(lam - 1))
    M = R[:, k0].reshape(D, D, order='F'); M = M / np.trace(M); M = (M + M.conj().T) / 2
    rate = -np.log(np.abs(lam)) / Tp
    P = qt.tensor((1j * np.pi * qt.num(N)).expm(), qt.qeye(2)).full()
    borde = np.repeat(np.arange(N), 2) > N - 6
    cand = []
    for k in np.argsort(rate)[:12]:
        Mk = R[:, k].reshape(D, D, order='F'); nrm = np.linalg.norm(Mk)
        pb = 1 - np.linalg.norm(Mk[np.ix_(~borde, ~borde)])**2 / nrm**2
        if rate[k] > 1e-12 and lam[k].real > 0 and pb < 0.5:
            cand.append((abs(np.trace(P @ Mk)) / nrm, rate[k]))
    gpf = max(cand)[1]
    a = C.ops(N)[0].full(); A = a - d * np.eye(D)
    kets = [qt.tensor(qt.basis(N, 0), qt.basis(2, 1)),
            qt.tensor(qt.displace(N, d) * qt.coherent(N, 1.3 * al), qt.basis(2, 1))]
    tmax = max(60 / kap2, 2e3)
    ns = np.unique(np.round(np.geomspace(1, tmax / Tp, 1500)).astype(np.int64))
    serie = []
    for kt in kets:
        c = Linv @ qt.ket2dm(kt).full().reshape(-1, order='F')
        w_ = c * p
        # λⁿ con logaritmo complejo para estabilidad
        ll = np.log(lam.astype(complex))
        serie.append(np.real(np.exp(np.outer(ns, ll)) @ w_))
    res = dict(gx=gx, w=w, gzk=gzk, gz=gz, al2=al2, N=N, G=G, kap2=kap2, wp=wp, t=ns * Tp, Pc_din=np.array(serie),
               Pc_ss=C.Pc(M, N, al, d), gpf=gpf, al2eff=np.trace(A @ A @ M), val=np.array(C.validar(M)), tprop=tprop)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez(f, **res)
    return res


if __name__ == '__main__':
    gx, w, gzk, al2 = map(float, sys.argv[1:5]); N = int(sys.argv[5])
    r = punto(gx, w, gzk, al2, N, '--rerun' in sys.argv)
    print(f"gx={gx} w={w} gz/κ={gzk} N={N}: κ₂/κ={float(r['kap2'])/KAP:.4f} γ_pf={float(r['gpf']):.5e} P_c={float(r['Pc_ss']):.6f} "
          f"val={r['val']} t={float(r['tprop']):.0f}s")
