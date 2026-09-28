"""Fig. 4 (cálculos): baño filtrado. El qubit decae a través de un modo filtro b (w_f = 2w, ancho κ_f, 4J²/κ_f = κ).

H = w a†a + (wq/2)σz + (a+a†)(gx σx + gz σz) + w_f b†b + J(σ+ b + σ- b†) [+ Ω(σ+ e^{-i wp t} + h.c.)]
Disipación: κ_f D[b] con filtro; κ D[σ-] sin filtro (baño plano, kf = 0).
Parámetros de Ma (Tarea 43): w = 6, g = 0.3, θ = π/4, κ = 0.03, wp = wq = 2(w - 4gx²/(3w)), Ω = |α|²G.

Modo 'estatico' (sin drive, gz = 0; ver V5): Liouvilliano estático denso.
   autovalor real no nulo más lento = Γ₁⁻ - Γ₁⁺;  Γ₁⁺ = (Γ₁⁻ - Γ₁⁺)(n̄ - n_virt), n_virt = ⟨n⟩ del fundamental de H.
Modo 'floquet' (con drive, gz completo): propagador de un período T_p; γ_pf = tasa del modo de paridad;
   estado estacionario (λ≈1), α_eff² = ⟨(a-d)²⟩ con d = gz/w, P_c con el código fijo; se guardan los 24 modos
   más lentos con su peso de borde (tasa de confinamiento, C9).
Uso: python calc_fig4.py estatico kf N Nf [--rerun]
     python calc_fig4.py floquet  kf N Nf al2 [--rerun]
"""
import sys, os, time
import numpy as np
import qutip as qt
import comun as C

W, KAP = 6.0, 0.03
GX, GZ = -0.3 * np.sin(np.pi / 4), 0.3 * np.cos(np.pi / 4)
WF = 2 * W
WP = 2 * (W - 4 * GX**2 / (3 * W))


def keff(d, kf):
    return KAP if kf == 0 else KAP * kf**2 / (4 * d**2 + kf**2)


def predic(kf):
    return GX**2 * keff(W, kf) / W**2, GX**2 * keff(3 * W, kf) / (9 * W**2)


def modelo(kf, N, Nf, gz):
    Nf = 1 if kf == 0 else Nf
    I = [qt.qeye(N), qt.qeye(2), qt.qeye(Nf)]
    op = lambda k, o: qt.tensor(*[o if i == k else I[i] for i in range(3)])
    a, sm, sz, sx = op(0, qt.destroy(N)), op(1, qt.sigmam()), op(1, qt.sigmaz()), op(1, qt.sigmax())
    H0 = W * a.dag() * a + 0.5 * WP * sz + (a + a.dag()) * (GX * sx + gz * sz)
    if kf == 0:
        return H0, a, sm, [np.sqrt(KAP) * sm], Nf
    b = op(2, qt.destroy(Nf))
    H0 = H0 + WF * b.dag() * b + np.sqrt(KAP * kf) / 2 * (sm.dag() * b + sm * b.dag())
    return H0, a, sm, [np.sqrt(kf) * b], Nf


def estatico(kf, N, Nf, rerun=False):
    f = os.path.join(C.DATA, 'fig4', f'est_kf{kf}_N{N}_Nf{Nf}.npz')
    if os.path.exists(f) and not rerun:
        return dict(np.load(f))
    H0, a, sm, c, Nf = modelo(kf, N, Nf, 0.0)
    ev = np.linalg.eigvals(qt.liouvillian(H0, c).full())
    real = sorted([-e.real for e in ev if abs(e.imag) < 1e-7 and -e.real > 1e-13])
    r = real[0]
    rho = qt.steadystate(H0, c, method='direct')
    nbar = qt.expect(a.dag() * a, rho); nvirt = qt.expect(a.dag() * a, H0.groundstate()[1])
    gp = r * (nbar - nvirt); gm = r + gp
    M = rho.full()
    res = dict(kf=kf, N=N, Nf=Nf, Gm=gm, Gp=gp, Gm_pred=predic(kf)[0], Gp_pred=predic(kf)[1], nbar=nbar, nvirt=nvirt,
               val=np.array(C.validar(M)))
    np.savez(f, **res)
    return res


def floquet(kf, N, Nf, al2, rerun=False):
    f = os.path.join(C.DATA, 'fig4', f'fl_kf{kf}_N{N}_Nf{Nf}_al{al2}.npz')
    if os.path.exists(f) and not rerun:
        return dict(np.load(f))
    H0, a, sm, c, Nf = modelo(kf, N, Nf, GZ)
    G = 2 * GX * GZ / W; Om = al2 * G
    H = [H0, [Om * sm.dag(), lambda t: np.exp(-1j * WP * t)], [Om * sm, lambda t: np.exp(1j * WP * t)]]
    Tp = 2 * np.pi / WP
    t0 = time.time()
    U = qt.propagator(H, Tp, c, options=C.OPTS).full()
    tprop = time.time() - t0
    D = 2 * N * Nf
    lam, R = np.linalg.eig(U)
    rate = -np.log(np.abs(lam)) / Tp
    idx = np.argsort(rate)[:24]
    P = qt.tensor((1j * np.pi * qt.num(N)).expm(), qt.qeye(2), qt.qeye(Nf)).full()
    borde = np.repeat(np.arange(N), 2 * Nf) > N - 6
    modos = []
    for k in idx:
        M = R[:, k].reshape(D, D, order='F'); nrm = np.linalg.norm(M)
        pb = 1 - np.linalg.norm(M[np.ix_(~borde, ~borde)])**2 / nrm**2
        modos.append([lam[k].real, lam[k].imag, rate[k], abs(np.trace(P @ M)) / nrm, pb])
    modos = np.array(modos)
    cand = [i for i, m in enumerate(modos) if m[2] > 1e-12 and m[0] > 0 and m[4] < 0.5]
    ip = max(cand, key=lambda i: modos[i, 3])
    M = R[:, idx[0]].reshape(D, D, order='F'); M = M / np.trace(M); M = (M + M.conj().T) / 2
    d = GZ / W
    A = a.full() - d * np.eye(D)
    al = np.sqrt(complex(al2 * np.sign(Om / G)))
    Dd = qt.displace(N, d)
    ca, cb = Dd * qt.coherent(N, np.sqrt(complex(Om / G))), Dd * qt.coherent(N, -np.sqrt(complex(Om / G)))
    Pc_op = qt.tensor((ca + cb).unit().proj() + (ca - cb).unit().proj(), qt.qeye(2), qt.qeye(Nf)).full()
    # C9: confinamiento DINÁMICO. Evolución período a período (t = nT_p) desde |0>|g>|0_f> y D(d)|1.3α>|g>|0_f>
    # hasta t = 6000 (κ₂t = 180); se guarda P_c(t) con el código fijo. El ajuste del retorno se hace en fig4.py.
    al_nom = np.sqrt(complex(Om / G))
    kets = [qt.tensor(qt.basis(N, 0), qt.basis(2, 1), qt.basis(Nf, 0)),
            qt.tensor(Dd * qt.coherent(N, 1.3 * al_nom), qt.basis(2, 1), qt.basis(Nf, 0))]
    pcv = Pc_op.reshape(-1, order='F').conj()
    nmax = int(6000 / Tp); paso = 10
    tray = []
    for kt in kets:
        v = qt.ket2dm(kt).full().reshape(-1, order='F'); serie = []
        for n in range(nmax + 1):
            if n % paso == 0:
                serie.append(np.real(pcv @ v))
            v = U @ v
        tray.append(serie)
    tdin = np.arange(0, nmax + 1, paso) * Tp
    res = dict(kf=kf, N=N, Nf=Nf, al2_nom=al2, gpf=modos[ip, 2], ip=ip, modos=modos, tdin=tdin, Pc_din=np.array(tray),
               al2eff=np.trace(A @ A @ M), Pc_fijo=np.real(np.trace(Pc_op @ M)),
               Pe=np.real(np.trace((sm.dag() * sm).full() @ M)), Gm_pred=predic(kf)[0], Gp_pred=predic(kf)[1],
               val=np.array(C.validar(M)), tprop=tprop, rho=M)
    np.savez(f, **res)
    return res


if __name__ == '__main__':
    modo, kf, N, Nf = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rr = '--rerun' in sys.argv
    if modo == 'estatico':
        r = estatico(kf, N, Nf, rr)
        print(f"est kf={kf} N={N} Nf={Nf}: Γ⁻={float(r['Gm']):.5e} ({float(r['Gm'])/float(r['Gm_pred']):.4f}) Γ⁺={float(r['Gp']):.5e} ({float(r['Gp'])/float(r['Gp_pred']):.4f}) val={r['val']}")
    else:
        r = floquet(kf, N, Nf, float(sys.argv[5]), rr)
        print(f"fl kf={kf} N={N} Nf={Nf}: γ_pf={float(r['gpf']):.5e} α_eff²={complex(r['al2eff']):.4f} P_c={float(r['Pc_fijo']):.5f} "
              f"P_e={float(r['Pe']):.5f} val={r['val']} t={float(r['tprop']):.0f}s")
        for m in r['modos'][:10]:
            print(f"    λ={m[0]:+.8f}{m[1]:+.1e}i r={m[2]:.4e} par={m[3]:.3f} borde={m[4]:.1e}")
