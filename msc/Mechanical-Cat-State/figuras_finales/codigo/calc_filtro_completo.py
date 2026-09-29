"""Verificación del panel (b) de la figura central: modelo completo con filtro en un punto genérico.
H(t) = w a†a + (wq/2)σz + (a+a†)(gx σx + gz σz) + w_f b†b + J(σ+ b + σ- b†) + Ω(σ+ e^{-i wp t} + h.c.), κ_f D[b],
w_f = 2w, 4J²/κ_f = κ, wq = wp = 2(w − 4gx²/3w), Ω = |α|²G. Propagador de un período (laboratorio), descomposición
espectral completa: γ_pf (modo de paridad) y P_c(t) exacto a t = nT_p desde |0>|g>|0_f> (código fijo polarónico).
Guarda data/filtro_completo/<clave>.npz. Uso: python calc_filtro_completo.py gx w gz kap kf al2 N Nf [--rerun]
"""
import sys, os, time
import numpy as np
import qutip as qt
import comun as C


def punto(gx, w, gz, kap, kf, al2, N, Nf, rerun=False, variante='full', espectro=False, gam=0.0, x=None):
    suf = ('' if variante == 'full' else f'_{variante}') + (f'_gam{gam:g}' if gam else '') + (f'_x{x:g}' if x else '')
    f = os.path.join(C.DATA, 'filtro_completo', f'gx{gx:.6g}_w{w:g}_gz{gz:.6g}_kf{kf:g}_al{al2:g}_N{N}_Nf{Nf}{suf}.npz')
    if os.path.exists(f) and not rerun:
        return dict(np.load(f))
    I = [qt.qeye(N), qt.qeye(2), qt.qeye(Nf)]
    op = lambda k, o: qt.tensor(*[o if i == k else I[i] for i in range(3)])
    a, sm, sz, sx, b = op(0, qt.destroy(N)), op(1, qt.sigmam()), op(1, qt.sigmaz()), op(1, qt.sigmax()), op(2, qt.destroy(Nf))
    wp = 2 * (w - 4 * gx**2 / (3 * w)); G = 2 * gx * gz / w; Om = al2 * G; J = np.sqrt(kap * kf) / 2
    if variante == 'nogz_pair':
        # P10(1): sin g_z σ_z (a+a†); intercambio de pares explícito −G(σ₊a² + h.c.) (signo de R1)
        H0 = w * a.dag() * a + 0.5 * wp * sz + gx * (a + a.dag()) * sx - G * (sm.dag() * a * a + sm * a.dag() * a.dag())
    else:
        H0 = w * a.dag() * a + 0.5 * wp * sz + (a + a.dag()) * (gx * sx + gz * sz)
    H0 = H0 + 2 * w * b.dag() * b + J * (sm.dag() * b + sm * b.dag())
    H = [H0, [Om * sm.dag(), lambda t: np.exp(-1j * wp * t)], [Om * sm, lambda t: np.exp(1j * wp * t)]]
    Tp = 2 * np.pi / wp
    t0 = time.time()
    # baños térmicos (convención de la Tarea 37): filtro a f_q con n_q; oscilador a f_q/2 con n_m. x = h f_q/(k_B T)
    nq = 1 / np.expm1(x) if x else 0.0; nm = 1 / np.expm1(x / 2) if x else 0.0
    cops = [np.sqrt(kf * (nq + 1)) * b] + ([np.sqrt(kf * nq) * b.dag()] if nq else [])
    if gam:
        cops += [np.sqrt(gam * (nm + 1)) * a] + ([np.sqrt(gam * nm) * a.dag()] if nm else [])
    U = qt.propagator(H, Tp, cops, options=C.OPTS).full()
    tprop = time.time() - t0
    lam, R = np.linalg.eig(U); Linv = np.linalg.inv(R)
    D = 2 * N * Nf; d = gz / w if variante == 'full' else 0.0; al = np.sqrt(complex(Om / G))
    Dd = qt.displace(N, d)
    ca, cb = Dd * qt.coherent(N, al), Dd * qt.coherent(N, -al)
    Pc_op = qt.tensor((ca + cb).unit().proj() + (ca - cb).unit().proj(), qt.qeye(2), qt.qeye(Nf)).full()
    p = np.einsum('ij,jik->k', Pc_op, R.reshape(D, D, -1, order='F'))
    k0 = np.argmin(abs(lam - 1))
    M = R[:, k0].reshape(D, D, order='F'); M = M / np.trace(M); M = (M + M.conj().T) / 2
    rate = -np.log(np.abs(lam)) / Tp
    P = qt.tensor((1j * np.pi * qt.num(N)).expm(), qt.qeye(2), qt.qeye(Nf)).full()
    borde = np.repeat(np.arange(N), 2 * Nf) > N - 6
    cand = []
    for k in np.argsort(rate)[:12]:
        Mk = R[:, k].reshape(D, D, order='F'); nrm = np.linalg.norm(Mk)
        pb = 1 - np.linalg.norm(Mk[np.ix_(~borde, ~borde)])**2 / nrm**2
        if rate[k] > 1e-12 and lam[k].real > 0 and pb < 0.5:
            cand.append((abs(np.trace(P @ Mk)) / nrm, rate[k]))
    gpf = max(cand)[1]
    # γ_bf: modo de pozo (en el laboratorio λ ≈ −1), mayor traslape con a entre los modos lentos
    A_op = a.full(); candb = []
    for k in np.argsort(rate)[:12]:
        Mk = R[:, k].reshape(D, D, order='F'); nrm = np.linalg.norm(Mk)
        if rate[k] > 1e-14 and lam[k].real < 0:
            candb.append((abs(np.trace(A_op @ Mk)) / nrm, rate[k]))
    gbf = max(candb)[1] if candb else np.nan
    kap2 = 4 * G**2 / kap
    ns = np.unique(np.round(np.geomspace(1, max(60 / kap2, 2e3) / Tp, 1500)).astype(np.int64))
    c = Linv @ qt.ket2dm(qt.tensor(qt.basis(N, 0), qt.basis(2, 1), qt.basis(Nf, 0))).full().reshape(-1, order='F')
    serie = np.real(np.exp(np.outer(ns, np.log(lam.astype(complex)))) @ (c * p))
    A = a.full() - d * np.eye(D)
    # P_e promediado en un período (C5)
    rr = qt.mesolve(H, qt.Qobj(M, dims=[[N, 2, Nf], [N, 2, Nf]]), np.linspace(0, Tp, 41), cops,
                    e_ops={'Pe': sm.dag() * sm}, options=C.OPTS)
    Pe_prom = np.mean(np.real(rr.e_data['Pe'][:-1]))
    extra = {}
    if espectro:
        # P10(2): S(ν) = FT <σ₊(τ)σ₋(0)> desde el estacionario en fase 0 del drive (regresión cuántica)
        taus = np.arange(0, 400, 0.05)
        X0 = qt.Qobj(sm.full() @ M, dims=[[N, 2, Nf], [N, 2, Nf]])
        rs = qt.mesolve(H, X0, taus, cops, e_ops={'c': sm.dag()}, options=dict(atol=1e-10, rtol=1e-8, nsteps=10**7))
        corr = np.array(rs.e_data['c'])
        nu = np.fft.fftfreq(len(taus), taus[1] - taus[0]) * 2 * np.pi
        S = 2 * np.real(np.fft.fft(corr * np.hanning(2 * len(taus))[len(taus):])) * (taus[1] - taus[0])
        extra = dict(taus=taus, corr=corr, nu=nu, S=S)
    res = dict(variante=variante, gam=gam, x=x if x else 0.0, nq=nq, gbf=gbf, herm_cruda=np.linalg.norm(R[:, k0].reshape(D, D, order='F') / np.trace(R[:, k0].reshape(D, D, order='F')) - (R[:, k0].reshape(D, D, order='F') / np.trace(R[:, k0].reshape(D, D, order='F'))).conj().T), Pe_prom=Pe_prom, **extra,gx=gx, w=w, gz=gz, kap=kap, kf=kf, al2=al2, N=N, Nf=Nf, kap2=kap2, gpf=gpf, t=ns * Tp, Pc_din=serie,
               Pc_ss=np.real(np.trace(Pc_op @ M)), al2eff=np.trace(A @ A @ M), val=np.array(C.validar(M)), tprop=tprop)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez(f, **res)
    return res


if __name__ == '__main__':
    gx, w, gz, kap, kf, al2 = map(float, sys.argv[1:7]); N, Nf = int(sys.argv[7]), int(sys.argv[8])
    var = next((x.split('=')[1] for x in sys.argv if x.startswith('--variante=')), 'full')
    gam = float(next((x.split('=')[1] for x in sys.argv if x.startswith('--gam=')), 0.0))
    xt = float(next((x.split('=')[1] for x in sys.argv if x.startswith('--x=')), 0.0)) or None
    r = punto(gx, w, gz, kap, kf, al2, N, Nf, '--rerun' in sys.argv, var, '--espectro' in sys.argv, gam, xt)
    print(f"gx={gx} w={w} gz={gz} kf={kf}: κ₂/κ={float(r['kap2'])/kap:.4f} γ_pf={float(r['gpf']):.5e} P_c={float(r['Pc_ss']):.5f} P_e={float(r['Pe_prom']):.5f} var={var} γ_bf={float(r['gbf']):.4e} η={float(r['gpf'])/float(r['gbf']):.4g} herm_cruda={float(r['herm_cruda']):.1e} "
          f"val={r['val']} t={float(r['tprop']):.0f}s")
