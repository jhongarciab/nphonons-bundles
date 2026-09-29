"""V4: figura de mérito κ₁/κ₂ por el método espectral en el MARCO DE LABORATORIO.

Modelo completo (sin RWA): H(t) = w a†a + (wq/2)σz + (a+a†)(gx σx + gz σz) + Ω(σ+ e^{-i wp t} + h.c.), κ D[σ-].
Convención (la del trabajo, Tarea 42): wp = wq = 2(w - 4gx²/(3w)), κ = 0.03, |α|² = 4, Ω = |α|² G, G = 2 gx gz / w.
  -> α² = Ω/G = +4 (gato real en ±2).

Método: propagador del superoperador sobre UN período del drive T_p = 2π/wp; sus autovalores λ_k dan
las tasas r_k = -ln|λ_k| / T_p. En el laboratorio el oscilador rota a wp/2, así que en un período
|α> -> |-α>: el modo de "pozo" (|α><α| - |-α><-α|) tiene λ ≈ -1 y el modo de PARIDAD tiene λ real ≈ +1.
El modo de paridad se identifica como el autovector derecho R_k con mayor |Tr(P R_k)| (P = e^{iπ a†a})
entre los modos lentos, y se exige peso de borde pequeño (n > N-6).
κ₁ = r_par / (2|α|²); κ₂ = 4G²/κ; predicción κ₁/κ₂ = (5/72)(κ/gz)².
Opcional: evolución temporal período a período para comparar (ajuste A e^{-kt} + c con P_c > 0.99).
Uso: python v4_espectral.py gx w gz_sobre_kappa N [n_periodos_evolucion] [salida.npz]
"""
import sys, time
import numpy as np
import qutip as qt
from scipy.optimize import curve_fit

import os
KAP, AL2 = 0.03, float(os.environ.get("AL2", "4.0"))  # |α|² nominal (V4b lo varía)
OPTS = dict(atol=1e-12, rtol=1e-10, nsteps=10**6)


def main():
    gx, w, gzk, N = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    nev = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    out = sys.argv[6] if len(sys.argv) > 6 else f"res_v4/gx{gx}_w{w}_gz{gzk}_N{N}.npz"
    gz = gzk * KAP
    G = 2 * gx * gz / w
    Om = AL2 * G
    wp = 2 * (w - 4 * gx**2 / (3 * w)); wq = wp
    Tp = 2 * np.pi / wp
    kap2 = 4 * G**2 / KAP
    pred = 5 / 72 * (KAP / gz)**2

    a = qt.tensor(qt.destroy(N), qt.qeye(2)); sm = qt.tensor(qt.qeye(N), qt.sigmam())
    sz = qt.tensor(qt.qeye(N), qt.sigmaz()); sx = qt.tensor(qt.qeye(N), qt.sigmax())
    H0 = w * a.dag() * a + 0.5 * wq * sz + (a + a.dag()) * (gx * sx + gz * sz)
    H = [H0, [Om * sm.dag(), lambda t: np.exp(-1j * wp * t)], [Om * sm, lambda t: np.exp(1j * wp * t)]]
    c = [np.sqrt(KAP) * sm]
    t0 = time.time()
    U = qt.propagator(H, Tp, c, options=OPTS).full()
    tprop = time.time() - t0

    D = 2 * N
    lam, R = np.linalg.eig(U)
    rate = -np.log(np.abs(lam)) / Tp
    idx = np.argsort(rate)[:12]                                   # modos más lentos
    P = qt.tensor((1j * np.pi * qt.num(N)).expm(), qt.qeye(2)).full()
    nidx = np.repeat(np.arange(N), 2)
    borde = nidx > N - 6
    filas = []
    for k in idx:
        M = R[:, k].reshape(D, D, order='F')
        nrm = np.linalg.norm(M)
        tp = abs(np.trace(P @ M)) / nrm
        pb = np.linalg.norm(M[np.ix_(borde, borde)])**2 / nrm**2 + np.linalg.norm(M[np.ix_(borde, ~borde)])**2 / nrm**2 \
            + np.linalg.norm(M[np.ix_(~borde, borde)])**2 / nrm**2
        filas.append((k, lam[k], rate[k], tp, pb))
    # modo de paridad: mayor |Tr(P R)| entre los lentos no estacionarios con λ real positivo y borde < 0.5
    cand = [f for f in filas if f[2] > 1e-12 and f[1].real > 0 and f[4] < 0.5]
    kpar = max(cand, key=lambda f: f[3])
    r_par = kpar[2]
    kap1 = r_par / (2 * AL2)
    # estado estacionario (λ más cercano a 1): P_c y ⟨a²⟩
    k0 = idx[0]
    rho_ss = qt.Qobj(R[:, k0].reshape(D, D, order='F'), dims=[[N, 2], [N, 2]])
    rho_ss = rho_ss / rho_ss.tr(); rho_ss = (rho_ss + rho_ss.dag()) / 2
    alpha = np.sqrt(complex(Om / G))
    ca, cb = qt.coherent(N, alpha), qt.coherent(N, -alpha)
    Pc_op = qt.tensor((ca + cb).unit().proj() + (ca - cb).unit().proj(), qt.qeye(2))
    Pc_ss = qt.expect(Pc_op, rho_ss); a2_ss = qt.expect(a * a, rho_ss)
    Mss = rho_ss.full()
    val = (abs(np.trace(Mss) - 1), np.linalg.norm(Mss - Mss.conj().T), np.linalg.eigvalsh(Mss).min())

    print(f"gx={gx} w={w} gz/κ={gzk} N={N}  wp=wq={wp:.6f}  G={G:.4e}  κ₂/κ={kap2/KAP:.3f}  t_prop={tprop:.0f}s")
    print("  modos lentos: λ, tasa, |Tr(P R)|/‖R‖, peso de borde")
    for f in filas[:8]:
        print(f"    {f[1].real:+.9f}{f[1].imag:+.2e}i  r={f[2]:.4e}  par={f[3]:.3f}  borde={f[4]:.1e}" + ("   <- paridad" if f[0] == kpar[0] else ""))
    print(f"  r_par={r_par:.5e}  κ₁={kap1:.5e}  κ₁/κ₂={kap1/kap2:.5e}  pred={pred:.5e}  razón={kap1/kap2/pred:.4f}")
    print(f"  estacionario: P_c={Pc_ss.real:.5f} ⟨a²⟩={a2_ss.real:+.4f}{a2_ss.imag:+.4f}i  |Tr-1|={val[0]:.1e} ‖ρ-ρ†‖={val[1]:.1e} mín eig={val[2]:.1e}")

    res = dict(gx=gx, w=w, gzk=gzk, N=N, r_par=r_par, kap1=kap1, kap2=kap2, pred=pred, Pc_ss=Pc_ss.real,
               a2_ss=a2_ss, lam=lam[idx], tprop=tprop)
    if nev:
        # evolución temporal período a período desde |0>|g>, estroboscópica t = nT_p
        v = qt.ket2dm(qt.tensor(qt.basis(N, 0), qt.basis(2, 1))).full().reshape(-1, order='F')
        Pv = P.reshape(-1, order='F').conj(); Pcv = Pc_op.full().reshape(-1, order='F').conj()
        paso = max(1, nev // 2000); ts, par, pc, peor = [], [], [], [0, 0, 0]
        for n in range(nev + 1):
            if n % paso == 0:
                ts.append(n * Tp); par.append((Pv @ v).real); pc.append((Pcv @ v).real)
                if n % (paso * 50) == 0:
                    M = v.reshape(D, D, order='F')
                    peor = [max(peor[0], abs(np.trace(M) - 1)), max(peor[1], np.linalg.norm(M - M.conj().T)),
                            min(peor[2], np.linalg.eigvalsh((M + M.conj().T) / 2).min())]
            v = U @ v
        ts, par, pc = map(np.array, (ts, par, pc))
        m = pc > 0.99
        f = lambda t, A, k, c0: A * np.exp(-k * t) + c0
        q, _ = curve_fit(f, ts[m], par[m], p0=(par[m][0], r_par, 0), maxfev=20000)
        print(f"  evolución: P_c>0.99 en t∈[{ts[m][0]:.0f},{ts[m][-1]:.0f}] (κ₂t∈[{kap2*ts[m][0]:.0f},{kap2*ts[m][-1]:.0f}]); "
              f"k(piso)={q[1]:.5e} piso={q[2]:+.1e}  k/r_par={q[1]/r_par:.4f}  razón temporal={q[1]/(2*AL2)/kap2/pred:.4f}")
        print(f"  validación evolución: |Tr-1|≤{peor[0]:.1e} ‖ρ-ρ†‖≤{peor[1]:.1e} mín eig≥{peor[2]:.1e}")
        res.update(ts=ts, par=par, pc=pc, k_temp=q[1], piso=q[2], peor=peor)
    np.savez(out, **res)


if __name__ == '__main__':
    main()
