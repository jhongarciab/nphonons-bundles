"""Fig. 3 (cálculo pesado): tasa de phase-flip por el método espectral en un punto (gx, w, gz/κ, |α|²_nom).

Modelo completo en el laboratorio, convención de la Tarea 42: wq = wp = 2(w - 4gx²/(3w)), κ = 0.03,
Ω = |α|²_nom G, G = 2gx gz/w (α² = +|α|²_nom, gato real). Propagador de un período T_p = 2π/wp;
tasa del modo de paridad (λ real ≈ +1, mayor |Tr(P R)|, peso de borde < 0.5); estado estacionario = autovector λ≈1.
Guarda en data/fig3/<clave>.npz: γ_pf espectral, ρ estacionaria, α_eff² = ⟨(a-d)²⟩ (d = gz/w), P_c con el
código fijo D(d)|±α_nom>, modos lentos y validaciones.
Uso: python calc_fig3.py gx w gz_sobre_kappa al2_nom N [--rerun]
"""
import sys, os
import numpy as np
import qutip as qt
import comun as C

KAP = 0.03


def clave(gx, w, gzk, al2, N):
    return os.path.join(C.DATA, 'fig3', f'gx{gx}_w{w}_gz{gzk}_al{al2}_N{N}.npz')


def punto(gx, w, gzk, al2, N, rerun=False):
    f = clave(gx, w, gzk, al2, N)
    if os.path.exists(f) and not rerun:
        return dict(np.load(f))
    gz = gzk * KAP
    G = 2 * gx * gz / w
    Om = al2 * G
    wp = 2 * (w - 4 * gx**2 / (3 * w))
    Tp = 2 * np.pi / wp
    U, tprop = C.propagador(N, w, wp, gx, gz, Om, wp, KAP)
    lam, R = np.linalg.eig(U)
    rate = -np.log(np.abs(lam)) / Tp
    idx = np.argsort(rate)[:12]
    D = 2 * N
    P = qt.tensor((1j * np.pi * qt.num(N)).expm(), qt.qeye(2)).full()
    borde = np.repeat(np.arange(N), 2) > N - 6
    filas = []
    for k in idx:
        M = R[:, k].reshape(D, D, order='F'); nrm = np.linalg.norm(M)
        pb = 1 - np.linalg.norm(M[np.ix_(~borde, ~borde)])**2 / nrm**2
        filas.append([k, lam[k].real, lam[k].imag, rate[k], abs(np.trace(P @ M)) / nrm, pb])
    cand = [x for x in filas if x[3] > 1e-12 and x[1] > 0 and x[5] < 0.5]
    kp = max(cand, key=lambda x: x[4])
    M, _ = C.estacionario(U, N)
    a = C.ops(N)[0].full()
    d = gz / w
    A = a - d * np.eye(D)
    res = dict(gx=gx, w=w, gzk=gzk, gz=gz, al2_nom=al2, N=N, G=G, Om=Om, wp=wp, kap2=4 * G**2 / KAP,
               gpf=kp[3], par_overlap=kp[4], par_borde=kp[5], modos=np.array(filas),
               al2eff=np.trace(A @ A @ M), Pc_fijo=C.Pc(M, N, np.sqrt(complex(al2)), d),
               val=np.array(C.validar(M)), tprop=tprop, rho=M)
    np.savez(f, **res)
    return res


if __name__ == '__main__':
    gx, w, gzk, al2 = map(float, sys.argv[1:5]); N = int(sys.argv[5])
    r = punto(gx, w, gzk, al2, N, '--rerun' in sys.argv)
    print(f"gx={gx} w={w} gz/κ={gzk} |α|²={al2} N={N}: γ_pf={float(r['gpf']):.6e} α_eff²={complex(r['al2eff']):.4f} "
          f"P_c={float(r['Pc_fijo']):.6f} κ₂/κ={float(r['kap2'])/KAP:.3f} borde={float(r['par_borde']):.1e} val={r['val']} t={float(r['tprop']):.0f}s")
