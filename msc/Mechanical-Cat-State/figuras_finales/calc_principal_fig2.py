"""Figura principal 2 (regla universal): estado estacionario de Floquet de un punto genérico (w, gx, gz, κ, Ω, wq, wp, N).
Modelo completo en el laboratorio; propagador de un período T_p = 2π/wp; estado estroboscópico t = nT_p (λ = 1).
Guarda en data/pfig2/<clave>.npz: P_c con el código fijo polarónico D(gz/w)|±α_nom> (α_nom² = Ω/G), α_eff²,
P_e y paridad promediados en un período (C5), validaciones y ρ.
Uso: python calc_principal_fig2.py w gx gz kap Om wq wp N [--rerun]
"""
import sys, os
import numpy as np
import qutip as qt
import comun as C


def clave(w, gx, gz, kap, Om, wq, wp, N):
    return os.path.join(C.DATA, 'pfig2', f'w{w:g}_gx{gx:.6g}_gz{gz:.6g}_k{kap:g}_Om{Om:.6g}_wq{wq:.8g}_wp{wp:.8g}_N{N}.npz')


def punto(w, gx, gz, kap, Om, wq, wp, N, rerun=False):
    f = clave(w, gx, gz, kap, Om, wq, wp, N)
    if os.path.exists(f) and not rerun:
        return dict(np.load(f))
    U, tprop = C.propagador(N, w, wq, gx, gz, Om, wp, kap)
    M, _ = C.estacionario(U, N)
    a, sm, sz, sx = C.ops(N)
    d = gz / w
    G = 2 * gx * gz / w
    A = a.full() - d * np.eye(2 * N)
    H = C.hamiltoniano(N, w, wq, gx, gz, Om, wp)
    ts = np.linspace(0, 2 * np.pi / wp, 41)
    par = qt.tensor((1j * np.pi * qt.num(N)).expm(), qt.qeye(2))
    r = qt.mesolve(H, qt.Qobj(M, dims=[[N, 2], [N, 2]]), ts, [np.sqrt(kap) * sm],
                   e_ops={'Pe': sm.dag() * sm, 'par': par}, options=C.OPTS)
    res = dict(w=w, gx=gx, gz=gz, kap=kap, Om=Om, wq=wq, wp=wp, N=N, G=G, d=d,
               Pc_fijo=C.Pc(M, N, np.sqrt(complex(Om / G)), d), al2eff=np.trace(A @ A @ M),
               Pe_prom=np.mean(np.real(r.e_data['Pe'][:-1])), par_prom=np.mean(np.real(r.e_data['par'][:-1])),
               val=np.array(C.validar(M)), tprop=tprop, rho=M)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez(f, **res)
    return res


if __name__ == '__main__':
    w, gx, gz, kap, Om, wq, wp = map(float, sys.argv[1:8]); N = int(sys.argv[8])
    r = punto(w, gx, gz, kap, Om, wq, wp, N, '--rerun' in sys.argv)
    print(f"w={w} gx={gx:.5g} gz={gz:.5g} wp={wp:.8g} N={N}: P_c={float(r['Pc_fijo']):.6f} P_e={float(r['Pe_prom']):.5f} "
          f"α_eff²={complex(r['al2eff']):.3f} val={r['val']} t={float(r['tprop']):.0f}s")
