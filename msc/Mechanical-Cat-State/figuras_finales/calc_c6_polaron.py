"""C6: verificación de la base polarónica. Estado estacionario de Floquet en dos puntos y barrido de un
desplazamiento complejo d del código {D(d)|±α>}; se compara el óptimo con la predicción d = +gz/w.
Caché: data/c6_<caso>_N<N>.npz (ρ estacionaria y barrido). Uso: python calc_c6_polaron.py caso N [--rerun]
Casos: 'ma' (Ma re-sintonizado, wp = 11.98, α = 2i) y 'v4' (gx=0.05, w=6, gz/κ=12, α = 2, convención T42)."""
import sys, os
import numpy as np
from scipy.optimize import minimize
import comun as C

CASOS = {
    'ma': dict(w=6.0, gx=-0.3 * np.sin(np.pi / 4), gz=0.3 * np.cos(np.pi / 4), kap=0.03, Om=0.06, wq=12.0, wp=11.98),
}
gx, gz, w = 0.05, 12 * 0.03, 6.0
wpv = 2 * (w - 4 * gx**2 / (3 * w))
CASOS['v4'] = dict(w=w, gx=gx, gz=gz, kap=0.03, Om=4 * 2 * gx * gz / w, wq=wpv, wp=wpv)


def main():
    caso, N = sys.argv[1], int(sys.argv[2])
    p = CASOS[caso]
    f = os.path.join(C.DATA, f'c6_{caso}_N{N}.npz')
    if os.path.exists(f) and '--rerun' not in sys.argv:
        z = np.load(f); M = z['rho']; tprop = float(z['tprop'])
    else:
        U, tprop = C.propagador(N, p['w'], p['wq'], p['gx'], p['gz'], p['Om'], p['wp'], p['kap'])
        M, _ = C.estacionario(U, N)
    G = 2 * p['gx'] * p['gz'] / p['w']
    alpha = np.sqrt(complex(p['Om'] / G))
    dpred = p['gz'] / p['w']
    # barrido 1D sobre el eje real y optimización 2D
    ds = np.linspace(-2 * dpred, 3 * dpred, 41)
    pcs = np.array([C.Pc(M, N, alpha, d) for d in ds])
    opt = minimize(lambda x: -C.Pc(M, N, alpha, x[0] + 1j * x[1]), [dpred, 0.0], method='Nelder-Mead',
                   options=dict(xatol=1e-6, fatol=1e-12))
    dopt = opt.x[0] + 1j * opt.x[1]
    # también se optimiza |α| (el gato real es algo menor que el nominal)
    opt2 = minimize(lambda x: -C.Pc(M, N, x[2] * alpha / abs(alpha), x[0] + 1j * x[1]), [dpred, 0, abs(alpha)],
                    method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-12))
    val = C.validar(M)
    print(f"caso={caso} N={N} α={alpha:.3f} d_pred=+gz/w={dpred:.5f}  t_prop={tprop:.0f}s")
    print(f"  P_c(d=0)={C.Pc(M, N, alpha, 0):.6f}  P_c(d_pred)={C.Pc(M, N, alpha, dpred):.6f}  P_c(-d_pred)={C.Pc(M, N, alpha, -dpred):.6f}")
    print(f"  óptimo d={dopt.real:+.5f}{dopt.imag:+.5f}i (d/d_pred={dopt.real/dpred:.4f})  P_c={-opt.fun:.6f}")
    print(f"  óptimo con |α| libre: d={opt2.x[0]:+.5f}{opt2.x[1]:+.5f}i |α|={opt2.x[2]:.4f} (|α|²={opt2.x[2]**2:.4f}) P_c={-opt2.fun:.6f}")
    print(f"  validación: |Tr-1|={val[0]:.1e} ‖ρ-ρ†‖={val[1]:.1e} mín eig={val[2]:.1e}")
    np.savez(f, rho=M, tprop=tprop, ds=ds, pcs=pcs, dopt=dopt, dpred=dpred, opt2=opt2.x, alpha=alpha, **p)


if __name__ == '__main__':
    main()
