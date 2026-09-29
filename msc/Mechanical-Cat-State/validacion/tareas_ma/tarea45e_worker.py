# -*- coding: utf-8 -*-
"""Tarea 45(e): un punto del barrido en w_p para un kappa2/kappa dado (variando g_x, g_z=0.2121 fijo, d=12 fijo como en la Tarea 40).
Uso: tarea45e_worker.py <kappa2/kappa> <s> <outfile>    w_p = w_p*(g_x) + s*kappa2 ; t = 120/kappa2 (kappa2 t = 120); N=22, alpha^2=4."""
import sys, numpy as np, ma2
r2, s, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
kappa, w, gz = 0.03, 6.0, 0.3 * np.cos(np.pi / 4); G0 = 2 * (0.3 * np.sin(np.pi / 4)) * gz / w   # G del caso base (kappa2/kappa=1)
gx = 0.3 * np.sin(np.pi / 4) * np.sqrt(r2)
p = ma2.params(w=w, gx=gx, gz=gz, kappa=kappa, alpha2=4.0); k2 = p['kappa2']
wps = 2 * (w - 4 * gx**2 / (3 * w)); p['wp'] = wps + s * k2; p['d'] = 12.0
N = 22; U, T = ma2.floquet(p, N, atol=1e-12, rtol=1e-10)
o = ma2.ops(N); v0 = ma2.initial(N).ravel(order='F'); n = int(round(120 / k2 / T)); E = ma2.Evolver(U, int(np.ceil(np.log2(n))) + 1)
r = ma2.measure(E.vec(n, v0).reshape(2 * N, 2 * N, order='F'), o)
np.savez(out, r2=r2, s=s, gx=gx, kappa2=k2, wp=p['wp'], wp_star=wps, T=T, n=n, t=n * T, **r)
print(f"OK k2/k={r2} s={s} wp={p['wp']:.6f} t={n*T:.0f} Pc={r['Pc']:.4f} Pe={r['Pe']:.4f} a2={r['a2']:.3f}")
