# -*- coding: utf-8 -*-
"""Tarea 39: estado del modelo completo de Ma en 60 tiempos (Floquet + potencias por cuadrados).
Uso: tarea39_worker.py <wp> <N> <Gam_t_max> <outfile>"""
import sys, numpy as np, qutip as qt, ma_model as m
wp, N, gtmax, out = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
ops = m.observables_ops(N); v0 = m.initial(N).ravel(order='F')
U, T = m.floquet(wp, N)                       # atol 1e-12, rtol 1e-10
nmax = int(gtmax / (m.Gam * T)) + 1; kmax = int(np.ceil(np.log2(nmax))) + 1
E = m.Evolver(U, N, kmax)
ns = np.unique(np.round(np.geomspace(0.1, gtmax, 60) / (m.Gam * T)).astype(int))
rows = []
for n in ns:
    r = m.measure(E.rho(int(n), v0), N, ops); r['Gt'] = n * T * m.Gam; r['n'] = n; rows.append(r)
res = {k: np.array([r[k] for r in rows]) for k in rows[0]}
# chequeo: t exactamente 30/Gam (fase arbitraria dentro del periodo, como en los resultados previos)
H, c, args = m.hamiltonian(wp, N); t30 = 30 / m.Gam; n30 = int(t30 // T); s = t30 - n30 * T
Us = qt.propagator(H, s, c, args=args, options={'atol': 1e-12, 'rtol': 1e-10, 'nsteps': 200000}).full()
v = Us @ E.rho(n30, v0).ravel(order='F'); r30 = m.measure(v.reshape(2 * N, 2 * N, order='F'), N, ops)
np.savez(out, wp=wp, N=N, T=T, phase30=s / T, **res, **{f"chk30_{k}": v_ for k, v_ in r30.items()})
print(f"OK wp={wp} N={N} T={T:.4f} | stroboscopico Gt~30: " + str({k: round(float(res[k][np.argmin(abs(res['Gt']-30))]), 4) for k in ('Pc','par','Pe','a2')})
      + f" | t=30/Gam exacto (fase {s/T:.2f}): Pc={r30['Pc']:.4f} par={r30['par']:.4f} Pe={r30['Pe']:.4f} a2={r30['a2']:.3f}")
