# -*- coding: utf-8 -*-
"""Tarea 40: un punto del barrido en wp. Uso: tarea40_worker.py <wp> <outfile>  (N=22, Gam*t ~ 60, t=nT)"""
import sys, numpy as np, ma_model as m
wp, out = float(sys.argv[1]), sys.argv[2]; N = 22
ops = m.observables_ops(N); v0 = m.initial(N).ravel(order='F')
U, T = m.floquet(wp, N); n = int(round(60 / (m.Gam * T))); E = m.Evolver(U, N, 13)
r = m.measure(E.rho(n, v0), N, ops)
np.savez(out, wp=wp, T=T, n=n, Gt=n * T * m.Gam, **r)
print(f"OK wp={wp:.5f} Gt={n*T*m.Gam:.3f} Pc={r['Pc']:.4f} Pe={r['Pe']:.4f} a2={r['a2']:.3f} par={r['par']:.3f}")
