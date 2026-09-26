# -*- coding: utf-8 -*-
"""Tarea 42: kappa1/kappa2 en el modelo completo de Ma re-sintonizado.
Uso: tarea42_worker.py <w> <gx> <gz/kappa> <N> <outfile>   (kappa=0.03, alpha^2=4)"""
import sys, numpy as np, scipy.linalg as sl, ma2
w, gx, gzk, N, out = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
kappa = 0.03; p = ma2.params(w=w, gx=gx, gz=gzk * kappa, kappa=kappa, alpha2=4.0)
U, T = ma2.floquet(p, N)
o = ma2.ops(N); v0 = ma2.initial(N).ravel(order='F')
# --- espectro: modos ordenados por Re(lambda), peso de borde, gap fisico
ev, V = sl.eig(U); lam = -np.log(ev.astype(complex)) / T; od = np.argsort(lam.real)[:60]
edge, ovP, ova = [], [], []
Pop = np.kron(np.eye(2), np.diag((-1.0) ** np.arange(N))); aop = np.kron(np.eye(2), np.diag(np.sqrt(np.arange(1, N)), 1))
for i in od:
    X = V[:, i].reshape(2 * N, 2 * N, order='F'); Xq = X.reshape(2, N, 2, N)
    edge.append(1 - np.linalg.norm(Xq[:, :N - 6, :, :N - 6])**2 / np.linalg.norm(X)**2)
    ovP.append(abs(np.trace(Pop.conj().T @ X)) / np.linalg.norm(X)); ova.append(abs(np.trace(aop.conj().T @ X)) / np.linalg.norm(X))
lam60, edge, ovP, ova = lam[od], np.array(edge), np.array(ovP), np.array(ova)
good = [k for k in range(60) if edge[k] <= 1e-3]
gap_old = lam60[4].real; gap_phys = lam60[good[4]].real if len(good) > 4 else np.nan
pf_idx = max(range(1, 4), key=lambda k: ovP[k]); rate_spec = lam60[pf_idx].real
# --- evolucion desde |0>|g>, muestras geometricas hasta t_max
kap1_pred = (10 / 9) * p['gx']**2 * kappa / w**2; rate_pred = 2 * 4.0 * kap1_pred
tmax = max(4.0 / rate_pred, 60.0 / p['kappa2']); nmax = int(tmax / T) + 1; kmax = int(np.ceil(np.log2(nmax))) + 1
E = ma2.Evolver(U, kmax); ns = np.unique(np.round(np.geomspace(1, nmax, 240)).astype(int))
rows = []
for n in ns:
    v = E.vec(int(n), v0); r = ma2.measure(v.reshape(2 * N, 2 * N, order='F'), o); r['t'] = n * T; rows.append(r)
res = {k: np.array([r[k] for r in rows]) for k in rows[0]}
ok = (res['Pc'] > 0.99) & (res['par'] > 0.05) & (res['par'] < 0.95)
if ok.sum() >= 4:
    rate_fit = -np.polyfit(res['t'][ok], np.log(res['par'][ok]), 1)[0]
else: rate_fit = np.nan
kap1 = rate_fit / (2 * 4.0)
np.savez(out, **{f"p_{k}": v for k, v in p.items()}, N=N, T=T, lam=lam60, edge=edge, ovP=ovP, ova=ova, gap_old=gap_old, gap_phys=gap_phys,
         rate_spec=rate_spec, rate_fit=rate_fit, kappa1=kap1, rate_pred=rate_pred, nwin=int(ok.sum()), Pc_max=res['Pc'].max(),
         t=res['t'], Pc=res['Pc'], par=res['par'], trace_err=res['trace_err'].max(), herm=res['herm'].max(), mineig=res['mineig'].min())
k2 = p['kappa2']
print(f"OK w={w} gx={gx} gz/k={gzk} N={N} k2/k={k2/kappa:.3g} Pcmax={res['Pc'].max():.4f} rate_fit={rate_fit:.3e} (pred {rate_pred:.3e}, espectral {rate_spec:.3e}) "
      f"k1/k2={kap1/k2:.3e} pred={(5/72)*(kappa/(gzk*kappa))**2:.3e} gap_phys={gap_phys:.3e} (viejo {gap_old:.3e})")
