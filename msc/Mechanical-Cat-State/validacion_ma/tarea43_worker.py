# -*- coding: utf-8 -*-
"""Tarea 43: baño filtrado. Uso: tarea43_worker.py <kappa_f (0 = baño plano)> <N> <outfile>
Ma re-sintonizado con g_x=g_z=0.3/sqrt2 (Fig. 2), kappa=0.03, alpha^2=2 (dimension reducida: N=16, filtro Nf=Nfarg).
Filtro f a 2w acoplado J(s+ f + s- f^dag), decae kf; 4J^2/kf = kappa. kappa1 se mide de la tasa de paridad
(modo de Floquet con mayor overlap con la paridad) y por evolucion temporal de |0>|g>; kappa2 = brecha fisica."""
import sys, numpy as np, scipy.linalg as sl, ma2
kf, N, out = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
Nfarg = int(sys.argv[4]) if len(sys.argv) > 4 else 2; noevo = len(sys.argv) > 5 and sys.argv[5] == "noevo"
kappa, a2 = 0.03, 2.0; g = 0.3
p = ma2.params(w=6.0, gx=g * np.sin(np.pi / 4), gz=g * np.cos(np.pi / 4), kappa=kappa, alpha2=a2)
filt = None if kf == 0 else dict(kf=kf, J=np.sqrt(kappa * kf / 4), Nf=Nfarg)
Nf = 1 if filt is None else Nfarg; d = 2 * N * Nf
U, T = ma2.floquet(p, N, filt)
o = ma2.ops(N, Nf, a2); v0 = ma2.initial(N, Nf).ravel(order='F')
ev, V = sl.eig(U); lam = -np.log(ev.astype(complex)) / T; od = np.argsort(lam.real)[:60]
Pop = np.kron(np.kron(np.eye(2), np.diag((-1.0) ** np.arange(N))), np.eye(Nf))
edge, ovP = [], []
for i in od:
    X = V[:, i].reshape(d, d, order='F'); Xq = X.reshape(2, N, Nf, 2, N, Nf)
    edge.append(1 - np.linalg.norm(Xq[:, :N - 6, :, :, :N - 6, :])**2 / np.linalg.norm(X)**2); ovP.append(abs(np.trace(Pop.conj().T @ X)) / np.linalg.norm(X))
lam60, edge, ovP = lam[od], np.array(edge), np.array(ovP); good = [k for k in range(60) if edge[k] <= 1e-3]
pf = max(range(1, 4), key=lambda k: ovP[k]); rate_spec = lam60[pf].real
gap_phys = lam60[good[4]].real if len(good) > 4 else np.nan
# evolucion temporal
w, gx = p['w'], p['gx']
kflat = (10 / 9) * gx**2 * kappa / w**2
if filt is None: kpred = kflat
else:
    ke = lambda dl: 4 * filt['J']**2 * kf / (4 * dl**2 + kf**2)
    kpred = gx**2 * (ke(w) / w**2 + ke(3 * w) / (9 * w**2))
rate_pred = 2 * a2 * kpred; nmax = int(min(4.0 / rate_pred, 3e8) / T) + 1; kmax = int(np.ceil(np.log2(nmax))) + 1
if noevo: nmax = 4; kmax = 3
E = ma2.Evolver(U, kmax); ns = np.unique(np.round(np.geomspace(1, nmax, 200)).astype(int)); rows = []
for n in ns:
    r = ma2.measure(E.vec(int(n), v0).reshape(d, d, order='F'), o); r['t'] = n * T; rows.append(r)
res = {k: np.array([r[k] for r in rows]) for k in rows[0]}
ok = (res['Pc'] > 0.99) & (res['par'] > 0.05) & (res['par'] < 0.95)
rate_fit = -np.polyfit(res['t'][ok], np.log(res['par'][ok]), 1)[0] if ok.sum() >= 4 else np.nan
np.savez(out, kf=kf, N=N, Nf=Nf, T=T, kappa1_flat_pred=kflat, kappa1_pred=kpred, rate_pred=rate_pred, rate_spec=rate_spec, rate_fit=rate_fit,
         gap_phys=gap_phys, gap_old=lam60[4].real, kappa2=p['kappa2'], lam=lam60, edge=edge, ovP=ovP, Pc_max=res['Pc'].max(), nwin=int(ok.sum()),
         t=res['t'], Pc=res['Pc'], par=res['par'], trace_err=res['trace_err'].max(), herm=res['herm'].max(), mineig=res['mineig'].min())
print(f"OK kf={kf} N={N} rate_spec={rate_spec:.3e} rate_fit={rate_fit:.3e} pred={rate_pred:.3e} gap_phys={gap_phys:.3e} kappa2={p['kappa2']:.3e} Pcmax={res['Pc'].max():.4f}")
