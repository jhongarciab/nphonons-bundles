"""C2: MC de validacion adicional. (a) distribucion de fonones por ciclo (entre fotones radiativos): MC vs exacta, maestra vestida y local
(local simulada en el marco de LABORATORIO, N=12: prueba independiente de la equivalencia lab == polaron-local con el signo corregido);
(b) agrupacion temporal con la maestra local vs episodios exactos (clasificacion pre-salto)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from multiprocessing import Pool
from bundles_core import *
from bundles_mc import *
LAM, GAM, GPH, OM, N = 0.14, 2e-4, 4e-4, 0.1, 2

def cyc_hist(res, K=8):
    h = np.zeros(K + 1)
    for tb, tg in zip(res['ev'][0], res['ev'][1]):
        if len(tg) < 3: continue
        c = np.searchsorted(tb, tg)            # nº de fonones con t < tg_i
        k = np.diff(c)[1:-1] if len(c) > 3 else np.diff(c)
        k = np.diff(c)
        h += np.bincount(np.minimum(k, K), minlength=K + 1)
    return h / h.sum(), h.sum()

def job(a):
    kind, kap, idx = a
    dL = delta_res(N, LAM, OM)
    if kind == 'lab':
        m = build_lab(dL, LAM, OM, kap, GAM, GPH, 12); mex = build_pol(dL, LAM, OM, kap, GAM, GPH, 9, 'local')
    else:
        m = build_pol(dL, LAM, OM, kap, GAM, GPH, 9, kind); mex = m
    res = run_mc(m, T=3e5, ntraj=400, seed=5000 + idx, dt=2.0, burn=3000.0)
    h, nc = cyc_hist(res)
    exc = cycle_dist(mex, 8)[0]
    out = dict(kind=kind, kap=kap, cyc_mc=h, cyc_exact=exc, ncyc=nc, flux_mc=mc_stats(res, N, [5 / kap])['flux'], flux_exact=flux_exact(mex))
    if kind != 'lab':
        st = mc_stats(res, N, np.array([2, 5, 10]) / kap)
        out.update(P_mc=st['P'], P_mc_err=st['Perr'], P_ep_exact=episode_dist(mex, 12)[0][N], nb=st['nb'])
    return out

if __name__ == '__main__':
    name = os.path.join(os.path.dirname(__file__), '..', 'data', 'C2_cycles_local.json')
    if os.path.exists(name) and '--force' not in sys.argv: sys.exit()
    jobs = [('dressed', 0.003, 0), ('lab', 0.003, 1), ('local', 0.003, 2), ('local', 0.01, 3), ('dressed', 0.01, 4)]
    with Pool(2) as p: rows = p.map(job, jobs)
    json.dump(dict(meta=dict(n=N, lam=LAM, Om=OM, gam=GAM, gph=GPH, jobs=jobs, ntraj=400, T=3e5), rows=rows), open(name, 'w'), default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
    for r in rows:
        print(r['kind'], r['kap'], 'ciclos MC', np.round(r['cyc_mc'][:6], 4), 'exacta', np.round(r['cyc_exact'][:6], 4), 'ncyc', r['ncyc'], 'J mc/ex %.3e/%.3e' % (r['flux_mc'], r['flux_exact']))
        if 'P_mc' in r: print('    clustering', np.round(r['P_mc'], 4), 'P_ep exacta(local-clasif.)', round(r['P_ep_exact'], 4), r['nb'])
