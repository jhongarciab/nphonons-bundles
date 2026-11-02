"""C: validacion Monte Carlo (maestra vestida, marco de polaron) de la pureza por episodios y de la cerrada.
Uso: python3 run_C.py [--force]. Salida: ../data/C_mc.npz. 400 trayectorias/punto, semillas fijas, 2 procesos."""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[_v]="1"
import sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from multiprocessing import Pool
from bundles_core import *
from bundles_mc import *

LAM, GPH, GAM0 = 0.14, 4e-4, 2e-4
POINTS = ([(2, 0.1, 0.003, GAM0), (2, 0.05, 0.003, GAM0), (2, 0.2, 0.003, GAM0), (2, 0.1, 0.006, GAM0), (2, 0.1, 0.01, GAM0),
           (2, 0.1, 0.0015, GAM0), (2, 0.1, 0.003, 2e-5), (2, 0.1, 0.01, 2e-5)] +
          [(3, 0.2, 0.0015, GAM0), (3, 0.3, 0.002, GAM0), (3, 0.4, 0.003, GAM0), (3, 0.2, 0.003, GAM0), (3, 0.3, 0.0015, GAM0),
           (3, 0.2, 0.0008, GAM0), (3, 0.3, 0.003, 2e-5), (3, 0.2, 0.0015, 2e-5)])
NTRAJ = 400

def one(args):
    idx, (n, Om, kap, gam) = args
    N = n + 7
    dL = delta_res(n, LAM, Om)
    ex = exact_point(n, LAM, Om, kap, gam, GPH, N=N)
    T = float(np.clip(9000 * (n + LAM**2) / (NTRAJ * ex['flux']), 2e5, 2.5e6))
    m = build_pol(dL, LAM, Om, kap, gam, GPH, N)
    t0 = time.time()
    res = run_mc(m, T=T, ntraj=NTRAJ, seed=1000 + idx, dt=2.0, burn=3000.0)
    taus = np.array([2, 5, 10]) / kap
    st = mc_stats(res, n, taus)
    G0 = Gamma_j(LAM, Om, kap, gam, GPH, n, 0)
    Pep = ex['P']
    pred = [P_operational_pred(Pep, n, kap, gam, G0, t) for t in taus]
    return dict(idx=idx, n=n, Om=Om, kap=kap, gam=gam, T=T, P_ep=Pep, P_closed=P_closed(LAM, Om, kap, gam, GPH, n),
                flux_exact=ex['flux'], flux_closed=flux_closed(LAM, Om, kap, gam, GPH, n), flux_mc=st['flux'], flux_mc_err=st['flux_err'],
                taus=taus, P_mc=st['P'], P_mc_err=st['Perr'], nb=st['nb'], hist=st['hist'], P_pred=np.array(pred),
                rate_rad=st['rate_rad'], cpu=time.time() - t0)

if __name__ == '__main__':
    out = os.path.join(os.path.dirname(__file__), '..', 'data', 'C_mc.npz')
    if os.path.exists(out) and '--force' not in sys.argv:
        print('cache existe'); sys.exit()
    with Pool(2) as p:
        rows = p.map(one, list(enumerate(POINTS)), chunksize=1)
    keys = rows[0].keys()
    np.savez(out, **{k: np.array([r[k] for r in rows]) for k in keys},
             meta=json.dumps(dict(lam=LAM, gph=GPH, ntraj=NTRAJ, dt=2.0, burn=3000.0, seeds='1000+idx', frame='polaron dressed',
                                  clustering='gap<tau_c, tau_c=2,5,10/kappa')))
    for r in rows:
        print(r['n'], r['Om'], r['kap'], r['gam'], 'Pep=%.4f Pcl=%.4f' % (r['P_ep'], r['P_closed']), 'MC', np.round(r['P_mc'], 4), 'pred', np.round(r['P_pred'], 4),
              'nb', r['nb'], 'J %.3e/%.3e' % (r['flux_mc'], r['flux_exact']), 'cpu %.0f' % r['cpu'], flush=True)
