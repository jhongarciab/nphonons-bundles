"""D6: trayectorias representativas (T=2e4) con tiempos de evento de fonon (canal 0), foton radiativo (1) y desfase (2) para n=1,2,3.
8 trayectorias por n (semillas fijas); 'rep' = la de numero de fonones mas cercano a la media."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runlib import *
from bundles_mc import run_mc
lam, gam, gph = BASE['lam'], BASE['gam'], BASE['gph']
CASES = {1: dict(Om=0.015, kap=0.003), 2: dict(Om=0.1, kap=0.003), 3: dict(Om=0.3, kap=0.0015)}
if __name__ == '__main__':
    name = 'D6_trajectories.npz'
    if cached(name): sys.exit()
    out = {}; meta = dict(T=2e4, lam=lam, gam=gam, gph=gph, cases=CASES, burn=3000.0, note='tiempos medidos desde t=burn (trayectoria total T+burn); 8 trayectorias por n')
    for n, c in CASES.items():
        m = build_pol(delta_res(n, lam, c['Om']), lam, c['Om'], c['kap'], gam, gph, n + 7)
        res = run_mc(m, T=2e4 + 3000.0, ntraj=8, seed=77 + n, dt=1.0, burn=3000.0)
        cnt = np.array([len(t) for t in res['ev'][0]]); rep = int(np.argmin(abs(cnt - cnt.mean())))
        out[f'n{n}_rep'] = rep
        out[f'n{n}_flux_exact'] = flux_exact(m); out[f'n{n}_flux_closed'] = flux_closed(lam, c['Om'], c['kap'], gam, gph, n)
        for ch, nm in enumerate(('phonon', 'photon', 'dephasing')):
            for k in range(8): out[f'n{n}_{nm}_t{k}'] = res['ev'][ch][k] - 3000.0
    save(name, meta, **out)
