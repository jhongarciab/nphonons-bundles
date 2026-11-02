"""Validacion de la resonancia analitica: maximo del flujo exacto vs delta_res (2o orden) vs Bin-III (sin Stark de orden superior). data/resonance.txt"""
from runlib import *
from scipy.optimize import minimize_scalar
gam, gph, kap = 2e-4, 4e-4, 0.003
out = ['n lam Om | dL_num | err(2o orden) [gamma_c] | err(Bin-III + lam^2) [gamma_c]']
for n in (1, 2, 3):
    for lam in (0.08, 0.14, 0.2):
        for Om in ((0.01, 0.02, 0.05) if n == 1 else (0.05, 0.1, 0.2, 0.3, 0.4)):
            f = lambda dL: -flux_exact(build_pol(dL, lam, Om, kap, gam, gph, n + 6))
            d0 = delta_res(n, lam, Om); gc = (n * kap + gam + gph) / 2
            r = minimize_scalar(f, bounds=(d0 - 0.02, d0 + 0.02), method='bounded', options=dict(xatol=1e-8))
            bin3 = lam**2 - np.sqrt(n * n - 4 * Om**2) if 4 * Om**2 < n * n else np.nan
            out.append(f'{n} {lam} {Om} | {r.x:+.5f} | {(r.x - d0) / gc:+.3f} ({r.x - d0:+.1e}) | {(r.x - bin3) / gc:+.3f} ({r.x - bin3:+.1e})')
open(os.path.join(DATA, 'resonance.txt'), 'w').write('\n'.join(out)); print('\n'.join(out))
