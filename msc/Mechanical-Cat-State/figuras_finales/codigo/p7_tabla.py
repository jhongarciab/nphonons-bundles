"""Tabla de la validación térmica por κ₂/κ: cocientes completo/efectivo de γ_pf y γ_bf (x = 6.86) y x*(η) del efectivo y del
completo ESTIMADO (η_completo(x) ≈ (η c/e a x = 6.86) · η_efectivo(x); supone el cociente independiente de x: en κ₂/κ = 0.25 vale 1.22 a x = 6.86 y 1.20 a 10.1)."""
import os
import numpy as np
from scipy.optimize import brentq
import comun as C
import calc_termico as CT
V = np.loadtxt(os.path.join(C.DATA, 'termico_validacion.csv'), delimiter=',', skiprows=1)
eta = lambda x, k: float(CT.punto(x, k, 14.0, 1, 2e-5, 22)['eta'])
print('κ₂/κ  χ|α|²/κ  γ_pf c/e  γ_bf c/e  η c/e   x*(100) ef/compl   x*(220) ef/compl')
out = []
for k in (0.05, 0.1, 0.25, 0.4):
    r = V[(abs(V[:, 0] - k) < 1e-9) & (V[:, 1] == 6.86)][0]; q = r[10]
    xs = []
    for E in (100, 220):
        xe = brentq(lambda x: eta(x, k) - E, 6.86, 16, xtol=1e-3); xc = brentq(lambda x: q * eta(x, k) - E, 6.5, 16, xtol=1e-3)
        xs += [xe, xc]
    out.append([k, 0.68 * k, r[4], r[7], q, *xs])
    print(f'{k:5.2f} {0.68*k:7.3f} {r[4]:8.3f} {r[7]:8.3f} {q:7.3f}   {xs[0]:.2f}/{xs[1]:.2f}       {xs[2]:.2f}/{xs[3]:.2f}')
np.savetxt(os.path.join(C.DATA, 'termico_tabla_k2.csv'), np.array(out), delimiter=',', comments='',
           header='k2/k, chi*al2/kappa, gpf c/e, gbf c/e, eta c/e, x*(100) eff, x*(100) full est, x*(220) eff, x*(220) full est')
