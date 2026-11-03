"""Bin et al. (n=2, acoplo longitudinal, maestra vestida) con los parametros del dispositivo de referencia (Arrangoiz-Arriola 1902.04681) en unidades de omega_b.
wb/2pi = 2.257 GHz. gam = 1/T1; gph = 2/Tphi (coherencia decae a gph/2). Comparacion con la fuente de pares (rutas/P): P(2)=0.72, pureza 0.80, flujo 1.1 MHz (reset activo)."""
import sys, numpy as np
sys.path.insert(0, '../../bundles/src')
import bundles_core as bc
wb = 2*np.pi*2.257e9
T1, Tphi = 1.14e-6, 0.30e-6
cases = {'publicado': (1/(T1*wb), 2/(Tphi*wb)), 'mejorado': (1/(10e-6*wb), 2/(3e-6*wb))}
print('gam, gph (en wb): publicado', ['%.2e' % x for x in cases['publicado']], ' mejorado', ['%.2e' % x for x in cases['mejorado']])
print('%-9s %-7s %-8s %-8s | %-9s %-9s %-14s'%('param', 'lam', 'Om', 'kap_m', 'pureza P2', 'flujo Hz', 'Om_eff/wb'))
n = 2
for nm, (gam, gph) in cases.items():
    for kap_MHz in (0.37, 10.0):
        kap = kap_MHz*1e6/(2257e6)
        for lam in (0.0027, 0.0067, 0.011, 0.03, 0.1):
            best = None
            for Om in (0.005, 0.02, 0.05, 0.1):
                try:
                    r = bc.exact_point(n, lam, Om, kap, gam, gph)
                except Exception as e:
                    continue
                fl = r['flux']*wb
                if best is None or fl > best[2]: best = (Om, r['P'], fl)
            Om, P, fl = best
            print('%-9s %-7.4f %-8.3f %-8.1f MHz | %-9.3f %-9.3g %-14.2e'%(nm, lam, Om, kap_MHz, P, fl, bc.omega_eff(n, lam, Om)), flush=True)
