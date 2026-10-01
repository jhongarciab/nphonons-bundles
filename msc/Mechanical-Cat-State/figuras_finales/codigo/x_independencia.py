"""Fase 6, punto 1: ¿es independiente de x el cociente completo/efectivo? Modelo completo con filtro (N = 22, unidades de Ma, κ = 0.03 se divide)
a x ≈ 9 frente a x = 6.86, con el piso de T = 0 restado de cada modelo (completo: corrida de T = 0 de la fase 3; efectivo: x = 60).
Criterio fijado antes del resultado: cociente de γ_bf con piso restado a x ≈ 9 dentro del 3% de su valor a x = 6.86 ⇒ la estimación de x*
a partir de x = 6.86 se considera sostenida; si no, se reporta y no se fuerza la conclusión."""
import os, glob
import numpy as np
from scipy.optimize import brentq
import comun as C
import calc_termico as CT
K = 0.03
GX = {0.05: 0.0239579, 0.1: 0.0338816, 0.25: 0.0535714, 0.4: 0.0677631}
PUNTOS = [(0.05, 9.0), (0.1, 9.0), (0.4, 9.5), (0.25, 10.1)]          # (κ₂/κ, x); el último viene de la fase 2
def cargar(k, x):
    f = os.path.join(C.DATA, 'filtro_completo', f'gx{GX[k]:.6g}_w6_gz0.42_kf0.3_al4_N22_Nf2_gam6e-07' + (f'_x{x:g}' if x else '') + '.npz')
    return np.load(f) if os.path.exists(f) else None
def eta_e(x, k): return float(CT.punto(x, k, 14.0, 1, 2e-5, 22)['eta'])
filas = []
for k, x in PUNTOS:
    z, z0, zr = cargar(k, x), cargar(k, None), cargar(k, 6.86)
    if z is None or z0 is None or zr is None:
        print(f'κ₂/κ={k} x={x}: falta corrida'); continue
    e0 = CT.punto(60.0, k, 14.0, 1, 2e-5, 22)
    res = {}
    for nombre, xx, zz in (('6.86', 6.86, zr), (f'{x:g}', x, z)):
        e = CT.punto(xx, k, 14.0, 1, 2e-5, 22)
        pc, bc = float(zz['gpf']) / K, float(zz['gbf']) / K; pf0, bf0 = float(z0['gpf']) / K, float(z0['gbf']) / K
        pe, be = float(e['gpf']), float(e['gbf'])
        res[nombre] = dict(pc=pc, bc=bc, pe=pe, be=be, etac=pc / bc, etae=pe / be, rpf=pc / pe, rbf=bc / be, rbf_s=(bc - bf0) / (be - float(e0['gbf'])),
                           rpf_s=(pc - pf0) / (pe - float(e0['gpf'])), piso=bf0 / bc, pisoe=float(e0['gbf']) / be, rq=(pc / bc) / (pe / be))
    a, b = res['6.86'], res[f'{x:g}']
    dif = b['rbf_s'] / a['rbf_s'] - 1
    print(f'\nκ₂/κ={k}: x = 6.86 → {x:g}')
    for c, tit in (('pc', 'γ_pf completo'), ('pe', 'γ_pf efectivo'), ('bc', 'γ_bf completo'), ('be', 'γ_bf efectivo'), ('etac', 'η completo'), ('etae', 'η efectivo'),
                   ('rpf', 'γ_pf c/e crudo'), ('rbf', 'γ_bf c/e crudo'), ('rpf_s', 'γ_pf c/e (piso T=0 restado)'), ('rbf_s', 'γ_bf c/e (piso T=0 restado)'),
                   ('rq', 'η c/e crudo'), ('piso', 'piso completo/γ_bf térmico completo'), ('pisoe', 'piso efectivo/γ_bf térmico efectivo')):
        print(f'  {tit:40s}: {a[c]:.5g} → {b[c]:.5g}')
    print(f'  cambio relativo del cociente γ_bf con piso restado: {dif:+.2%}  ({"SOSTENIDA (<3%)" if abs(dif) < 0.03 else "NO sostenida (>3%)"})')
    sx = {}
    for E in (100, 220):
        x6 = brentq(lambda t: a['rq'] * eta_e(t, k) - E, 6.5, 16, xtol=1e-3); x9 = brentq(lambda t: b['rq'] * eta_e(t, k) - E, 6.5, 16, xtol=1e-3)
        qi = lambda t: np.interp(t, [6.86, x], [a['rq'], b['rq']])
        xi = brentq(lambda t: qi(t) * eta_e(t, k) - E, 6.5, 16, xtol=1e-3)
        xe = brentq(lambda t: eta_e(t, k) - E, 6.5, 16, xtol=1e-3)
        sx[E] = (xe, x6, x9, xi)
        print(f'  x*({E}): efectivo {xe:.3f} | completo con q(6.86) {x6:.3f} | con q(x={x:g}) {x9:.3f} | con q interpolado {xi:.3f}')
    filas.append([k, x, a['rbf_s'], b['rbf_s'], dif, a['rq'], b['rq'], b['piso'], *sx[100], *sx[220]])
if filas:
    np.savetxt(os.path.join(C.DATA, 'termico_x_independencia.csv'), np.array(filas), delimiter=',', comments='',
               header='k2/k, x, gbf c/e floor-subtracted at 6.86, at x, rel change, eta c/e at 6.86, at x, floor/thermal at x, x*(100) eff, full q(6.86), full q(x), full q interp, x*(220) same')
