"""Genera data/tablas.md (tablas markdown D2) a partir de data/D2*.npz."""
import os, sys, numpy as np
D = os.path.join(os.path.dirname(__file__), '..', 'data')
def row(n, lam, reg, m):
    d = np.load(f'{D}/D2_n{n}_lam{lam}.npz'); r = np.load(f'{D}/D2b_rate_n{n}_lam{lam}.npz')
    Pe, Pc, Pv1, Je, Jc, xi, Pr, Jr = d['P_exact'], d['P_closed'], d['P_closed_v1'], d['J_exact'], d['J_closed'], d['xi'], r['P_rate'], r['J_rate']
    sel = m(xi, Pe) if m else np.ones_like(xi, bool)
    rel = lambda a, b: np.abs(a - b)[sel] / b[sel]
    imp = lambda a: np.abs((1 - a) - (1 - Pe))[sel] / (1 - Pe)[sel]
    e = rel(Pc, Pe)
    return (f'| {n} | {lam} | {reg} | {sel.sum()} | {e.max()*100:.1f} / {e.mean()*100:.2f} | {np.mean(e<.05)*100:.0f} / {np.mean(e<.10)*100:.0f} | {np.median(imp(Pc))*100:.0f} / {np.mean(imp(Pc))*100:.0f} | {np.median(imp(Pv1))*100:.0f} | '
            f'{rel(Jc, Je).mean()*100:.1f} / {rel(Jc, Je).max()*100:.0f} | {np.mean(rel(Pr, Pe))*100:.2f} | {np.median(imp(Pr))*100:.0f} | {rel(Jr, Je).mean()*100:.1f} |')
hdr = ('| n | lam | region | pts | P cerrada: err rel max / medio [%] | % malla <5 / <10 % | impureza 1-P cerrada: mediana / media [%] | idem formula previa (v1) mediana [%] | flujo cerrado: medio / max [%] | modelo de tasas: P err medio [%] | impureza mediana [%] | flujo medio [%] |\n|' + '---|' * 12)
L = [hdr]
for n in (2, 3):
    for lam in (0.1, 0.14, 0.2):
        L.append(row(n, lam, 'toda', None))
        L.append(row(n, lam, 'xi<0.5', lambda x, P: x < 0.5))
        L.append(row(n, lam, 'P_ex>=0.85 & xi<0.5', lambda x, P: (x < 0.5) & (P >= 0.85)))
open(f'{D}/tablas.md', 'w').write('\n'.join(L)); print('\n'.join(L))

# ---- tabla C ----
sys.path.insert(0, os.path.dirname(__file__))
from bundles_core import P_closed
c = np.load(f'{D}/C_mc.npz'); T = ['\n\n| n | Om | kap | gam | kap/gam | P_ep (exacta) | P_cerr (v2) | P_MC tau=2/k | 5/k | 10/k | pred. operativa 2/k, 5/k, 10/k | J_MC / J_exacto |', '|' + '---|' * 12]
for i in range(len(c['n'])):
    n, Om, kap, gam = int(c['n'][i]), c['Om'][i], c['kap'][i], c['gam'][i]
    pm = ' | '.join(f'{c["P_mc"][i][j]:.4f}±{c["P_mc_err"][i][j]:.4f}' for j in range(3))
    pr = ', '.join(f'{x:.3f}' for x in c['P_pred'][i])
    T.append(f'| {n} | {Om} | {kap} | {gam:.0e} | {kap/gam:.0f} | {c["P_ep"][i]:.4f} | {P_closed(0.14, Om, kap, gam, 4e-4, n):.4f} | {pm} | {pr} | {c["flux_mc"][i]/c["flux_exact"][i]:.3f}±{c["flux_mc_err"][i]/c["flux_exact"][i]:.3f} |')
open(f'{D}/tablas.md', 'a').write('\n'.join(T)); print('\n'.join(T))
