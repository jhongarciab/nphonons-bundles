"""Cifras de la sección térmica de msc/HANDOFF.md, cada una leída de su archivo de datos. Escribe data/handoff_cifras.txt (clave = valor, archivo fuente).
Unidades: tasas en κ (las del completo, en unidades de Ma, se dividen entre κ = 0.03)."""
import os, glob, re, json
import numpy as np
import comun as C
import calc_termico as CT
K = 0.03
GX = {0.05: 0.0239579, 0.1: 0.0338816, 0.25: 0.0535714, 0.4: 0.0677631}
FC = os.path.join(C.DATA, 'filtro_completo')
def comp(k, x=None, N=22, plano=False, suf=''):
    g = GX[k]
    f = os.path.join(FC, f'gx{g:.6g}_w6_gz0.42_kf0.3_al4_N{N}_Nf{1 if plano else 2}' + ('_plano' if plano else '') + '_gam6e-07' + (f'_x{x:g}' if x else '') + suf + '.npz')
    return np.load(f), os.path.relpath(f, C.RAIZ)
ef = lambda x, k, filtro=1, **kw: CT.punto(x, k, 14.0, filtro, 2e-5, 22, **kw)
V = {}
def put(clave, valor, fuente): V[clave] = (valor, fuente)
XM = {0.05: 9.0, 0.1: 9.0, 0.25: 10.1, 0.4: 9.5}
for k in GX:
    z0, f0 = comp(k)
    e0 = ef(60.0, k)
    put(f'piso_c[{k}]', float(z0['gbf']) / K, f0); put(f'piso_e[{k}]', float(e0['gbf']), 'data/termico/x60_k%g_gz14_f1_g2e-05_N22_Nf2_w200_a4.npz' % k)
    put(f'gpf0_c[{k}]', float(z0['gpf']) / K, f0); put(f'gpf0_e[{k}]', float(e0['gpf']), 'idem efectivo x=60')
    for xx in (6.86, XM[k]):
        z, f = comp(k, xx); e = ef(xx, k); pc, bc, pe, be = float(z['gpf']) / K, float(z['gbf']) / K, float(e['gpf']), float(e['gbf'])
        for nom, val in (('gpf_c', pc), ('gbf_c', bc), ('gpf_e', pe), ('gbf_e', be), ('eta_c', pc / bc), ('eta_e', pe / be), ('r_pf', pc / pe), ('r_bf', bc / be),
                         ('r_bf_s', (bc - float(z0['gbf']) / K) / (be - float(e0['gbf']))), ('q_eta', (pc / bc) / (pe / be)), ('piso_frac_c', float(z0['gbf']) / float(z['gbf'])),
                         ('herm', float(z['herm_cruda'])), ('tr', float(z['val'][0])), ('mineig', float(z['val'][2])), ('t_prop_s', float(z['tprop']))):
            put(f'{nom}[{k},{xx:g}]', val, f)
D = {k: v[0] for k, v in V.items()}
with open(os.path.join(C.DATA, 'handoff_cifras.txt'), 'w') as o:
    o.write('# clave = valor | archivo fuente. Generado por codigo/handoff_cifras.py (tasas en κ).\n')
    for k, (v, f) in V.items(): o.write(f'{k} = {v:.6g} | {f}\n')
print(len(V), 'cifras escritas')
