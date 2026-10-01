"""x*(η) del completo con el piso medido (corrección sugerida en la fase 6): γ_bf,c(x) = piso_c + r_s·(γ_bf,e(x) − piso_e), γ_pf,c(x) = q_pf·γ_pf,e(x), con
r_s = cociente de γ_bf con el piso de T = 0 restado y q_pf = cociente crudo de γ_pf, ambos medidos en el x del completo (x ≈ 9); piso_c de la corrida de T = 0 del
completo, piso_e del efectivo a x = 60. Se compara con la estimación anterior x*(η) con η_c(x) = q·η_e(x) (q = η c/e crudo medido en ese x)."""
import os
import numpy as np
from scipy.optimize import brentq
import comun as C
import calc_termico as CT
K = 0.03
GX = {0.05: 0.0239579, 0.1: 0.0338816, 0.25: 0.0535714, 0.4: 0.0677631}
XM = {0.05: 9.0, 0.1: 9.0, 0.25: 10.1, 0.4: 9.5}
ef = lambda x, k: CT.punto(x, k, 14.0, 1, 2e-5, 22)
def cargar(k, x):
    return np.load(os.path.join(C.DATA, 'filtro_completo', f'gx{GX[k]:.6g}_w6_gz0.42_kf0.3_al4_N22_Nf2_gam6e-07' + (f'_x{x:g}' if x else '') + '.npz'))
print('κ₂/κ  x_m | x*(100): antes → con piso (Δ) | x*(220): antes → con piso (Δ) | x*(100) y x*(220) del efectivo')
filas = []
for k in (0.05, 0.1, 0.25, 0.4):
    xm = XM[k]; z, z0 = cargar(k, xm), cargar(k, None)
    pf0c, bf0c = float(z0['gpf']) / K, float(z0['gbf']) / K; e0 = ef(60.0, k); bf0e = float(e0['gbf'])
    e = ef(xm, k); pc, bc = float(z['gpf']) / K, float(z['gbf']) / K
    q, qpf, rs = (pc / bc) / float(e['eta']), pc / float(e['gpf']), (bc - bf0c) / (float(e['gbf']) - bf0e)
    eta_c = lambda x: qpf * ef(x, k)['gpf'] / (bf0c + rs * (ef(x, k)['gbf'] - bf0e))
    r = [k, xm]
    for E in (100, 220):
        antes = brentq(lambda x: q * float(ef(x, k)['eta']) - E, 6.5, 16, xtol=1e-4)
        con = brentq(lambda x: eta_c(x) - E, 6.5, 16, xtol=1e-4)
        r += [antes, con, con - antes]
    xe = [brentq(lambda x: float(ef(x, k)['eta']) - E, 6.5, 16, xtol=1e-4) for E in (100, 220)]
    print(f'{k:5.2f} {xm:5.1f} | {r[2]:.3f} → {r[3]:.3f} ({r[4]:+.3f}) | {r[5]:.3f} → {r[6]:.3f} ({r[7]:+.3f}) | {xe[0]:.3f} {xe[1]:.3f}   [piso_c={bf0c:.3e} piso_e={bf0e:.3e} r_s={rs:.4f} q_pf={qpf:.4f} q={q:.4f}]')
    filas.append(r + xe)
np.savetxt(os.path.join(C.DATA, 'termico_x_estrella_piso.csv'), np.array(filas), delimiter=',', comments='',
           header='k2/k, x_m, x*(100) q*eta_e, x*(100) with floor, diff, x*(220) q*eta_e, x*(220) with floor, diff, x*(100) eff, x*(220) eff')
