"""Baño plano, κ₂/κ = 0.25, g_z/κ = 14: γ_pf del efectivo a T = 0 y térmico, con n_q y n_m apagados por separado, y fórmula analítica
γ_pf = 2[down·|α|² + up·(|α|²+1)], down = Γ₁⁻(n+1) + Γ₁⁺ n + γ(n_m+1), up = Γ₁⁻ n + Γ₁⁺(n+1) + γ n_m, Γ₁⁻ = κ(g_x/ω)², Γ₁⁺ = Γ₁⁻/9 (plano)."""
import os, sys
import numpy as np
import comun as C
import calc_termico as CT
k2k, gzk, gam, w = 0.25, 14.0, 2e-5, 200.0
gx = w * np.sqrt(k2k / (16 * gzk**2)); Gm = (gx / w)**2; Gp = Gm / 9
def analitica(a2, nq, nm):
    dn = Gm * (nq + 1) + Gp * nq + gam * (nm + 1); up = Gm * nq + Gp * (nq + 1) + gam * nm
    return 2 * (dn * a2 + up * (a2 + 1))
x = 6.86; nq = 1 / np.expm1(x); nm = 1 / np.expm1(x / 2)
casos = {'T=0 (x=60)': (60.0, ''), 'térmico': (x, ''), 'n_q off': (x, 'NQ'), 'n_m off': (x, 'NM')}
for nombre, (xx, off) in casos.items():
    for v in ('NQ_OFF', 'NM_OFF'): os.environ.pop(v, None)
    if off: os.environ[off + '_OFF'] = '1'
    e = CT.punto(xx, k2k, gzk, 0, gam, 22)
    q, m = (0.0 if off == 'NQ' else 1 / np.expm1(xx)), (0.0 if off == 'NM' else 1 / np.expm1(xx / 2))
    print(f'efectivo plano {nombre:12s}: γ_pf={float(e["gpf"]):.5e}  γ_bf={float(e["gbf"]):.5e}  analítica(α²=4)={analitica(4.0, q, m):.5e}  analítica(α²=3.798)={analitica(3.798, q, m):.5e}')
print(f'Γ₁⁻={Gm:.4e} Γ₁⁺={Gp:.4e} n_q={nq:.4e} n_m={nm:.4e}')
