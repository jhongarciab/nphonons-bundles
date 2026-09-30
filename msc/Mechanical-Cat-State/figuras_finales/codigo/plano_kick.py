"""Hipótesis del desplazamiento (P7, fase 4): efectivo plano con los canales secundarios D[σ∓a], D[σ∓a†] (β = 2g_z/ω = 0.14),
(κ₂/κ, g_z/κ, γ/κ) = (0.25, 14, 2e-5), N = 22. Compara con el completo plano (γ_pf: T=0 8.331e-4, térmico 1.3694e-3, n_q off 8.445e-4,
n_m off 1.3580e-3; γ_bf: 3.04e-7, 2.869e-5; en unidades de κ). Reporta el incremento de γ_pf por n_q y por evento (÷ n_q κ)."""
import os, time
import numpy as np
import calc_termico as CT
x = 6.86; nq = 1 / np.expm1(x)
COMP = dict(T0=8.33135e-4, term=1.36936e-3, nqoff=8.44529e-4, nmoff=1.35797e-3, gbf_T0=3.04224e-7, gbf_term=2.86910e-5, gbf_nqoff=3.16571e-7, gbf_nmoff=2.86785e-5)
res = {}
for nombre, xx, off in (('T0', 60.0, ''), ('term', x, ''), ('nqoff', x, 'NQ'), ('nmoff', x, 'NM')):
    for v in ('NQ_OFF', 'NM_OFF'): os.environ.pop(v, None)
    if off: os.environ[off + '_OFF'] = '1'
    t0 = time.time(); e = CT.punto(xx, 0.25, 14.0, 0, 2e-5, 22, kick=True, rerun=True); dt = time.time() - t0
    e0 = CT.punto(xx, 0.25, 14.0, 0, 2e-5, 22)           # sin kick (caché del control anterior)
    res[nombre] = (float(e['gpf']), float(e['gbf']), float(e0['gpf']), float(e0['gbf']))
    print(f'{nombre:6s}: con kick γ_pf={res[nombre][0]:.5e} γ_bf={res[nombre][1]:.4e} | sin kick γ_pf={res[nombre][2]:.5e} γ_bf={res[nombre][3]:.4e} | '
          f'completo γ_pf={COMP[nombre]:.5e} γ_bf={COMP["gbf_" + nombre]:.4e} | tiempo {dt:.1f} s (calculado, rerun=True) herm={float(e["herm_cruda"]):.1e} mineig={float(e["mineig"]):+.1e}')
inc = lambda i: res['term'][i] - res['nqoff'][i]          # incremento atribuible a n_q (con n_m on)
print(f'\nincremento por n_q (término − n_q off, n_m on): kick γ_pf = {inc(0):.4e}; sin kick = {inc(2):.4e}; completo = {COMP["term"] - COMP["nqoff"]:.4e}')
print(f'incremento térmico total (térmico − T=0): kick = {res["term"][0] - res["T0"][0]:.4e}; sin kick = {res["term"][2] - res["T0"][2]:.4e}; completo = {COMP["term"] - COMP["T0"]:.4e}')
print(f'por evento (÷ n_q κ = {nq:.4e}): kick = {inc(0) / nq:.3f}; sin kick = {inc(2) / nq:.4f}; completo = {(COMP["term"] - COMP["nqoff"]) / nq:.3f}')
print(f'γ_bf térmico: kick {res["term"][1]:.4e}, sin kick {res["term"][3]:.4e}, completo {COMP["gbf_term"]:.4e}; γ_bf T=0: kick {res["T0"][1]:.4e}, sin kick {res["T0"][3]:.4e}, completo {COMP["gbf_T0"]:.4e}')
