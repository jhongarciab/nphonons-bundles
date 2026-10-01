"""Diagnóstico (2026-10-01), no confirmación del canal: en los 8 puntos (κ₂/κ, x) de la verificación con filtro (N = 22), E_obs = (r_x − r_0)·γ_e(x) con r_0 = cociente
completo/efectivo de γ_pf a T = 0 de cada κ₂/κ y r_x el cociente a x; E_ansatz = aumento de γ_pf del efectivo con filtro por el ansatz del desplazamiento con n_q en ese punto
(γ_pf con ansatz − γ_pf sin ansatz, mismo x). Criterio fijado antes: 'consistente' si E_obs/E_ansatz ∈ [0.7, 1.3] en los 8 puntos; si no, 'no consistente'. κ = 1."""
import os, numpy as np
import comun as C
import calc_termico as CT
np.savez = lambda *a, **k: None          # no escribir cachés
K = 0.03; FC = os.path.join(C.DATA, 'filtro_completo')
GX = {0.05: 0.0239579, 0.1: 0.0338816, 0.25: 0.0535714, 0.4: 0.0677631}; XM = {0.05: 9.0, 0.1: 9.0, 0.25: 10.1, 0.4: 9.5}
comp = lambda k, x=None: np.load(os.path.join(FC, f'gx{GX[k]:.6g}_w6_gz0.42_kf0.3_al4_N22_Nf2_gam6e-07' + (f'_x{x:g}' if x else '') + '.npz'))
out = ['κ₂/κ   x      r_0      r_x      γ_e(x)       E_obs        E_ansatz     E_obs/E_ansatz']; ratios = []
for k in GX:
    z0 = comp(k); e0 = CT.punto(60.0, k, 14.0, 1, 2e-5, 22); r0 = float(z0['gpf']) / K / float(e0['gpf'])
    for x in (6.86, XM[k]):
        z = comp(k, x); e = CT.punto(x, k, 14.0, 1, 2e-5, 22); ek = CT.punto(x, k, 14.0, 1, 2e-5, 22, kick=True, rerun=True)
        rx = float(z['gpf']) / K / float(e['gpf']); ge = float(e['gpf']); Eo = (rx - r0) * ge; Ea = float(ek['gpf']) - ge; ratios.append(Eo / Ea)
        out.append(f'{k:<6g} {x:<6g} {r0:.5f}  {rx:.5f}  {ge:.5e}  {Eo:+.3e}  {Ea:+.3e}  {Eo / Ea:.3f}')
ok = all(0.7 <= r <= 1.3 for r in ratios)
out.append(f'E_obs/E_ansatz entre {min(ratios):.3f} y {max(ratios):.3f}; en [0.7, 1.3] en {sum(0.7 <= r <= 1.3 for r in ratios)} de {len(ratios)} puntos: {"consistente" if ok else "no consistente"} (criterio fijado antes: los 8)')
txt = '\n'.join(out); print(txt); open(os.path.join(C.DATA, 'diagnostico_E_obs.txt'), 'w').write(txt + '\n')
