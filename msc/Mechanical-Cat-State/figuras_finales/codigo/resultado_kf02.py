"""Resultado de la pareja κ_f/ω = 0.2 (N = 20) contra la regla de decisión de data/preregistro_kf0.2.md (fijada antes de lanzar). κ = 1. Salida: data/resultado_kf0.2.txt"""
import os, numpy as np
import comun as C
K = 0.03; FC = os.path.join(C.DATA, 'filtro_completo')
f = lambda s: np.load(os.path.join(FC, f'gx0.0535714_w6_gz0.42_kf1.2_al4_N20_Nf2_gam6e-07{s}.npz'))
zt, z0 = f('_x6.86'), f('')
# efectivo del pre-registro (valores fijados antes de las corridas)
g0e, g1e = 1.66411e-04, 1.78395e-04; De = g1e - g0e; Ea = 7.341e-06
pt, p0 = float(zt['gpf']) / K, float(z0['gpf']) / K; bt, b0 = float(zt['gbf']) / K, float(z0['gbf']) / K
r0, rx = p0 / g0e, pt / g1e; Eo = (rx - r0) * g1e; Dc = (pt - p0) - De
out = [f'completo κ_f/ω = 0.2, N = 20, (κ₂/κ, x) = (0.25, 6.86), κ = 1',
       f'γ_pf  T=0 {p0:.5e}  térmico {pt:.5e}  | γ_bf T=0 {b0:.4e}  térmico {bt:.4e} | η térmico {pt/bt:.3f}, η(T=0) {p0/b0:.1f}',
       f'efectivo (pre-registro): γ_pf T=0 {g0e:.5e}  térmico {g1e:.5e}  Δ_e {De:.4e}  E_ansatz {Ea:+.3e}',
       f'r_0 = {r0:.5f}   r_x = {rx:.5f}',
       f'Δ_c = γ_pf(térm) − γ_pf(T=0) = {pt-p0:.4e}   D = Δ_c − Δ_e = {Dc:+.3e}   (sin canal esperado ≈ {(r0-1)*De:+.2e}; con canal ≈ {(r0-1)*De+Ea:+.2e})',
       f'E_obs = (r_x − r_0)·γ_e = {Eo:+.3e}   E_obs/E_ansatz = {Eo/Ea:.3f}',
       f'al2eff: T=0 {z0["al2eff"].real:.5f}  térmico {zt["al2eff"].real:.5f}  (parte imaginaria {z0["al2eff"].imag:+.1e}, {zt["al2eff"].imag:+.1e})',
       f'hermiticidad cruda: T=0 {float(z0["herm_cruda"]):.2e}, térmico {float(zt["herm_cruda"]):.2e}; mín. autovalor {float(z0["val"][2]):+.1e}, {float(zt["val"][2]):+.1e}; |Tr−1| {float(z0["val"][0]):.0e}, {float(zt["val"][0]):.0e}']
dec = 'canal de tamaño ansatz' if Eo > 3e-6 else ('no hay canal' if abs(Eo) < 1e-6 else 'indeterminado')
out.append(f'REGLA (sobre E_obs): E_obs > 3e-6 canal de tamaño ansatz; |E_obs| < 1e-6 no hay canal; resto indeterminado  ->  {dec}')
txt = '\n'.join(out); print(txt); open(os.path.join(C.DATA, 'resultado_kf0.2.txt'), 'w').write(txt + '\n')
