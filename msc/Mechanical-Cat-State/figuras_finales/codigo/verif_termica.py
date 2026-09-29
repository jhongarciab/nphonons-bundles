"""Validación de la figura térmica: modelo completo con filtro y temperatura (data/filtro_completo/*_x*.npz y T = 0)
frente al modelo efectivo (calc_termico) en los mismos parámetros. Reporta γ_pf, γ_bf y η y los cocientes
completo/efectivo; criterio de aceptación: ±5% en γ_pf y γ_bf, y γ_bf(T=0) < 1% del γ_bf térmico.
Escribe data/termico_validacion.csv."""
import os, glob
import numpy as np
import comun as C
import calc_termico as CT

PUNTOS = [(0.0535714, 0.25, 10.1), (0.0239579, 0.05, 6.86)]   # g_x en unidades de Ma (κ = 0.03, ω = 6)
KAP = 0.03                                                      # las tasas del completo se dividen por κ
filas = []
for gx, k2k, x in PUNTOS:
    ft = glob.glob(os.path.join(C.DATA, 'filtro_completo', f'gx{gx:.6g}_w6_gz0.42_kf0.3_al4_N22_Nf2_gam6e-07_x{x:g}.npz'))
    f0 = glob.glob(os.path.join(C.DATA, 'filtro_completo', f'gx{gx:.6g}_w6_gz0.42_kf0.3_al4_N22_Nf2_gam6e-07.npz'))
    if not ft:
        print(f"κ₂/κ={k2k}: falta la corrida térmica del modelo completo"); continue
    z = np.load(ft[0]); e = CT.punto(x, k2k, 14.0, 1, 2e-5, 22)
    e0 = CT.punto(60.0, k2k, 14.0, 1, 2e-5, 22)              # efectivo a T → 0 (n_q ~ 1e-26)
    gb0 = float(np.load(f0[0])['gbf']) / KAP if f0 else np.nan
    gpf_c, gbf_c = float(z['gpf']) / KAP, float(z['gbf']) / KAP; gpf_e, gbf_e = float(e['gpf']), float(e['gbf'])
    r = [k2k, x, gpf_c, gpf_e, gpf_c / gpf_e, gbf_c, gbf_e, gbf_c / gbf_e, gpf_c / gbf_c, gpf_e / gbf_e,
         (gpf_c / gbf_c) / (gpf_e / gbf_e), gb0, float(e0['gbf']), gb0 / gbf_c, float(z['herm_cruda']), *np.array(z['val'])]
    filas.append(r)
    ok = abs(r[4] - 1) <= 0.05 and abs(r[7] - 1) <= 0.05 and (r[13] < 0.01 if np.isfinite(r[13]) else False)
    print(f"κ₂/κ={k2k} x={x}: γ_pf c/e={r[4]:.4f}  γ_bf c/e={r[7]:.4f}  η c/e={r[10]:.4f} (η_c={r[8]:.4g}, η_e={r[9]:.4g})  "
          f"γ_bf(T=0) completo={gb0:.3e} efectivo={r[12]:.3e}  fracción del térmico={r[13]:.2e}  herm_cruda={r[14]:.1e}  "
          f"{'ACEPTADO' if ok else 'NO CUMPLE el criterio'}")
np.savetxt(os.path.join(C.DATA, 'termico_validacion.csv'), np.array(filas), delimiter=',', comments='',
           header='kappa_2/kappa, x, gamma_pf full, gamma_pf eff, ratio, gamma_bf full, gamma_bf eff, ratio, eta full, eta eff, ratio, '
                  'gamma_bf(T=0) full, gamma_bf(T=0) eff, gamma_bf(T=0)/gamma_bf thermal (full), herm before hermitizing, |Tr-1|, ||rho-rho^dag||, min eig')
