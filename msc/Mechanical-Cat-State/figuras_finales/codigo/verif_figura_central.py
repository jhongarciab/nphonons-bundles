"""Verificación del panel (b) de la figura central con el modelo completo con filtro (data/filtro_completo/).
κ₁ completo: γ_pf espectral invirtiendo C1 con Γ₁± filtrados y |α_eff²|; κ₂^eff completo = confinamiento dinámico / c.
Se compara ε_completo con ε del mapa en el mismo (g_z/κ, κ₂/κ). Escribe data/figura_central_verificacion.csv."""
import os, glob
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import comun as C
import principal_fig3 as PF3

m = np.load(os.path.join(C.DATA, 'figura_central_mapas.npz'))
interp = RegularGridInterpolator((np.log(m['k2']), np.log(m['gz'])), np.log(m['eps_filtro']))
c = float(m['c'])
filas = []
for f in sorted(glob.glob(os.path.join(C.DATA, 'filtro_completo', '*.npz'))):
    z = np.load(f)
    gx, w, gz, kap, kf = (float(z[k]) for k in ('gx', 'w', 'gz', 'kap', 'kf'))
    ke = lambda d: kap * kf**2 / (4 * d**2 + kf**2)
    gm, gp = gx**2 * ke(w) / w**2, gx**2 * ke(3 * w) / (9 * w**2); r = gp / gm
    a2 = abs(complex(z['al2eff'])); g = float(z['gpf'])
    k1 = g * (1 + r) / (2 * (a2 * (1 + r) + r))
    conf = PF3.conf_modelo_completo(dict(t=z['t'], Pc_din=[z['Pc_din']]))[0]
    k2k = float(z['kap2']) / kap; gzk = gz / kap
    eps_full = k1 / (conf / c) / kap if False else k1 / (conf / c)
    eps_map = float(np.exp(interp([[np.log(k2k), np.log(gzk)]])[0]))
    # comparación de las dos piezas por separado
    k1_pred = gm + gp
    dmin = np.exp(np.interp(np.log(k2k), np.log(PF3.minimo()[:, 0]), np.log(PF3.minimo()[:, 1])))
    filas.append([k2k, gzk, float(z['Pc_ss']), k1 / k1_pred, conf, eps_full, eps_map, eps_full / eps_map])
    print(f"κ₂/κ={k2k:.3f} g_z/κ={gzk:.1f} P_c={float(z['Pc_ss']):.5f}: κ₁/κ₁_pred={k1/k1_pred:.4f}  conf={conf:.4e} "
          f"(conf/(Δ_plano·κ)={conf/(dmin*kap):.3f})  ε_completo={eps_full:.4e} ε_mapa={eps_map:.4e}  razón={eps_full/eps_map:.3f}  val={z['val']}")
np.savetxt(os.path.join(C.DATA, 'figura_central_verificacion.csv'), np.array(filas), delimiter=',', comments='',
           header='kappa_2/kappa, g_z/kappa, P_c, kappa_1 full / pred (filtered), confinement full [2pi GHz], eps full, eps map (b), ratio')
