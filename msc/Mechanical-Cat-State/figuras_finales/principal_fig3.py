"""Figura principal 3 — espacio de diseño. Solo lee cachés:
  data/minimo/k*_N24.npz  (calc_minimo.py): tasa de confinamiento Δ(κ₂) del modelo mínimo, |α|² = 4
  data/pfig3/*.npz        (calc_pfig3.py): puntos del modelo completo (γ_pf, confinamiento dinámico)
Mapa en (g_z/κ, κ₂/κ): ε = κ₁/κ₂^eff, κ₁ = (10/9)g_x²κ/ω² con (g_x/ω)² = (κ₂/κ)/(16(g_z/κ)²),
κ₂^eff = Δ(κ₂)/c, c = lim_{κ₂→0} Δ/κ₂. En régimen adiabático ε = (5/72)(κ/g_z)².
Δ: tasa DINÁMICA desde |0>|g> (C9); se usa esa y se reporta también la espectral.
Escribe data/principal_fig3_minimo.csv, data/principal_fig3_puntos.csv y el mapa en data/principal_fig3_mapa.npz.
"""
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import comun as C
import estilo as E
import calc_minimo as CM
import fig4 as F4            # reutiliza la función de ajuste del retorno (misma definición que la Fig. 4)

KAP = 0.03
W_OVER_K = 200.0              # para la sombra de validez perturbativa


def minimo():
    filas = []
    for f in glob.glob(os.path.join(C.DATA, 'minimo', 'k*_N24.npz')):
        z = np.load(f)
        td = [CM.tasa_dinamica(z['t'], s) for s in z['Pc_din']]
        # brecha espectral recalculada desde los modos guardados: umbral 1e-6 (los ~1e-9 son ruido del espacio oscuro)
        fis = [m for m in z['modos'] if -m[0] > 1e-6 and m[2] < 0.5]
        filas.append([float(z['k2k']), td[0], td[1], -fis[0][0] if fis else np.nan])
    T = np.array(sorted(filas))
    return T


def conf_modelo_completo(z):
    """Retorno dinámico de P_c(t) (código fijo) desde |0>|g>: misma ventana que la Fig. 4."""
    t = z['t']; tasas = []
    for serie in z['Pc_din']:
        e = np.abs(serie - serie[-1])
        i1 = np.argmax(e < 0.1 * e.max())
        piso = np.median(e[int(0.9 * len(e)):]) + 1e-12
        fin = np.where(e < 3 * piso)[0]; i2 = fin[fin > i1][0] if np.any(fin > i1) else len(e)
        tasas.append(-np.polyfit(t[i1:i2], np.log(e[i1:i2]), 1)[0] if i2 - i1 >= 10 else np.nan)
    return tasas


def main():
    Tm = minimo()
    k2, D0, D1, Dsp = Tm.T
    # pendiente adiabática: extrapolación lineal de Δ/κ₂ a κ₂ → 0 con los tres κ₂/κ más pequeños
    c = np.polyval(np.polyfit(k2[:3], D0[:3] / k2[:3], 1), 0.0)
    np.savetxt(os.path.join(C.DATA, 'principal_fig3_minimo.csv'), np.c_[Tm, D0 / k2, D0 / (c * k2)], delimiter=',', comments='',
               header='kappa_2/kappa, Delta dynamic from |0>|g> [kappa], Delta dynamic from |1.3 alpha>|g> [kappa], '
                      'Delta spectral (C9 edge<0.5) [kappa], Delta/kappa_2, kappa_2^eff/kappa_2 = Delta/(c kappa_2)')
    lk, lD = np.log(k2), np.log(D0)
    Delta = lambda x: np.exp(np.interp(np.log(x), lk, lD))
    gz = np.geomspace(1, 100, 300); kk = np.geomspace(1e-3, 10, 300)
    GZ, KK = np.meshgrid(gz, kk)
    k1 = (10 / 9) * KK / (16 * GZ**2)              # κ₁/κ
    eps = k1 / (Delta(KK) / c)
    np.savez(os.path.join(C.DATA, 'principal_fig3_mapa.npz'), gz=gz, k2=kk, eps=eps, c=c)
    # puntos del modelo completo
    P = []
    for f in sorted(glob.glob(os.path.join(C.DATA, 'pfig3', '*.npz'))):
        z = np.load(f)
        gx, w, gzk = float(z['gx']), float(z['w']), float(z['gzk'])
        gm = gx**2 * KAP / (w**2 + KAP**2 / 4); gp = gx**2 * KAP / (9 * w**2 + KAP**2 / 4); r = gp / gm
        a2 = abs(complex(z['al2eff'])); g = float(z['gpf'])
        k1f = g * (1 + r) / (2 * (a2 * (1 + r) + r))
        cf = conf_modelo_completo(z)
        k2k = float(z['kap2']) / KAP
        eps_full = k1f / (cf[0] / c)
        eps_map = (10 / 9) * k2k / (16 * gzk**2) / (Delta(k2k) / c)
        P.append([gx, w, gzk, k2k, g, k1f, cf[0], cf[1], cf[0] / KAP / Delta(k2k), eps_full, eps_map, eps_full / eps_map])
    P = np.array(P)
    np.savetxt(os.path.join(C.DATA, 'principal_fig3_puntos.csv'), P, delimiter=',', comments='',
               header='g_x, omega, g_z/kappa, kappa_2/kappa, gamma_pf, kappa_1 (C1 inverted), confinement dyn |0>|g>, confinement dyn |1.3a>|g>, '
                      'confinement full / Delta_min, eps full model, eps map, ratio full/map')

    E.aplicar()
    fig, ax = plt.subplots(figsize=(E.COL1, 2.9))
    lev = np.geomspace(1e-6, 1, 25)
    cs = ax.contourf(GZ, KK, eps, levels=lev, norm=LogNorm(), cmap='cividis')
    ax.contour(GZ, KK, eps, levels=[1e-3, 1 / 220], colors=['w', 'w'], linestyles=['--', '-'], linewidths=0.9)
    # sombra fuera de validez perturbativa (ω/κ = 200): g_z/ω > 0.1 o g_x/ω > 0.1
    malo = (GZ / W_OVER_K > 0.1) | (np.sqrt(KK) / (4 * GZ) > 0.1)
    ax.contourf(GZ, KK, malo.astype(float), levels=[0.5, 1.5], colors='none', hatches=['////'])
    if len(P):
        ax.scatter(P[:, 2], P[:, 3], c=P[:, 9], norm=LogNorm(lev[0], lev[-1]), cmap='cividis', edgecolors='w', s=22, lw=0.7, zorder=5)
    for nom, x, y, mk in [('Ma', 7.1, 1.0, '*'), ('Naseem', 60, 2.07, 'P')]:
        ax.plot(x, y, mk, color=E.OKABE[1], ms=8, mec='k', mew=0.5, zorder=6)
        ax.annotate(nom, (x, y), xytext=(4, 4) if nom == 'Ma' else (-30, -12), textcoords='offset points', fontsize=7, color='w')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color='w', lw=0.9, label=r'$\epsilon=1/220$'),
                       Line2D([], [], color='w', lw=0.9, ls='--', label=r'$\epsilon=10^{-3}$'),
                       Line2D([], [], marker='o', ls='none', mfc='0.6', mec='w', label='full model')],
              loc='lower right', fontsize=6.3, facecolor='0.35', framealpha=0.8, labelcolor='w')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$g_z/\kappa$'); ax.set_ylabel(r'$\kappa_2/\kappa$')
    cb = fig.colorbar(cs, ax=ax, pad=0.02); cb.set_label(r'$\epsilon=\kappa_1/\kappa_2^{\rm eff}$')
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(C.AQUI, f'principal_fig3.{ext}'))
    print(f"c = lim Δ/κ₂ = {c:.4f} (extrapolado; Δ/κ₂ en los 3 menores κ₂/κ: {D0[:3]/k2[:3]})")
    print("κ₂/κ  Δ_din(|0>)  Δ_din(|1.3α>)  Δ_esp  Δ/κ₂  κ₂eff/κ₂")
    for row in np.c_[Tm, D0 / k2, D0 / (c * k2)]:
        print("  " + " ".join(f"{v:.4g}" for v in row))
    # verificación adiabática
    i = 0
    print(f"verif. adiabática: κ₂/κ={k2[i]:.3g}: ε·(g_z/κ)² = {(10/9)*k2[i]/16 / (D0[i]/c):.5f} vs 5/72 = {5/72:.5f}")
    print("puntos modelo completo: g_x ω g_z/κ κ₂/κ conf_full/Δ_min ε_full ε_map razón")
    for r in P:
        print(f"  {r[0]:.3f} {r[1]:.0f} {r[2]:4.1f} {r[3]:.3f} {r[8]:.3f} {r[9]:.3e} {r[10]:.3e} {r[11]:.3f}")


if __name__ == '__main__':
    main()
