"""Figura central — mapa de diseño con baño plano (a) y filtrado (b), y verificación por colapso (c).
Solo lee cachés: data/minimo/ (Δ plano, |α|²=4), data/minimo_filtro/ (Δ con filtro κ_f/κ = 10, N_f = 3),
data/fig3/ y data/pfig3/ (puntos del modelo completo para (c)).
ε = κ₁/κ₂^eff,  κ₂^eff = Δ/c con c = 4.14 (pendiente adiabática del baño PLANO: el costo del filtro queda incluido).
  plano:    κ₁/κ = (10/9)(g_x/ω)²
  filtrado: κ₁/κ = (g_x/ω)²[κ_eff(ω)/κ + κ_eff(3ω)/(9κ)],  κ_eff(δ)/κ = (κ_f/δ)²/(4 + (κ_f/δ)²)... con δ = ω, 3ω
  (g_x/ω)² = (κ₂/κ)/(16(g_z/κ)²).
Región sombreada: χ|α|² > 0.3κ, χ = (8/3)g_x²/ω, ω/κ = 200, |α|² = 4  ->  κ₂/κ > 0.00225 (g_z/κ)².
Escribe data/figura_central_mapas.npz y data/figura_central_umbral.csv.
"""
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import comun as C
import estilo as E
import calc_minimo as CM
import principal_fig2 as PF2

KFW = 0.05          # κ_f/ω
WK = 200.0          # ω/κ para la región de χ
AL2 = 4.0


def delta(carpeta, patron, clave_conf=None):
    T = []
    for f in glob.glob(os.path.join(C.DATA, carpeta, patron)):
        z = np.load(f)
        conf = float(z['conf']) if clave_conf else CM.tasa_dinamica(z['t'], z['Pc_din'][0])
        T.append((float(z['k2k']), conf))
    return np.array(sorted(T))


def main():
    Dp = delta('minimo', 'k*_N24.npz')
    Df = delta('minimo_filtro', 'k*_kf10_N20_Nf3.npz', True)
    K_VERIF = 0.1       # panel (b): por encima, κ₁ filtrado no verificado (P10)
    KMAX_F = 1.5        # con filtro y κ₂/κ > 1.5 el retorno de P_c no es monótono (sobrepaso): tasa no definida
    Df = Df[(Df[:, 0] <= KMAX_F) & np.isfinite(Df[:, 1])]
    c = np.polyval(np.polyfit(Dp[:3, 0], Dp[:3, 1] / Dp[:3, 0], 1), 0.0)
    ip = lambda T, x: np.exp(np.interp(np.log(x), np.log(T[:, 0]), np.log(T[:, 1])))
    gz = np.geomspace(0.05, 100, 400); kk = np.geomspace(1e-3, 10, 300)
    GZ, KK = np.meshgrid(gz, kk)
    gxw2 = KK / (16 * GZ**2)
    f_filt = KFW**2 / (4 + KFW**2) + KFW**2 / (36 + KFW**2) / 9
    eps_a = (10 / 9) * gxw2 / (ip(Dp, KK) / c)
    eps_b = f_filt * gxw2 / (ip(Df, KK) / c)
    eps_b[KK > KMAX_F] = np.nan
    chi_mala = AL2 * (8 / 3) * gxw2 * WK > 0.3
    np.savez(os.path.join(C.DATA, 'figura_central_mapas.npz'), gz=gz, k2=kk, eps_plano=eps_a, eps_filtro=eps_b, c=c, f_filt=f_filt)
    # umbral ε = 1/220 en g_z/κ para cada κ₂/κ
    umb = []
    for j, k in enumerate(kk):
        u = []
        for e in (eps_a, eps_b):
            row = e[j]; i = np.where(row < 1 / 220)[0]
            u.append(np.exp(np.interp(np.log(1 / 220), np.log(row[::-1]), np.log(gz[::-1]))) if len(i) and i[0] > 0 else np.nan)
        umb.append([k, *u, u[1] / u[0], ip(Df, k) / ip(Dp, k)])
    umb = np.array(umb)
    np.savetxt(os.path.join(C.DATA, 'figura_central_umbral.csv'), umb, delimiter=',', comments='',
               header='kappa_2/kappa, g_z/kappa threshold eps=1/220 (flat), idem (filter kappa_f/omega=0.05), ratio filter/flat, '
                      'Delta_filter/Delta_flat')

    E.aplicar()
    fig = plt.figure(figsize=(E.COL2, 2.7))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 0.5, 0.8], wspace=0.12)
    norm = LogNorm(1e-7, 10)
    lev = np.geomspace(1e-7, 10, 33)
    for i, (e, lab) in enumerate([(eps_a, '(a) flat bath'), (eps_b, r'(b) filtered, $\kappa_f/\omega=0.05$')]):
        ax = fig.add_subplot(gs[i])
        cs = ax.contourf(GZ, KK, np.clip(e, 1.01e-7, 9.9), levels=lev, norm=norm, cmap='cividis')
        if i == 0:
            ax.contour(GZ, KK, e, levels=[1 / 220], colors='w', linewidths=1.0)
        else:
            # P10: en régimen saturado (κ₂/κ ≳ 0.1) el κ₁ filtrado no está verificado (exceso ×2.3 en κ₂/κ = 0.3)
            ea = np.where(KK <= K_VERIF, e, np.nan); eb = np.where(KK >= K_VERIF, e, np.nan)
            ax.contour(GZ, KK, ea, levels=[1 / 220], colors='w', linewidths=1.0)
            ax.contour(GZ, KK, eb, levels=[1 / 220], colors='w', linewidths=1.0, linestyles='--')
            ax.axhspan(K_VERIF, KMAX_F, color='w', alpha=0.18, lw=0)
            ax.text(8, 0.35, 'not verified\n(P10)', fontsize=5.5, color='w', ha='center')
        ax.contourf(GZ, KK, chi_mala.astype(float), levels=[0.5, 1.5], colors=['w'], alpha=0.28)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'$g_z/\kappa$')
        if i == 0:
            ax.set_ylabel(r'$\kappa_2/\kappa$')
        else:
            ax.set_yticklabels([])
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, va='top', fontsize=7.5, color='k',
                bbox=dict(boxstyle='round,pad=0.2', fc='w', ec='none', alpha=0.8))
        if i == 1:
            ax.axhspan(KMAX_F, 10, color='0.75', zorder=0)
            ax.text(0.3, 3.2, 'non-monotonic return', fontsize=6, ha='left', va='center')
        ax.set_ylim(1e-3, 10)
    cax = fig.add_subplot(gs[2])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='cividis'), cax=cax)
    cb.ax.set_title(r'$\epsilon$', fontsize=8, pad=3)
    cb.ax.axhline(1 / 220, color='w', lw=1.0)
    # (c) colapso
    cx = fig.add_subplot(gs[4])
    T = PF2.cargar(); ok = T[:, 8] > 0.99
    sc = cx.scatter(T[ok, 6], T[ok, 11] / (5 / 72), c=T[ok, 6], norm=LogNorm(3e-3, 1.2), cmap='viridis', s=14, edgecolors='k', linewidths=0.3)
    cx.axhline(1, color='k', lw=0.7); cx.set_xscale('log'); cx.set_ylim(0.95, 1.01)
    cx.set_xlabel(r'$\kappa_2/\kappa$'); cx.set_ylabel(r'$(\kappa_1/\kappa_2)(g_z/\kappa)^2/(5/72)$', fontsize=7)
    cx.text(0.04, 0.96, '(c)', transform=cx.transAxes, va='top', fontsize=8)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(C.AQUI, f'figura_central.{ext}'))
    print(f"c = {c:.4f}; f_filt = κ₁_filt/[(g_x/ω)²κ] = {f_filt:.4e} (plano 10/9)")
    for k in (1e-3, 0.01, 0.03, 0.1, 0.3, 1.0):
        j = np.argmin(abs(kk - k))
        print(f"κ₂/κ={kk[j]:.3g}: umbral g_z/κ plano={umb[j,1]:.3f} filtro={umb[j,2]:.4f} factor={umb[j,3]:.4f}  Δ_f/Δ_p={umb[j,4]:.3f}")
    print(f"predicción adiabática: factor = sqrt(f_filt/(10/9)) = {np.sqrt(f_filt/(10/9)):.4f}; κ_f/(2ω) = {KFW/2:.4f}")


if __name__ == '__main__':
    main()
