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
    EPS_MIN = 1e-5       # por debajo mandan otros canales (pérdida intrínseca, temperatura, desfase del qubit)
    norm = LogNorm(EPS_MIN, 10)
    lev = np.geomspace(EPS_MIN, 10, 25)
    GAM_COLOR = 2e-5      # el color de (b) incluye el piso γ/κ = 2e-5
    eps_b_col = eps_b + GAM_COLOR / (ip(Df, KK) / c)
    for i, (e, lab) in enumerate([(eps_a, '(a) flat bath'), (eps_b, r'(b) filtered, $\kappa_f/\omega=0.05$')]):
        ax = fig.add_subplot(gs[i])
        cs = ax.contourf(GZ, KK, np.clip(e if i == 0 else eps_b_col, EPS_MIN * 1.01, 9.9), levels=lev, norm=norm, cmap='cividis')
        if i == 0:
            ax.contour(GZ, KK, e, levels=[1 / 220], colors='w', linewidths=1.0)
        else:
            # P10 resuelto (artefacto de N = 16): verificado con N = 20 en κ₂/κ = 0.03 y 0.3 (ε completo/mapa = 0.987, 0.975)
            # sin piso: líneas finas blancas (continua ε = 1/220, discontinua ε = 1e-3)
            ax.contour(GZ, KK, e, levels=[1 / 220], colors='w', linewidths=0.6)
            ax.contour(GZ, KK, e, levels=[1e-3], colors='w', linewidths=0.6, linestyles='--')
            # con piso intrínseco γ = ω/Q (ω/κ = 200): κ₁ → κ₁^filt + γ
            for gk, col in [(2e-4, E.OKABE[1]), (2e-5, E.OKABE[5])]:     # piso γ/κ = (ω/κ)/Q
                ep = e + gk / (ip(Df, KK) / c)
                ax.contour(GZ, KK, ep, levels=[1 / 220], colors=[col], linewidths=1.1)
                ax.contour(GZ, KK, ep, levels=[1e-3], colors=[col], linewidths=1.1, linestyles='--')
            from matplotlib.lines import Line2D
            ax.legend(handles=[Line2D([], [], color='w', lw=0.6, label='no floor'),
                               Line2D([], [], color=E.OKABE[1], lw=1.1, label=r'$\gamma/\kappa=2\times10^{-4}$'),
                               Line2D([], [], color=E.OKABE[5], lw=1.1, label=r'$\gamma/\kappa=2\times10^{-5}$'),
                               Line2D([], [], color='0.5', lw=1, label=r'$\epsilon=1/220$'),
                               Line2D([], [], color='0.5', lw=1, ls='--', label=r'$\epsilon=10^{-3}$')],
                      loc='lower right', fontsize=5.5, facecolor='0.3', framealpha=0.9, labelcolor='w').set_zorder(10)
            ax.contourf(GZ, KK, chi_mala.astype(float), levels=[0.5, 1.5], colors=['w'], alpha=0.25, zorder=4)
        if i == 0:
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
    # piso intrínseco: umbral y κ₂/κ desde el que el piso deja de importar (<10% de κ₁^filt en el umbral)
    for Q in (1e6, 1e7):
        gam = WK / Q
        print(f"Q={Q:.0e}: γ/κ={gam:.1e}; ε ≥ γ/κ₂^eff ⇒ ε=1/220 exige κ₂^eff/κ ≥ {220*gam:.3g}, ε=1e-3 exige ≥ {1e3*gam:.3g}")
        for k in (0.01, 0.03, 0.1, 0.3, 1.0):
            j = np.argmin(abs(kk - k)); row = eps_b[j] + gam / (ip(Df, kk[j]) / c)
            i = np.where(row < 1 / 220)[0]
            u = np.exp(np.interp(np.log(1 / 220), np.log(row[::-1]), np.log(gz[::-1]))) if len(i) and row.min() < 1 / 220 else np.nan
            print(f"   κ₂/κ={kk[j]:.3g}: umbral g_z/κ con piso={u:.4f} (sin piso {umb[j,2]:.4f}); piso/κ₁^filt en el umbral sin piso = {gam/(f_filt*kk[j]/(16*umb[j,2]**2)):.2f}")
    # parte del contorno ε = 1/220 con γ/κ = 2e-4 dentro de la zona verificada (χ|α|² ≤ 0.3κ)
    ep = eps_b + 2e-4 / (ip(Df, KK) / c)
    filas_ok = [(kk[j], gz[np.where(ep[j] < 1 / 220)[0]]) for j in range(len(kk)) if np.any(ep[j] < 1 / 220)]
    for k, gzs in filas_ok[::4]:
        gmin = gzs.min(); gchi = np.sqrt(k / (0.3 / (AL2 * (8 / 3) * WK / 16)))
        print(f"γ/κ=2e-4: κ₂/κ={k:.3f}: ε<1/220 para g_z/κ ≥ {gmin:.2f}; verificado (χ|α|²≤0.3κ) para g_z/κ ≥ {gchi:.2f}")
    print(f"predicción adiabática: factor = sqrt(f_filt/(10/9)) = {np.sqrt(f_filt/(10/9)):.4f}; κ_f/(2ω) = {KFW/2:.4f}")


if __name__ == '__main__':
    main()
