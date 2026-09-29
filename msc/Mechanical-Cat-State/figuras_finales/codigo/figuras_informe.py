"""Figuras del informe (borrador para discusión), en español y con tasas normalizadas a κ.
Solo LEE cachés y CSV de data/ (no propaga ni escribe CSV). Salida: ../../../docs/Informe/figuras/.
  fig_resonancia   (a) P_c(ω_p) de Ma con recuadros de Wigner (gato / sin gato); (b) colapso en x = (ω_p − ω_p*)/G
  fig_merito       (a) (κ₁/κ₂)(g_z/κ)²/(5/72) frente a κ₂/κ, color g_x/ω; (b) γ_pf medido/predicho frente a |α|², con y sin el +1
  fig_filtro       (a) Γ₁^± / κ frente a κ_f/κ; (b) mejora de γ_pf y confinamiento relativo al baño plano
  fig_chi          confinamiento completo / Δ_min frente a χ|α|²/κ (efectivo con y sin χ)
  fig_mapa         ε = κ₁/κ₂^eff en (g_z/κ, κ₂/κ): (a) baño plano, (b) baño filtrado con piso intrínseco
Uso: cd figuras_finales && ../.venv_qutip5/bin/python codigo/figuras_informe.py [resonancia merito filtro chi mapa]
"""
import os, sys, csv, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator
import comun as C
import estilo as E

SALIDA = os.path.abspath(os.path.join(C.RAIZ, '..', '..', 'docs', 'Informe', 'figuras'))
KAP, W, GX = 0.03, 6.0, 0.3 * np.sin(np.pi / 4)


def guardar(fig, nombre):
    os.makedirs(SALIDA, exist_ok=True)
    fig.savefig(os.path.join(SALIDA, nombre + '.pdf'))
    fig.savefig(os.path.join(SALIDA, nombre + '.png'))
    plt.close(fig)
    print('escrito', nombre)


def rotulo(ax, txt, x=0.03, y=0.96, **kw):
    ax.text(x, y, txt, transform=ax.transAxes, va='top', ha='left', fontsize=9, **kw)


# ---------------------------------------------------------------- resonancia
def resonancia():
    E.aplicar()
    fig = plt.figure(figsize=(E.COL2, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.22)
    ax = fig.add_subplot(gs[0])
    A = np.genfromtxt(os.path.join(C.DATA, 'fig2a.csv'), delimiter=',', skip_header=1)
    wp, pc = A[:, 0], A[:, 1]
    ww = np.linspace(wp[0], wp[-1], 500)
    ax.plot(ww, PchipInterpolator(wp, pc)(ww), color=E.OKABE[0])
    ax.plot(wp, pc, 'o', color=E.OKABE[0], ms=2.5)
    wps = 2 * (6 - 4 * GX**2 / (3 * 6))
    ax.axvline(wps, color='k', ls=':', lw=0.8)
    ax.axvline(12.0, color='0.5', ls='-.', lw=0.8)
    ax.text(wps - 0.0008, 0.05, r'$\omega_p^*$', rotation=90, ha='right', va='bottom', fontsize=7.5)
    ax.text(12.0 + 0.0008, 0.05, r'$2\omega$', rotation=90, ha='left', va='bottom', fontsize=7.5, color='0.35')
    ax.set_xlim(11.953, 12.017); ax.set_ylim(0, 1.03)
    ax.set_xlabel(r'$\omega_p/2\pi$ (GHz)'); ax.set_ylabel(r'$P_c$')
    rotulo(ax, '(a)')
    z = np.load(os.path.join(C.DATA, 'principal_fig2_wigner.npz'))
    vm = max(abs(z['Wa']).max(), abs(z['Wb']).max())
    for i, (Wm, t) in enumerate([(z['Wb'], r'$\omega_p=\omega_p^*$'), (z['Wa'], r'$\omega_p=2\omega$')]):
        ins = ax.inset_axes([0.47 + 0.27 * i, 0.04, 0.25, 0.36])
        ins.pcolormesh(z['x'], z['x'], Wm, cmap='RdBu_r', vmin=-vm, vmax=vm, shading='auto', rasterized=True)
        ins.set_aspect('equal'); ins.set_xticks([]); ins.set_yticks([])
        ins.set_title(t, fontsize=6, pad=1.5)
    bx = fig.add_subplot(gs[1])
    filas = list(csv.reader(open(os.path.join(C.DATA, 'principal_fig2.csv'), encoding='utf8')))[1:]
    S = {}
    for r in filas:
        S.setdefault(r[0], []).append((float(r[10]), float(r[11]), float(r[6]), float(r[2])))
    col = {4: E.OKABE[0], 2: E.OKABE[1]}
    for nom, L in sorted(S.items()):
        L = sorted(L); x = np.array([a[0] for a in L]); p = np.array([a[1] for a in L])
        a2 = int(round(L[0][2])); xx = np.linspace(x[0], x[-1], 400); yy = PchipInterpolator(x, p)(xx)
        if nom.startswith('Ma'):
            bx.plot(xx, yy, color='k', lw=1.0, ls=(0, (1, 1)), zorder=5); continue
        if L[0][3] == 1000:
            bx.plot(x, p, 'D', ms=2.3, color=col[a2], alpha=0.8)
        bx.plot(xx, yy, color=col[a2], lw=0.8, alpha=0.8)
    bx.axvline(0, color='0.4', lw=0.6)
    bx.set_xlim(-6, 6); bx.set_ylim(0, 1.03)
    bx.set_xlabel(r'$x=(\omega_p-\omega_p^*)/G$'); bx.set_ylabel(r'$P_c$')
    rotulo(bx, '(b)')
    hs = [Line2D([], [], color=E.OKABE[0], lw=1.6, label=r'$|\alpha|^2=4$'),
          Line2D([], [], color=E.OKABE[1], lw=1.6, label=r'$|\alpha|^2=2$'),
          Line2D([], [], color='k', ls=(0, (1, 1)), label=r'Ma, $\omega_q$ fijo'),
          Line2D([], [], color='0.4', marker='D', ls='none', ms=3, label=r'Naseem ($\omega/\kappa=1000$)')]
    bx.legend(handles=hs, loc='upper center', fontsize=6.5, bbox_to_anchor=(0.5, -0.19), ncol=4, columnspacing=1.0, handlelength=1.8)
    guardar(fig, 'fig_resonancia')


# ---------------------------------------------------------------- mérito
def merito():
    import principal_fig2 as PF
    T = PF.cargar(); ok = T[:, 8] > 0.99
    E.aplicar()
    fig = plt.figure(figsize=(E.COL2, 2.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 0.045, 0.5, 1], wspace=0.08)
    ax, cax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[3])
    norm = LogNorm(4e-3, 3e-2)
    mk = lambda wk: 'o' if wk < 180 else ('s' if wk < 220 else ('D' if wk < 300 else '^'))
    for i, r in enumerate(T):
        if ok[i]:
            ax.scatter(r[6], r[11] / (5 / 72), c=[r[5]], norm=norm, cmap='viridis', marker=mk(r[4]), s=24,
                       edgecolors='k', linewidths=0.4, zorder=3)
    ax.axhline(1, color='k', lw=0.8); ax.set_xscale('log'); ax.set_ylim(0.965, 1.005)
    ax.set_xlabel(r'$\kappa_2/\kappa$')
    ax.set_ylabel(r'$(\kappa_1/\kappa_2)(g_z/\kappa)^2\,/\,(5/72)$')
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=cax)
    cb.ax.set_title(r'$g_x/\omega$', fontsize=7, pad=4); cb.set_ticks([4e-3, 1e-2, 3e-2]); cb.set_ticklabels(['0.004', '0.01', '0.03']); cb.ax.minorticks_off()
    hs = [Line2D([], [], marker=m, ls='none', mfc='0.7', mec='k', ms=4, label=l) for m, l in
          [('o', r'$\omega/\kappa<180$'), ('s', r'$\omega/\kappa=200$'), ('D', r'$\omega/\kappa=233$--$267$'), ('^', r'$\omega/\kappa=500$')]]
    ax.legend(handles=hs, fontsize=6.3, loc='lower left', ncol=2, columnspacing=0.8, handletextpad=0.2)
    rotulo(ax, '(a)', y=0.965, x=0.02)
    B = np.genfromtxt(os.path.join(C.DATA, 'fig3b.csv'), delimiter=',', skip_header=1)
    a2 = B[:, 3]
    bx.plot(a2, B[:, 13], 'o-', color=E.OKABE[0], ms=4, label=r'con el $+1$ (Ec. de $\gamma_{\rm pf}$)')
    bx.plot(a2, B[:, 14], 's--', color=E.OKABE[1], ms=4, label=r'sin el $+1$: $2|\alpha|^2\kappa_1$')
    xx = np.linspace(1.6, 6.4, 100)
    bx.plot(xx, 1 + 1 / (10 * xx), ':', color=E.OKABE[1], lw=0.9)
    bx.axhline(1, color='k', lw=0.6)
    bx.set_xlabel(r'$|\alpha|^2$'); bx.set_ylabel(r'$\gamma_{\rm pf}$ medido / predicho')
    bx.set_ylim(0.985, 1.065); bx.legend(fontsize=6.3, loc='upper right')
    rotulo(bx, '(b)', y=0.965, x=0.03)
    guardar(fig, 'fig_merito')


# ---------------------------------------------------------------- filtro
def filtro():
    import calc_fig4 as F4
    Aa = np.genfromtxt(os.path.join(C.DATA, 'fig4a.csv'), delimiter=',', skip_header=1)
    Bb = np.genfromtxt(os.path.join(C.DATA, 'fig4b.csv'), delimiter=',', skip_header=1)
    E.aplicar()
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(E.COL1, 4.7), gridspec_kw=dict(hspace=0.36))
    kf = np.geomspace(0.05, 40, 200)
    gx, w = F4.GX, F4.W
    Gm = gx**2 * KAP * kf**2 / (4 * w**2 + kf**2) / w**2
    Gp = gx**2 * KAP * kf**2 / (36 * w**2 + kf**2) / (9 * w**2)
    ax.loglog(kf / KAP, Gm / KAP, color=E.OKABE[0], lw=0.9)
    ax.loglog(kf / KAP, Gp / KAP, color=E.OKABE[1], lw=0.9)
    m = Aa[:, 0] > 0
    ax.loglog(Aa[m, 0] / KAP, Aa[m, 1] / KAP, 'o', color=E.OKABE[0], ms=4, label=r'$\Gamma_1^-$ (simulación)')
    ax.loglog(Aa[m, 0] / KAP, Aa[m, 2] / KAP, 's', color=E.OKABE[1], ms=4, label=r'$\Gamma_1^+$ (simulación)')
    ax.axhline(Aa[~m, 1][0] / KAP, color=E.OKABE[0], ls=':', lw=0.8)
    ax.axhline(Aa[~m, 2][0] / KAP, color=E.OKABE[1], ls=':', lw=0.8)
    ax.text(3, Aa[~m, 1][0] / KAP * 0.4, 'baño plano', fontsize=6.5, color=E.OKABE[0])
    ax.set_xlabel(r'$\kappa_f/\kappa$'); ax.set_ylabel(r'tasa $/\kappa$')
    ax.legend(loc='lower right', fontsize=6.3); rotulo(ax, '(a)')
    mb = Bb[:, 0] > 0
    a0 = Bb[0, 3]
    bx.loglog(Bb[mb, 0] / KAP, Bb[mb, 5], 'o', color=E.OKABE[0], ms=4, label='mejora medida')
    bx.loglog(Bb[mb, 0] / KAP, Bb[mb, 6], 'x', color=E.OKABE[0], ms=5, label='mejora predicha')
    Gm0, Gp0 = gx**2 * KAP / w**2, gx**2 * KAP / (9 * w**2)
    bx.loglog(kf / KAP, (Gm0 * a0 + Gp0 * (a0 + 1)) / (Gm * a0 + Gp * (a0 + 1)), color=E.OKABE[0], lw=0.8)
    bx.set_xlabel(r'$\kappa_f/\kappa$'); bx.set_ylabel(r'$\gamma_{\rm pf}^{\rm plano}/\gamma_{\rm pf}$', color=E.OKABE[0])
    cx = bx.twinx()
    cx.semilogx(Bb[mb, 0] / KAP, Bb[mb, 8], 'D--', color=E.OKABE[2], ms=3.5, lw=0.8)
    cx.set_ylabel('confinamiento / plano', color=E.OKABE[2]); cx.set_ylim(0, 1.2)
    cx.tick_params(axis='y', colors=E.OKABE[2])
    bx.legend(loc='lower left', fontsize=6.3); rotulo(bx, '(b)', x=0.86)
    guardar(fig, 'fig_filtro')


# ---------------------------------------------------------------- chi
def chi():
    T = np.genfromtxt(os.path.join(C.DATA, 'p9_diagnostico.csv'), delimiter=',', skip_header=1)
    E.aplicar()
    fig, ax = plt.subplots(figsize=(E.COL1, 2.5))
    ax.semilogx(T[:, 6], T[:, 7], 'o', color=E.OKABE[0], ms=4, label='modelo completo')
    ax.semilogx(T[:, 6], T[:, 8], 'x', color=E.OKABE[1], ms=5, label=r'efectivo con $\chi\,n\,|e\rangle\langle e|$')
    ax.semilogx(T[:, 6], T[:, 9], '+', color=E.OKABE[2], ms=6, label=r'efectivo sin $\chi$')
    for r in T:
        if r[4] > 300:
            ax.annotate(rf"$\omega/\kappa={r[4]:.0f}$", (r[6], r[7]), xytext=(3, -9), textcoords='offset points', fontsize=5.5)
    ax.axhline(1, color='k', lw=0.6); ax.axvline(0.3, color='0.5', ls='--', lw=0.7)
    ax.text(0.31, 0.5, r'$\chi|\alpha|^2=0.3\,\kappa$', rotation=90, fontsize=6, color='0.4', va='center', ha='left')
    ax.set_xlabel(r'$\chi|\alpha|^2/\kappa$'); ax.set_ylabel(r'confinamiento / $\Delta_{\rm min}$')
    ax.legend(fontsize=6.3, loc='lower left')
    guardar(fig, 'fig_chi')


# ---------------------------------------------------------------- mapa
def mapa():
    import figura_central as FC
    KFW, WK, AL2, KMAX_F, EPS_MIN = FC.KFW, FC.WK, FC.AL2, 1.5, 1e-5
    Dp = FC.delta('minimo', 'k*_N24.npz')
    Df = FC.delta('minimo_filtro', 'k*_kf10_N20_Nf3.npz', True)
    Df = Df[(Df[:, 0] <= KMAX_F) & np.isfinite(Df[:, 1])]
    c = np.polyval(np.polyfit(Dp[:3, 0], Dp[:3, 1] / Dp[:3, 0], 1), 0.0)
    ip = lambda T, x: np.exp(np.interp(np.log(x), np.log(T[:, 0]), np.log(T[:, 1])))
    gz = np.geomspace(0.05, 100, 400); kk = np.geomspace(1e-3, 10, 300)
    GZ, KK = np.meshgrid(gz, kk); gxw2 = KK / (16 * GZ**2)
    f_filt = KFW**2 / (4 + KFW**2) + KFW**2 / (36 + KFW**2) / 9
    eps_a = (10 / 9) * gxw2 / (ip(Dp, KK) / c)
    eps_b = f_filt * gxw2 / (ip(Df, KK) / c); eps_b[KK > KMAX_F] = np.nan
    chi_mala = AL2 * (8 / 3) * gxw2 * WK > 0.3
    E.aplicar()
    fig = plt.figure(figsize=(E.COL2, 3.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.045], height_ratios=[1, 0.16], wspace=0.1, hspace=0.42)
    norm = LogNorm(EPS_MIN, 10); lev = np.geomspace(EPS_MIN, 10, 25)
    for i, (e, lab) in enumerate([(eps_a, '(a) baño plano'), (eps_b, r'(b) baño filtrado, $\kappa_f/\omega=0.05$')]):
        ax = fig.add_subplot(gs[0, i])
        ax.contourf(GZ, KK, np.clip(e, EPS_MIN * 1.01, 9.9), levels=lev, norm=norm, cmap='cividis')
        ax.contour(GZ, KK, e, levels=[1 / 220], colors='w', linewidths=1.0)
        if i == 1:
            ax.contour(GZ, KK, e, levels=[1e-3], colors='w', linewidths=0.8, linestyles='--')
            for gk, col in [(2e-4, E.OKABE[1]), (2e-5, E.OKABE[5])]:
                ep = e + gk / (ip(Df, KK) / c)
                ax.contour(GZ, KK, ep, levels=[1 / 220], colors=[col], linewidths=1.2)
                ax.contour(GZ, KK, ep, levels=[1e-3], colors=[col], linewidths=1.2, linestyles='--')
            ax.axhspan(KMAX_F, 10, color='0.75', zorder=0)
            ax.text(0.9, 2.2, 'retorno no monótono', fontsize=6, ha='center', va='center', color='0.2')
        ax.contourf(GZ, KK, chi_mala.astype(float), levels=[0.5, 1.5], colors=['w'], alpha=0.3, zorder=4)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-3, 10)
        ax.set_xlabel(r'$g_z/\kappa$')
        if i == 0:
            ax.set_ylabel(r'$\kappa_2/\kappa$')
        else:
            ax.set_yticklabels([])
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, va='top', fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.2', fc='w', ec='none', alpha=0.85))
        if i == 0:
            P = np.genfromtxt(os.path.join(C.DATA, 'principal_fig3_puntos.csv'), delimiter=',', skip_header=1)
            ax.scatter(P[:, 2], P[:, 3], c=P[:, 9], norm=norm, cmap='cividis', s=20, edgecolors='w', linewidths=0.9, zorder=6)
            ax.plot([7.07], [1.0], '*', color='w', mec='k', ms=9, zorder=7); ax.text(7.9, 1.15, 'Ma', color='w', fontsize=7, zorder=7)
            ax.plot([60], [2.07], 'P', color='w', mec='k', ms=7, zorder=7); ax.text(27, 3.0, 'Naseem', color='w', fontsize=7, zorder=7)
            ax.text(0.05, 0.06, r'zona clara: $\chi|\alpha|^2>0.3\,\kappa$', transform=ax.transAxes, fontsize=6, color='k',
                    bbox=dict(boxstyle='round,pad=0.15', fc='w', ec='none', alpha=0.7))
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='cividis'), cax=cax); cb.ax.set_title(r'$\epsilon$', fontsize=8, pad=3)
    cb.ax.axhline(1 / 220, color='w', lw=1.0)
    lax = fig.add_subplot(gs[1, :2]); lax.axis('off')
    hs = [Line2D([], [], color='k', lw=1.2, label=r'$\epsilon=1/220$ (continua) y $10^{-3}$ (discontinua)'),
          Line2D([], [], color='w', marker='o', mfc='0.55', mec='0.3', ls='none', ms=5, label='modelo completo, (a): color $=\\epsilon$ medido'),
          Line2D([], [], color='0.5', lw=1.0, label='sin piso (blanco)'),
          Line2D([], [], color=E.OKABE[5], lw=1.3, label=r'con piso $\gamma/\kappa=2\times10^{-5}$'),
          Line2D([], [], color=E.OKABE[1], lw=1.3, label=r'con piso $\gamma/\kappa=2\times10^{-4}$')]
    lax.legend(handles=hs, loc='center', ncol=2, fontsize=6.6, frameon=False)
    guardar(fig, 'fig_mapa')


if __name__ == '__main__':
    todo = dict(resonancia=resonancia, merito=merito, filtro=filtro, chi=chi, mapa=mapa)
    for k in (sys.argv[1:] or todo):
        todo[k]()
