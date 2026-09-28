"""Fig. 4 — baño filtrado. Solo lee la caché data/fig4/*.npz (calc_fig4.py / run_fig4.sh), escribe
data/fig4a.csv, data/fig4b.csv y dibuja fig4.pdf / fig4.png.  --rerun: calcula lo que falte (caro).

(a) Γ₁⁻ y Γ₁⁺ del método estático (sin drive, g_z = 0) frente a κ_f, con g_x²κ_eff(ω)/ω² y g_x²κ_eff(3ω)/(9ω²).
(b) mejora de γ_pf (C7) respecto al baño plano, medida (espectral) y predicha con C1 (α_eff² de cada punto);
    eje secundario: tasa de confinamiento relativa al baño plano (cualitativa, C9).
Tasa de confinamiento (C9, dinámica): retorno de P_c(t) al estacionario desde |0>|g> y D(d)|1.3α>|g>
(evolución con el propagador, t = nT_p hasta t = 6000), ajuste exponencial de la cola.
"""
import sys, glob, os
import numpy as np
import matplotlib.pyplot as plt
import comun as C
import estilo as E
import calc_fig4 as F4

KFS = [0.1, 0.3, 1.0, 3.0]


def confinamiento(z, ret=False):
    """Tasa de confinamiento DINÁMICA (C9): retorno de P_c(t) (código fijo, t = nT_p) al estacionario desde
    |0>|g> y D(d)|1.3α>|g>. Ajuste lineal de ln|P_c^ss - P_c(t)| en la cola: desde que el exceso cae por debajo
    del 10% de su máximo hasta que llega a 3× el piso tardío (mediana del último 20%, deriva lenta de paridad). Devuelve la media de los dos estados."""
    t = z['tdin']; pss = float(z['Pc_fijo']); tasas = []
    for serie in z['Pc_din']:
        e = np.abs(pss - np.asarray(serie))
        i1 = np.argmax(e < 0.1 * e.max())
        piso = np.median(e[int(0.8 * len(e)):])          # deriva lenta (paridad) al final
        fin = np.where(e < 3 * piso)[0]; i2 = fin[fin > i1][0] if np.any(fin > i1) else len(e)
        if i2 - i1 < 10:
            tasas.append(np.nan); continue
        tasas.append(-np.polyfit(t[i1:i2], np.log(e[i1:i2]), 1)[0])
    return tasas if ret else np.nanmean(tasas)


def main():
    if '--rerun' in sys.argv:
        for kf in [0.0] + KFS:
            F4.estatico(kf, 8, 3 if kf else 1)
            F4.floquet(kf, 16, 2 if kf else 1, 2.0)
    est = {}
    for f in glob.glob(os.path.join(C.DATA, 'fig4', 'est_*.npz')):
        z = np.load(f); est[(float(z['kf']), int(z['N']), int(z['Nf']))] = z
    fl = {}
    for f in glob.glob(os.path.join(C.DATA, 'fig4', 'fl_*.npz')):
        z = np.load(f); fl[(float(z['kf']), int(z['N']))] = z
    # ---- (a)
    filas = []
    for kf in [0.0] + KFS:
        z = est[(kf, 8, 3 if kf else 1)]
        filas.append([kf, float(z['Gm']), float(z['Gp']), float(z['Gm_pred']), float(z['Gp_pred']), *np.array(z['val'])])
    Aa = np.array(filas)
    np.savetxt(os.path.join(C.DATA, 'fig4a.csv'), Aa, delimiter=',', comments='',
               header='kappa_f [2pi GHz] (0 = flat bath), Gamma_1^- static [2pi GHz], Gamma_1^+ static [2pi GHz], '
                      'Gamma_1^- pred g_x^2 kappa_eff(w)/w^2, Gamma_1^+ pred g_x^2 kappa_eff(3w)/(9w^2), |Tr rho-1|, ||rho-rho^dag||, min eig')
    # ---- (b)
    z0 = fl[(0.0, 16)]
    g0 = float(z0['gpf']); c0 = confinamiento(z0)
    a0 = abs(complex(z0['al2eff']))
    p0 = 2 * (float(z0['Gm_pred']) * a0 + float(z0['Gp_pred']) * (a0 + 1))
    filas = []
    for kf in [0.0] + KFS:
        if (kf, 16) not in fl:
            continue
        z = fl[(kf, 16)]
        a2 = abs(complex(z['al2eff'])); g = float(z['gpf'])
        pred = 2 * (float(z['Gm_pred']) * a2 + float(z['Gp_pred']) * (a2 + 1))
        cf = confinamiento(z); cfs = confinamiento(z, True)
        filas.append([kf, g, pred, a2, float(z['Pc_fijo']), g0 / g, p0 / pred, cf, cf / c0, cfs[0], cfs[1], *np.array(z['val'])])
    Bb = np.array(filas)
    np.savetxt(os.path.join(C.DATA, 'fig4b.csv'), Bb, delimiter=',', comments='',
               header='kappa_f [2pi GHz] (0 = flat), gamma_pf spectral [2pi GHz], gamma_pf pred C1 [2pi GHz], |alpha_eff^2|, '
                      'P_c fixed polaron code, improvement measured gamma_flat/gamma, improvement predicted (C1), '
                      'confinement rate (dynamic, mean) [2pi GHz], confinement rate / flat, rate from |0>|g>, rate from D(d)|1.3 alpha>|g>, |Tr rho-1|, ||rho-rho^dag||, min eig')

    E.aplicar()
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(E.COL1, 4.6), gridspec_kw=dict(hspace=0.35))
    kk = np.geomspace(0.05, 40, 200)
    gx, w = F4.GX, F4.W
    ax.loglog(kk, gx**2 * F4.KAP * kk**2 / (4 * w**2 + kk**2) / w**2, color=E.OKABE[0], lw=0.9)
    ax.loglog(kk, gx**2 * F4.KAP * kk**2 / (36 * w**2 + kk**2) / (9 * w**2), color=E.OKABE[1], lw=0.9)
    m = Aa[:, 0] > 0
    ax.loglog(Aa[m, 0], Aa[m, 1], 'o', color=E.OKABE[0], ms=4, label=r'$\Gamma_1^-$')
    ax.loglog(Aa[m, 0], Aa[m, 2], 's', color=E.OKABE[1], ms=4, label=r'$\Gamma_1^+$')
    ax.axhline(Aa[~m, 1][0], color=E.OKABE[0], ls=':', lw=0.8)
    ax.axhline(Aa[~m, 2][0], color=E.OKABE[1], ls=':', lw=0.8)
    ax.text(0.06, Aa[~m, 1][0] * 1.4, 'flat bath', fontsize=6.5, color=E.OKABE[0])
    ax.set_xlabel(r'$\kappa_f/2\pi$ (GHz)'); ax.set_ylabel(r'rate$/2\pi$ (GHz)')
    ax.legend(loc='lower right'); E.etiqueta(ax, '(a)')
    mb = Bb[:, 0] > 0
    bx.loglog(Bb[mb, 0], Bb[mb, 5], 'o', color=E.OKABE[0], ms=4, label=r'$\gamma_{\rm pf}$ improvement, measured')
    bx.loglog(Bb[mb, 0], Bb[mb, 6], 'x', color=E.OKABE[0], ms=5, label='predicted (C1)')
    # predicción continua con |α|² del plano
    Gm = lambda k: gx**2 * F4.KAP * k**2 / (4 * w**2 + k**2) / w**2
    Gp = lambda k: gx**2 * F4.KAP * k**2 / (36 * w**2 + k**2) / (9 * w**2)
    bx.loglog(kk, p0 / (2 * (Gm(kk) * a0 + Gp(kk) * (a0 + 1))), color=E.OKABE[0], lw=0.8)
    bx.set_xlabel(r'$\kappa_f/2\pi$ (GHz)'); bx.set_ylabel(r'$\gamma_{\rm pf}^{\rm flat}/\gamma_{\rm pf}$', color=E.OKABE[0])
    cx = bx.twinx()
    cx.semilogx(Bb[mb, 0], Bb[mb, 8], 'D--', color=E.OKABE[2], ms=3.5, lw=0.8)
    cx.set_ylabel('confinement rate / flat', color=E.OKABE[2]); cx.set_ylim(0, 1.2)
    cx.tick_params(axis='y', colors=E.OKABE[2])
    bx.legend(loc='upper right', fontsize=6.3); E.etiqueta(bx, '(b)'); bx.texts[-1].set_position((0.03, 0.2))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(C.AQUI, f'fig4.{ext}'))
    print("(a) κ_f Γ⁻ razón Γ⁺ razón")
    for r in Aa:
        print(f"   {r[0]:4.1f} {r[1]:.4e} {r[1]/r[3]:.4f} {r[2]:.4e} {r[2]/r[4]:.4f}")
    print("(b) κ_f γ_pf pred razón |α_eff²| P_c mejora_med mejora_pred conf conf/plano")
    for r in Bb:
        print(f"   {r[0]:4.1f} {r[1]:.4e} {r[2]:.4e} {r[1]/r[2]:.4f} {r[3]:.4f} {r[4]:.5f} {r[5]:.1f} {r[6]:.1f} {r[7]:.3e} {r[8]:.3f}")
    if (1.0, 20) in fl:
        z = fl[(1.0, 20)]; z1 = fl[(1.0, 16)]
        print(f"conv κ_f=1 N=16→20: γ_pf {float(z1['gpf']):.6e}→{float(z['gpf']):.6e}; conf {confinamiento(z1, True)}→{confinamiento(z, True)}")
    for n in (16, 22, 28):
        if (0.0, n) in fl:
            z = fl[(0.0, n)]; print(f"plano N={n}: γ_pf={float(z['gpf']):.6e} conf={confinamiento(z, True)} 5º modo espectral={z['modos'][4][2]:.4e} Im={z['modos'][4][1]:+.3e}")
    for k in [(1.0, 8, 2), (1.0, 8, 3), (1.0, 12, 3), (0.1, 8, 2), (0.1, 8, 3), (0.1, 12, 3)]:
        if k in est:
            print(f"conv est {k}: Γ⁻={float(est[k]['Gm']):.6e} Γ⁺={float(est[k]['Gp']):.6e}")


if __name__ == '__main__':
    main()
