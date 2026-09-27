"""Fig. 2 — resonancia vestida (modelo completo de Ma). Solo lee la caché data/fig2/*.npz (generada por
calc_fig2.py / run_fig2.sh), escribe los CSV data/fig2a.csv y data/fig2b.csv y dibuja fig2.pdf / fig2.png.
Para cambiar el estilo basta con volver a correr este script (no recalcula nada).
  --rerun : recalcula los puntos que falten llamando a calc_fig2.punto (caro).
"""
import sys, glob, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
import comun as C
import estilo as E

W, GZ, KAP = 6.0, 0.3 * np.cos(np.pi / 4), 0.03
GXMA = -0.3 * np.sin(np.pi / 4)


def cargar():
    filas = []
    for f in sorted(glob.glob(os.path.join(C.DATA, 'fig2', '*.npz'))):
        z = np.load(f)
        r = {k: z[k] for k in z.files if k != 'rho'}
        # código fijo: D(gz/w)|±α_nom> (no se adapta al estado; necesario para anchos de resonancia)
        al = np.sqrt(complex(float(z['Om']) / float(z['G'])))
        r['Pc_fijo'] = C.Pc(z['rho'], int(z['N']), al, float(z['d']))
        filas.append(r)
    return filas


def fwhm(wp, pc):
    o = np.argsort(wp); wp, pc = wp[o], pc[o]
    cs = CubicSpline(wp, pc)
    x = np.linspace(wp[0], wp[-1], 20001); y = cs(x)
    k = np.argmax(y); pmax, wmax = y[k], x[k]
    h = pmax / 2
    izq = [i for i in range(k) if (y[i] - h) * (y[i + 1] - h) <= 0]
    der = [i for i in range(k, len(x) - 1) if (y[i] - h) * (y[i + 1] - h) <= 0]
    if not izq or not der:
        return wmax, pmax, np.nan, np.nan
    wl = brentq(lambda t: cs(t) - h, x[izq[-1]], x[izq[-1] + 1])
    wr = brentq(lambda t: cs(t) - h, x[der[0]], x[der[0] + 1])
    return wmax, pmax, wl, wr


def main():
    if '--rerun' in sys.argv:
        import calc_fig2
        for l in open(os.path.join(C.DATA, 'fig2', 'trabajos.txt')):
            gx, wp, wq, Om, N = map(float, l.split())
            calc_fig2.punto(gx, wp, wq, Om, int(N))
    F = cargar()
    # ---- (a) Ma, wq = 12, N = 20
    A = sorted([f for f in F if abs(f['gx'] - GXMA) < 1e-6 and float(f['wq']) == 12.0 and int(f['N']) == 20],
               key=lambda f: float(f['wp']))
    wpa = np.array([float(f['wp']) for f in A])
    pcn = np.array([float(f['Pc_nuevo']) for f in A]); pcv = np.array([float(f['Pc_viejo']) for f in A])
    pcf = np.array([float(f['Pc_fijo']) for f in A])
    pe = np.array([float(f['Pe']) for f in A]); al2 = np.array([complex(f['al2eff']) for f in A])
    np.savetxt(os.path.join(C.DATA, 'fig2a.csv'), np.c_[wpa, pcn, pcv, pcf, pe, al2.real, al2.imag], delimiter=',',
               header='omega_p [2pi GHz], P_c polaron code {D(gz/w)|+-alpha_eff>}, P_c old code {|+-alpha_nom>}, P_c fixed polaron code {D(gz/w)|+-alpha_nom>}, '
                      'P_e (stroboscopic t=nT_p), Re alpha_eff^2, Im alpha_eff^2', comments='')
    wpred = 2 * (W - 4 * GXMA**2 / (3 * W))
    # ---- (b) wq = wp, cinco κ₂/κ
    series = {}
    for f in F:
        if abs(float(f['wq']) - float(f['wp'])) < 1e-9 and int(f['N']) == 20:
            series.setdefault(round(float(f['gx']), 6), []).append(f)
    Bf, crudo = [], []
    for gx, L in sorted(series.items(), key=lambda t: -t[0]):
        wp = np.array([float(f['wp']) for f in L]); pc = np.array([float(f['Pc_fijo']) for f in L])   # código fijo (ver README)
        G = abs(2 * gx * GZ / W); k2 = 4 * G**2 / KAP
        wmax, pmax, wl, wr = fwhm(wp, pc)
        w0 = 2 * (W - 4 * gx**2 / (3 * W))
        crudo += [[gx, G, k2 / KAP, a, (a - w0) / G, b] for a, b in sorted(zip(wp, pc))]
        Bf.append([gx, G, k2 / KAP, wmax, pmax, wl, wr, wr - wl, (wr - wl) / G, len(L)])
    Bf = np.array(Bf)
    np.savetxt(os.path.join(C.DATA, 'fig2b.csv'), Bf, delimiter=',',
               header='g_x [2pi GHz], G=|2 g_x g_z/w| [2pi GHz], kappa_2/kappa, omega_p at max, P_c max, '
                      'omega_p left half-max, omega_p right half-max, FWHM [2pi GHz], c=FWHM/G, n_points', comments='')
    np.savetxt(os.path.join(C.DATA, 'fig2b_curvas.csv'), np.array(crudo), delimiter=',',
               header='g_x [2pi GHz], G [2pi GHz], kappa_2/kappa, omega_p = omega_q [2pi GHz], (omega_p - omega_p*)/G, '
                      'P_c fixed polaron code {D(gz/w)|+-alpha_nom>}', comments='')
    ok = np.isfinite(Bf[:, 7])
    c_fit = np.sum(Bf[ok, 7] * Bf[ok, 1]) / np.sum(Bf[ok, 1]**2)

    # ---- dibujo
    E.aplicar()
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(E.COL1, 4.4), gridspec_kw=dict(hspace=0.38))
    x = np.linspace(wpa[0], wpa[-1], 800)
    xs99 = x[CubicSpline(wpa, pcn)(x) > 0.99]
    ax.axvspan(xs99[0], xs99[-1], color=E.OKABE[2], alpha=0.15, lw=0)
    ax.plot(x, CubicSpline(wpa, pcn)(x), color=E.OKABE[0], label=r'polaron code, $\alpha_{\rm eff}$')
    ax.plot(wpa, pcn, 'o', ms=2.5, color=E.OKABE[0])
    ax.plot(x, CubicSpline(wpa, pcv)(x), '--', color=E.OKABE[1], label=r'bare code, nominal $\alpha$')
    ax.plot(x, CubicSpline(wpa, pcf)(x), ':', color=E.OKABE[3], label=r'polaron code, nominal $\alpha$')
    ax.axvline(wpred, color='k', lw=0.7, ls=':')
    ax.axvline(2 * W, color='0.5', lw=0.7, ls='-.')
    ax.text(wpred, 0.02, r' $2(\omega-4g_x^2/3\omega)$', fontsize=7, rotation=90, va='bottom', ha='right')
    ax.text(2 * W, 0.55, r' $2\omega$', fontsize=7, rotation=90, va='bottom', ha='right', color='0.4')
    ax.set_xlabel(r'$\omega_p/2\pi$ (GHz)'); ax.set_ylabel(r'$P_c$')
    ax.set_ylim(0, 1.02); ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.3, handlelength=1.8, columnspacing=1.0)
    E.etiqueta(ax, '(a)')
    ins = ax.inset_axes([0.64, 0.1, 0.33, 0.36])
    m = (wpa > 11.972) & (wpa < 11.99)
    ins.plot(wpa[m], 1 - pcn[m], 'o-', ms=2, color=E.OKABE[0]); ins.plot(wpa[m], 1 - pcv[m], 's--', ms=2, color=E.OKABE[1]); ins.plot(wpa[m], 1 - pcf[m], '^:', ms=2, color=E.OKABE[3])
    ins.set_yscale('log'); ins.axvline(wpred, color='k', lw=0.6, ls=':')
    ins.set_ylabel(r'$1-P_c$', fontsize=6.5, labelpad=1); ins.tick_params(labelsize=5.5)
    ins.set_xticks([11.975, 11.985])
    for i, (G, k2, fw, cc) in enumerate(Bf[:, [1, 2, 7, 8]]):
        bx.loglog(G, fw, 'o', color=E.OKABE[i % 7], ms=4, label=rf'$\kappa_2/\kappa={k2:.2g}$')
    gg = np.geomspace(Bf[:, 1].min() / 1.5, Bf[:, 1].max() * 1.5, 50)
    bx.loglog(gg, c_fit * gg, 'k-', lw=0.8, label=rf'$c\,G$, $c={c_fit:.2f}$')
    from matplotlib.ticker import NullFormatter, FixedLocator
    bx.xaxis.set_minor_formatter(NullFormatter()); bx.yaxis.set_minor_formatter(NullFormatter())
    bx.xaxis.set_major_locator(FixedLocator([2e-3, 5e-3, 1e-2, 2e-2])); bx.set_xticklabels(['0.002', '0.005', '0.01', '0.02'])
    bx.set_xlabel(r'$G/2\pi$ (GHz)'); bx.set_ylabel(r'FWHM$/2\pi$ (GHz)')
    bx.legend(loc='lower right', fontsize=6.5)
    E.etiqueta(bx, '(b)')
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(C.AQUI, f'fig2.{ext}'))
    # resumen en consola
    ka = np.argmax(pcn)
    print(f"    fijo (polarón, α nominal): máx {pcf.max():.6f} en wp={wpa[np.argmax(pcf)]}; en 11.98 {np.interp(11.98, wpa, pcf):.6f}")
    print(f"(a) máx P_c nuevo = {pcn.max():.6f} en wp={wpa[ka]}; viejo máx = {pcv.max():.6f} en wp={wpa[np.argmax(pcv)]}; pred {wpred:.5f}")
    xs = x[CubicSpline(wpa, pcn)(x) > 0.99]
    print(f"    ventana P_c>0.99 (nuevo): [{xs[0]:.5f}, {xs[-1]:.5f}]; en 11.98: nuevo {np.interp(11.98, wpa, pcn):.6f} viejo {np.interp(11.98, wpa, pcv):.6f}")
    print("(b) κ₂/κ, G, P_max, FWHM, c:")
    for r in Bf:
        print(f"    {r[2]:.3f}  {r[1]:.4e}  {r[4]:.5f}  {r[7]:.4e}  {r[8]:.3f}")
    print(f"    c (ajuste por el origen) = {c_fit:.3f}; media {np.nanmean(Bf[:, 8]):.3f} ± {np.nanstd(Bf[:, 8]):.3f}")


if __name__ == '__main__':
    main()
