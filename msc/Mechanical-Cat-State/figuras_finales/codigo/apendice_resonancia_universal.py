"""Figura principal 2 — regla de operación universal: P_c (código fijo polarónico) frente a x = (ω_p − ω_p*)/G,
ω_p* = 2(ω − 4g_x²/3ω), G = |2g_xg_z/ω|, para sistemas distintos (ω_q = ω_p salvo Ma).
Solo lee cachés (no propaga):
  data/fig2/*_N20.npz con ω_q = ω_p  -> serie 1 (ω = 6, g_z = 0.2121, κ₂/κ ∈ {0.03, 0.1, 0.3, 1, 2}, |α|² = 4)
                                        P_e promediado en data/fig2/pe_*.npz (calc_pe_fig2viejas.py)
  data/pfig2/*.npz                   -> series 2–5 (calc_principal_fig2.py / run_principal_fig2.sh)
  data/fig2/*wq12*_N22.npz           -> Ma con ω_q = 12 fijo (línea distinta)
Recuadro: Wigner de Ma en ω_p = 2ω y ω_p* (de apendice_fig2_estados / data/principal_fig2_wigner.npz).
Escribe data/principal_fig2.csv (todas las curvas) y data/principal_fig2_resumen.csv (máximo, FWHM, asimetría).
"""
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import comun as C
import estilo as E


def cargar():
    filas = []
    # serie 1 (caché de la Fig. 2 de validación)
    for f in glob.glob(os.path.join(C.DATA, 'fig2', 'gx*_N20.npz')):
        z = np.load(f)
        if abs(float(z['wq']) - float(z['wp'])) > 1e-9:
            continue
        fp = os.path.join(C.DATA, 'fig2', 'pe_' + os.path.basename(f))
        pe = float(np.load(fp)['Pe_prom']) if os.path.exists(fp) else np.nan
        w, gx, gz, kap = 6.0, float(z['gx']), 0.3 * np.cos(np.pi / 4), 0.03
        al = np.sqrt(complex(float(z['Om']) / float(z['G'])))
        filas.append(dict(w=w, gx=gx, gz=gz, kap=kap, Om=float(z['Om']), wq=float(z['wq']), wp=float(z['wp']), N=20,
                          Pc=C.Pc(z['rho'], 20, al, float(z['d'])), Pe=pe))
    for f in glob.glob(os.path.join(C.DATA, 'pfig2', '*.npz')):
        z = np.load(f)
        filas.append(dict(w=float(z['w']), gx=float(z['gx']), gz=float(z['gz']), kap=float(z['kap']), Om=float(z['Om']),
                          wq=float(z['wq']), wp=float(z['wp']), N=int(z['N']), Pc=float(z['Pc_fijo']), Pe=float(z['Pe_prom']),
                          val=z['val']))
    # Ma con ω_q fijo
    for f in glob.glob(os.path.join(C.DATA, 'fig2', 'gx*_wq12.000000_Om0.060000_N22.npz')):
        z = np.load(f)
        filas.append(dict(w=6.0, gx=float(z['gx']), gz=0.3 * np.cos(np.pi / 4), kap=0.03, Om=0.06, wq=12.0, wp=float(z['wp']), N=22,
                          Pc=C.Pc(z['rho'], 22, 2j, float(z['d'])), Pe=float(z['Pe_prom']), ma=True))
    return filas


def sistemas(filas):
    S = {}
    for r in filas:
        G = abs(2 * r['gx'] * r['gz'] / r['w']); al2 = r['Om'] / G
        k2 = 4 * G**2 / r['kap'] / r['kap']
        ma = r.get('ma', False)
        key = 'Ma (ω_q fixed)' if ma else f"w{r['w']:g}_gz{r['gz']/r['kap']:.3g}_k2{k2:.3g}_a{al2:.3g}"
        r.update(G=G, al2=al2, k2k=k2, wps=2 * (r['w'] - 4 * r['gx']**2 / (3 * r['w'])))
        r['x'] = (r['wp'] - r['wps']) / G
        S.setdefault(key, {})
        kw = round(r['wp'], 9)
        if kw not in S[key] or r['N'] > S[key][kw]['N']:
            S[key][kw] = r
    return {(k, max(r['N'] for r in d.values())): list(d.values()) for k, d in S.items()}


def analiza(x, p):
    o = np.argsort(x); x, p = np.asarray(x)[o], np.asarray(p)[o]
    if len(x) < 4 or not np.all(np.isfinite(p)):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    ip = PchipInterpolator(x, p)
    xx = np.linspace(x[0], x[-1], 40001); yy = ip(xx)
    k = np.argmax(yy); pmax, xmax = yy[k], xx[k]
    h = pmax / 2
    try:
        il = [i for i in range(k) if (yy[i] - h) * (yy[i + 1] - h) <= 0][-1]
        ir = [i for i in range(k, len(xx) - 1) if (yy[i] - h) * (yy[i + 1] - h) <= 0][0]
        xl = brentq(lambda t: ip(t) - h, xx[il], xx[il + 1]); xr = brentq(lambda t: ip(t) - h, xx[ir], xx[ir + 1])
    except IndexError:
        xl = xr = np.nan
    # máximo sub-rejilla: parábola por los puntos con |x| ≤ 0.6 (PCHIP solo puede tener el máximo en un nodo)
    m = np.abs(x) <= 0.61
    xpar = np.nan
    if m.sum() >= 3:
        c2, c1, c0 = np.polyfit(x[m], p[m], 2)
        xpar = -c1 / (2 * c2) if c2 < 0 else np.nan
    return xmax, pmax, xl, xr, xpar


def main():
    S = sistemas(cargar())
    filas_csv, resumen = [], []
    for (nom, N), L in sorted(S.items()):
        for r in L:
            filas_csv.append([nom, N, r['w'], r['gx'], r['gz'], r['kap'], r['al2'], r['k2k'], r['wq'], r['wp'], r['x'], r['Pc'], r['Pe']])
        xmax, pmax, xl, xr, xpar = analiza([r['x'] for r in L], [r['Pc'] for r in L])
        resumen.append([nom, N, L[0]['w'], L[0]['gz'] / L[0]['kap'], L[0]['k2k'], L[0]['al2'], len(L), xmax, pmax, xr - xl, xmax - xl, xr - xmax, xpar])
    import csv
    with open(os.path.join(C.DATA, 'principal_fig2.csv'), 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['system', 'N', 'omega [kappa-units or 2pi GHz]', 'g_x', 'g_z', 'kappa', '|alpha|^2', 'kappa_2/kappa', 'omega_q', 'omega_p',
                     'x=(omega_p-omega_p*)/G', 'P_c fixed polaron code', 'P_e period-averaged'])
        wr.writerows(filas_csv)
    with open(os.path.join(C.DATA, 'principal_fig2_resumen.csv'), 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['system', 'N', 'omega', 'g_z/kappa', 'kappa_2/kappa', '|alpha|^2', 'n_points', 'x at max', 'P_c max', 'FWHM in x',
                     'left half-width', 'right half-width', 'x at max (parabola |x|<=0.6)'])
        wr.writerows(resumen)
    print(f"{'sistema':38s} {'N':>3} {'n':>3} {'x_max':>7} {'P_max':>8} {'FWHM':>6} {'izq':>6} {'der':>6} {'x_par':>7}")
    for r in resumen:
        print(f"{r[0]:38s} {r[1]:3d} {r[6]:3d} {r[7]:+7.3f} {r[8]:8.5f} {r[9]:6.3f} {r[10]:6.3f} {r[11]:6.3f} {r[12]:+7.3f}")
    dibujar(S)
    return S, resumen


def dibujar(S):
    E.aplicar()
    fig, ax = plt.subplots(figsize=(E.COL2 * 0.78, 3.0))
    col = {4.0: E.OKABE[0], 2.0: E.OKABE[1]}
    for (nom, N), L in sorted(S.items()):
        L = sorted(L, key=lambda r: r['x'])
        x = np.array([r['x'] for r in L]); p = np.array([r['Pc'] for r in L])
        a2 = round(L[0]['al2']); r0 = L[0]
        xx = np.linspace(x[0], x[-1], 400); yy = PchipInterpolator(x, p)(xx)
        if nom.startswith('Ma'):
            ax.plot(xx, yy, color='k', lw=0.9, ls=(0, (1, 1)), zorder=5, label=r'Ma, $\omega_q$ fixed')
            continue
        if r0['w'] == 1000:
            ls, lab = '-', 'Naseem params.'
            ax.plot(x, p, 'D', ms=2.5, color=col[a2], alpha=0.8)
        elif r0['w'] == 6 and abs(r0['gz'] / r0['kap'] - 7.05) < 0.1:
            ls, lab = '-', r'$\omega/\kappa=200$, $g_z/\kappa=7.1$'
        elif r0['w'] == 6:
            ls, lab = '--', r'$\omega/\kappa=200$, $g_z/\kappa\in\{4,7,20\}$'
        else:
            ls, lab = '-.', r'$\omega/\kappa\in\{133,267\}$'
        ax.plot(xx, yy, color=col[a2], lw=0.8, ls=ls, alpha=0.85, label=lab)
    # leyenda sin duplicados + colores de |α|²
    from matplotlib.lines import Line2D
    g = '0.35'
    hs = [Line2D([], [], color=E.OKABE[0], lw=2, label=r'$|\alpha|^2=4$'), Line2D([], [], color=E.OKABE[1], lw=2, label=r'$|\alpha|^2=2$'),
          Line2D([], [], color=g, ls='-', label=r'$\omega/\kappa=200$, $g_z/\kappa\approx7$'),
          Line2D([], [], color=g, ls='--', label=r'$\omega/\kappa=200$, $g_z/\kappa=4,\,20$'),
          Line2D([], [], color=g, ls='-.', label=r'$\omega/\kappa=133,\,267$'),
          Line2D([], [], color=g, ls='-', marker='D', ms=3, label=r'Naseem ($\omega/\kappa=1000$)'),
          Line2D([], [], color='k', ls=(0, (1, 1)), label=r'Ma, $\omega_q$ fixed')]
    ax.legend(handles=hs, fontsize=6, loc='upper left', bbox_to_anchor=(1.01, 1.0), handlelength=2.4)
    xma = (12.0 - 2 * (6 - 4 * 0.3**2 / 2 / 18)) / (2 * 0.3**2 / 2 / 6)
    ax.axvline(0, color='0.4', lw=0.6)
    ax.annotate(r'Ma, $\omega_p=2\omega$', xy=(xma, 0.814), xytext=(3.0, 0.62), fontsize=6,
                arrowprops=dict(arrowstyle='->', lw=0.6))
    ax.set_xlim(-6, 6); ax.set_ylim(0, 1.03)
    ax.set_xlabel(r'$x=(\omega_p-\omega_p^*)/G$'); ax.set_ylabel(r'$P_c$')
    # recuadro: Wigner de Ma en ω_p = 2ω y ω_p*
    fw = os.path.join(C.DATA, 'principal_fig2_wigner.npz')
    if os.path.exists(fw):
        z = np.load(fw); vm = max(abs(z['Wa']).max(), abs(z['Wb']).max())
        for i, (Wm, t) in enumerate([(z['Wb'], r'$x=0$'), (z['Wa'], r'$\omega_p=2\omega$')]):
            ins = ax.inset_axes([0.03 + 0.15 * i, 0.58, 0.14, 0.36])
            ins.pcolormesh(z['x'], z['x'], Wm, cmap='RdBu_r', vmin=-vm, vmax=vm, shading='auto', rasterized=True)
            ins.set_aspect('equal'); ins.set_xticks([]); ins.set_yticks([])
            ins.set_title(t, fontsize=5.8, pad=1.5)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(C.AQUI, f'apendice_resonancia_universal.{ext}'))


if __name__ == '__main__':
    main()
