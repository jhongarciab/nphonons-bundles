"""Figura térmica (P7). Solo lee la caché data/termico/ (calc_termico.py, run_termico.py).
η = γ_pf/γ_bf (definición de la Tarea 37) frente a x = h f_q/(k_B T), n_q = 1/(e^x − 1).
Paneles: (a) η(x) en la isla (κ₂/κ = 0.25, g_z/κ = 14, ω/κ = 200, |α|² = 4) con baño plano y filtrado (κ_f/ω = 0.05),
γ/κ = 2e-5 y 2e-4; eje superior f_q a 10 mK (k_B T/h = 208.37 MHz, scipy.constants). (b) mapa η(x, g_z/κ) y
(c) mapa η(x, κ₂/κ), baño filtrado y γ/κ = 2e-5, con contornos η = 100 y 220 (y los del baño plano en línea discontinua).
(d) γ_bf/κ frente a n_q en el límite de n_q pequeño con ajuste potencial.
Región caliente n_q > 0.3 (x < 1.47) achurada: no se usa para umbrales ni se interpreta.
Escribe data/termico_curvas.csv, data/termico_umbrales.csv, data/termico_mapas.npz y data/termico_ajuste_bf.csv.
"""
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.constants import k as kB, h
import comun as C
import estilo as E

KT10 = kB * 0.010 / h / 1e9          # GHz: f_q = x · k_B T/h a 10 mK
X_CAL = np.log(1 + 1 / 0.3)          # n_q = 0.3
XS_REJILLA = np.geomspace(1.0, 25.0, 40)   # misma rejilla que run_termico.XS
X_PISO = 13.0                        # por encima, γ_bf toca el piso de truncamiento (no convergido)


def cargar():
    T = []
    for f in glob.glob(os.path.join(C.DATA, 'termico', '*.npz')):
        if '_oc' in os.path.basename(f) or '_a' in os.path.basename(f).split('_w200')[1][:2] and not f.endswith('_a4.npz'):
            continue                                   # excluye controles (ocupación real) y barridos en |α|²
        z = np.load(f)
        if not np.any(np.isclose(float(z['x']), XS_REJILLA, rtol=0, atol=1e-6)):
            continue                                   # solo la rejilla (excluye búsquedas de umbral y pruebas)
        T.append([float(z['x']), float(z['nq']), float(z['k2k']), float(z['gzk']), int(z['filtro']), float(z['gam']), int(z['N']),
                  float(z['gpf']), float(z['gbf']), float(z['eta']), float(z['pcode']), float(z['herm_cruda']),
                  float(z['tr_err']), float(z['mineig']), float(z['chi_a2'])])
    return np.array(T)


def umbral(xs, eta, nivel):
    o = np.argsort(xs); xs, eta = np.asarray(xs)[o], np.asarray(eta)[o]
    m = (xs > X_CAL) & np.isfinite(eta)          # fuera de la región caliente y sin huecos de la rejilla
    xs, eta = xs[m], eta[m]
    i = np.where((eta[:-1] - nivel) * (eta[1:] - nivel) <= 0)[0]
    if not len(i):
        return np.nan
    i = i[0]
    return np.exp(np.interp(np.log(nivel), np.log(eta[i:i + 2]), np.log(xs[i:i + 2]))) if eta[i + 1] > eta[i] else np.nan


def main():
    T = cargar()
    X, NQ, K2, GZ, FIL, GAM, NN, GPF, GBF, ETA, PC, HERM, TR, ME, CHI = T.T
    isla = (abs(K2 - 0.25) < 1e-9) & (abs(GZ - 14) < 1e-9)
    # ---------- curvas base y convergencia
    filas, umb = [], []
    for fil in (1, 0):
        for gam in (2e-5, 2e-4):
            m22 = isla & (FIL == fil) & (GAM == gam) & (NN == 22); m24 = isla & (FIL == fil) & (GAM == gam) & (NN == 24)
            xs = X[m22]; o = np.argsort(xs)
            e22 = ETA[m22][o]; xs = xs[o]
            e24 = np.array([ETA[m24][np.argmin(abs(X[m24] - x))] for x in xs])
            dif = abs(e24 - e22) / e22
            for j, x in enumerate(xs):
                i22 = np.where(m22 & (X == x))[0][0]
                filas.append([fil, gam, x, NQ[i22], 22, GPF[i22], GBF[i22], e22[j], e24[j], dif[j], int(dif[j] > 0.03), int(x < X_CAL),
                              PC[i22], HERM[i22], TR[i22], ME[i22]])
            for nivel in (100, 220):
                u22, u24 = umbral(xs, e22, nivel), umbral(xs, e24, nivel)
                # bandera de convergencia en el punto del umbral: dif de η en los dos x de la rejilla que lo encierran
                j = np.searchsorted(xs, u22) if np.isfinite(u22) else 0
                dloc = dif[max(j - 1, 0):j + 1].max() if np.isfinite(u22) else np.nan
                umb.append([fil, gam, nivel, u22, u24, abs(u24 - u22), dloc, int(dloc > 0.03) if np.isfinite(dloc) else -1,
                            u22 * KT10 if np.isfinite(u22) else np.nan])
    filas = np.array(filas); umb = np.array(umb)
    np.savetxt(os.path.join(C.DATA, 'termico_curvas.csv'), filas, delimiter=',', comments='',
               header='filter(1)/flat(0), gamma/kappa, x=hf_q/kT, n_q, N, gamma_pf/kappa, gamma_bf/kappa, eta(N=22), eta(N=24), '
                      '|rel diff|, not_converged(>3%), hot(n_q>0.3), P_c, ||rho-rho^dag|| before hermitizing, |Tr-1|, min eig')
    np.savetxt(os.path.join(C.DATA, 'termico_umbrales.csv'), umb, delimiter=',', comments='',
               header='filter(1)/flat(0), gamma/kappa, eta level, x* (N=22), x* (N=24), |dx*| (N 22->24), local rel diff eta, '
                      'not_converged flag, f_q at 10 mK [GHz]')
    # ---------- mapas
    mapas = {}
    for fil in (1, 0):
        for eje, sel, par in (('gz', lambda: (abs(K2 - 0.25) < 1e-9), GZ), ('k2', lambda: (abs(GZ - 14) < 1e-9), K2)):
            m = sel() & (FIL == fil) & (GAM == 2e-5) & (NN == 22)
            xs = np.unique(X[m]); ps = np.unique(par[m])
            Z = np.full((len(ps), len(xs)), np.nan)
            for i, pv in enumerate(ps):
                for j, xv in enumerate(xs):
                    k = np.where(m & (par == pv) & (X == xv))[0]
                    if len(k):
                        Z[i, j] = ETA[k[0]]
            mapas[(fil, eje)] = (xs, ps, Z)
    np.savez(os.path.join(C.DATA, 'termico_mapas.npz'), **{f'{e}_f{f}_{c}': v for (f, e), vals in mapas.items() for c, v in zip(('x', 'p', 'eta'), vals)})
    # umbral en función del parámetro
    u_par = []
    for (fil, eje), (xs, ps, Z) in mapas.items():
        for i, pv in enumerate(ps):
            u_par.append([fil, 0 if eje == 'gz' else 1, pv, umbral(xs, Z[i], 100), umbral(xs, Z[i], 220)])
    u_par = np.array(u_par)
    np.savetxt(os.path.join(C.DATA, 'termico_umbral_parametro.csv'), u_par, delimiter=',', comments='',
               header='filter(1)/flat(0), parameter (0: g_z/kappa at kappa_2/kappa=0.25; 1: kappa_2/kappa at g_z/kappa=14), value, '
                      'x*(eta=100), x*(eta=220)  [gamma/kappa=2e-5, N=22]')
    # ---------- ajuste γ_bf ∝ n_q^p en el límite frío (la parte térmica: γ_bf(n_q) − γ_bf(T→0))
    aj = []
    for fil in (1, 0):
        for gam in (2e-5, 2e-4):
            m = isla & (FIL == fil) & (GAM == gam) & (NN == 22)
            o = np.argsort(NQ[m]); nq, gbf = NQ[m][o], GBF[m][o]
            g0 = gbf[0]                                   # x = 25: n_q ~ 1e-11, referencia T → 0
            sel = (nq > 1e-6) & (nq < 0.05) & (gbf - g0 > 10 * abs(g0) + 1e-14)
            p, lnA = np.polyfit(np.log(nq[sel]), np.log(gbf[sel] - g0), 1)
            res = np.log(gbf[sel] - g0) - (p * np.log(nq[sel]) + lnA)
            coef_lin = np.median((gbf[sel] - g0) / nq[sel])
            aj.append([fil, gam, p, np.exp(lnA), np.std(res), coef_lin, sel.sum(), nq[sel].min(), nq[sel].max(), g0])
    aj = np.array(aj)
    np.savetxt(os.path.join(C.DATA, 'termico_ajuste_bf.csv'), aj, delimiter=',', comments='',
               header='filter(1)/flat(0), gamma/kappa, exponent p (gamma_bf-gamma_bf0 = A n_q^p), A [kappa], rms log residual, '
                      'median (gamma_bf-gamma_bf0)/n_q [kappa], n points, n_q min, n_q max, gamma_bf at T->0 [kappa]')
    # ---------- figura
    E.aplicar()
    fig, axs = plt.subplots(2, 2, figsize=(E.COL2, 4.6), gridspec_kw=dict(hspace=0.42, wspace=0.3))
    ax = axs[0, 0]
    for fil, col in ((1, E.OKABE[0]), (0, E.OKABE[1])):
        for gam, ls in ((2e-5, '-'), (2e-4, '--')):
            f = filas[(filas[:, 0] == fil) & (filas[:, 1] == gam)]
            fiable = (f[:, 10] == 0) & (f[:, 2] <= X_PISO)
            ax.semilogy(f[fiable, 2], f[fiable, 7], ls, color=col, lw=1)
            # más allá: piso de truncamiento a T → 0 (no físico); se dibuja tenue, es una cota inferior de η
            k0 = np.where(fiable)[0].max()
            ax.semilogy(f[k0:, 2], f[k0:, 7], ':', color=col, lw=0.7, alpha=0.5)
    for nivel in (100, 220):
        ax.axhline(nivel, color='k', lw=0.6, ls=':')
    ax.axvspan(0, X_CAL, color='0.6', alpha=0.35, hatch='///', lw=0)
    ax.axvspan(X_PISO, 25, color='0.85', alpha=0.5, lw=0)
    ax.text(19, 30, 'truncation floor\n($\\eta$ = lower bound)', fontsize=5.5, ha='center')
    ax.set_xlim(1, 25); ax.set_xlabel(r'$x=hf_q/k_BT$'); ax.set_ylabel(r'$\eta=\gamma_{\rm pf}/\gamma_{\rm bf}$')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=E.OKABE[0], label='filtered'), Line2D([], [], color=E.OKABE[1], label='flat'),
                       Line2D([], [], color='k', ls='-', label=r'$\gamma/\kappa=2\times10^{-5}$'),
                       Line2D([], [], color='k', ls='--', label=r'$\gamma/\kappa=2\times10^{-4}$')], fontsize=6, loc='lower right')
    sec = ax.secondary_xaxis('top', functions=(lambda x: x * KT10, lambda f: f / KT10))
    sec.set_xlabel(r'$f_q$ at 10 mK (GHz)', fontsize=7)
    ax.text(0.02, 0.96, '(a)', transform=ax.transAxes, va='top')
    # mapa (γ/κ, x) en la isla
    for fil in (1, 0):
        m = isla & (FIL == fil) & (NN == 22)
        xs = np.unique(X[m]); gs = np.unique(GAM[m])
        Z = np.full((len(gs), len(xs)), np.nan)
        for i, g in enumerate(gs):
            for j, xv in enumerate(xs):
                k = np.where(m & (GAM == g) & (X == xv))[0]
                if len(k):
                    Z[i, j] = ETA[k[0]]
        mapas[(fil, 'gam')] = (xs, gs, Z)
    np.savez(os.path.join(C.DATA, 'termico_mapa_gamma.npz'), x=mapas[(1, 'gam')][0], gam=mapas[(1, 'gam')][1],
             eta_filtro=mapas[(1, 'gam')][2], eta_plano=mapas[(0, 'gam')][2])
    for axm, eje, ylab, lab in ((axs[0, 1], 'gam', r'$\gamma/\kappa$ ($\kappa_2/\kappa=0.25$, $g_z/\kappa=14$)', '(b)'), (axs[1, 0], 'k2', r'$\kappa_2/\kappa$ ($g_z/\kappa=14$, $\gamma/\kappa=2\times10^{-5}$)', '(c)')):
        xs, ps, Z = mapas[(1, eje)]
        pc = axm.pcolormesh(xs, ps, Z, norm=LogNorm(1, 1e5), cmap='viridis', shading='nearest', rasterized=True)
        axm.contour(xs, ps, Z, levels=[100, 220], colors='w', linewidths=[0.9, 0.9], linestyles=['-', '--'])
        xs0, ps0, Z0 = mapas[(0, eje)]
        axm.contour(xs0, ps0, Z0, levels=[100, 220], colors=[E.OKABE[1]], linewidths=0.9, linestyles=['-', '--'])
        axm.axvspan(1, X_CAL, color='0.6', alpha=0.5, hatch='///', lw=0)
        axm.set_xscale('log'); axm.set_yscale('log'); axm.set_xlim(1, 25)
        axm.set_xlabel(r'$x=hf_q/k_BT$'); axm.set_ylabel(ylab)
        axm.text(0.03, 0.96, lab, transform=axm.transAxes, va='top', color='w')
        fig.colorbar(pc, ax=axm, pad=0.02).set_label(r'$\eta$ (filtered)', fontsize=7)
        axm.legend(handles=[Line2D([], [], color='w', label='filtered'), Line2D([], [], color=E.OKABE[1], label='flat'),
                            Line2D([], [], color='0.5', ls='-', label=r'$\eta=100$'), Line2D([], [], color='0.5', ls='--', label=r'$\eta=220$')],
                   fontsize=5.5, loc='lower right', facecolor='0.3', framealpha=0.85, labelcolor='w')
    bx = axs[1, 1]
    for fil, col in ((1, E.OKABE[0]), (0, E.OKABE[1])):
        m = isla & (FIL == fil) & (GAM == 2e-5) & (NN == 22)
        o = np.argsort(NQ[m])
        bx.loglog(NQ[m][o], GBF[m][o], 'o', ms=2.5, color=col)
        a = aj[(aj[:, 0] == fil) & (aj[:, 1] == 2e-5)][0]
        nn = np.geomspace(a[7], a[8], 20)
        bx.loglog(nn, a[3] * nn**a[2] + a[9], '-', color=col, lw=0.8, label=rf'{"filtered" if fil else "flat"}: $p={a[2]:.2f}$')
    bx.axvspan(0.3, 3, color='0.6', alpha=0.35, hatch='///', lw=0)
    bx.set_xlabel(r'$n_q$'); bx.set_ylabel(r'$\gamma_{\rm bf}/\kappa$'); bx.legend(fontsize=6)
    bx.text(0.03, 0.96, '(d)', transform=bx.transAxes, va='top')
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(C.AQUI, f'figura_termica.{ext}'))
    # ---------- resumen
    print("umbrales (isla): filtro γ/κ nivel x*(22) x*(24) |dx| difloc bandera f_q@10mK")
    for r in umb:
        print("  " + " ".join(f"{v:.4g}" for v in r))
    print("ajuste γ_bf − γ_bf0 = A n_q^p:")
    for r in aj:
        print(f"  filtro={int(r[0])} γ/κ={r[1]:.0e}: p={r[2]:.3f} A={r[3]:.3e} rms={r[4]:.2e} mediana(Δγ_bf/n_q)={r[5]:.4f}κ n={int(r[6])} n_q∈[{r[7]:.1e},{r[8]:.1e}] γ_bf0={r[9]:.2e}")
    print("umbral frente al parámetro (γ/κ=2e-5, N=22):")
    for r in u_par:
        print(f"  filtro={int(r[0])} {'g_z/κ' if r[1]==0 else 'κ₂/κ'}={r[2]:.4g}: x*(100)={r[3]:.3f} x*(220)={r[4]:.3f}")
    print(f"no convergidos (>3%, N 22→24, isla): {int(filas[:,10].sum())} de {len(filas)}; en región caliente: {int((filas[:,10]*filas[:,11]).sum())}")
    print(f"hermiticidad cruda máx: {HERM.max():.1e}; |Tr−1| máx: {TR.max():.1e}; mín autovalor: {ME.min():.1e}")


if __name__ == '__main__':
    main()
