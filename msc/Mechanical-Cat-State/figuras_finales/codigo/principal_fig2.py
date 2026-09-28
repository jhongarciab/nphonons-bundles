"""Figura principal 2 — universalidad de la figura de mérito. Solo lee cachés:
  data/fig3/gx*.npz  (validación Fig. 3: 15 puntos |α|²=4 y barrido |α|²=2,4,6; N=22/26/28)
  data/pfig3/*.npz   (P9: mismos g_x/ω, g_z/κ, κ₂/κ con ω/κ = 500 y 1000)
y = (κ₁/κ₂)(g_z/κ)² frente a x = g_x/ω; κ₁ de la tasa de paridad espectral invirtiendo C1 con |α_eff²|,
κ₂ = 4G²/κ. Teoría: y = 5/72. Color por κ₂/κ, marcador por ω/κ. Puntos sin gato estable (P_c < 0.99) en gris.
(b): y frente a κ₂/κ (corrección no adiabática).
Escribe data/principal_fig2.csv.
"""
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import comun as C
import estilo as E

KAP = 0.03


def k1_inv(g, gx, w, a2):
    gm = gx**2 * KAP / (w**2 + KAP**2 / 4); gp = gx**2 * KAP / (9 * w**2 + KAP**2 / 4); r = gp / gm
    return g * (1 + r) / (2 * (a2 * (1 + r) + r))


def cargar():
    filas = {}
    for f in glob.glob(os.path.join(C.DATA, 'fig3', 'gx*.npz')):
        z = np.load(f)
        if int(z['N']) == 28:                      # solo convergencia
            continue
        gx, w, gzk = float(z['gx']), float(z['w']), float(z['gzk'])
        filas[(gx, w, gzk, float(z['al2_nom']))] = (z, float(z['Pc_fijo']))
    for f in glob.glob(os.path.join(C.DATA, 'pfig3', '*.npz')):
        z = np.load(f)
        gx, w, gzk = float(z['gx']), float(z['w']), float(z['gzk'])
        if w > 10:
            filas[(gx, w, gzk, 4.0)] = (z, float(z['Pc_ss']))
    T = []
    for (gx, w, gzk, al2), (z, pc) in filas.items():
        a2 = abs(complex(z['al2eff'])); k2 = float(z['kap2'])
        k1 = k1_inv(float(z['gpf']), gx, w, a2)
        chi = 8 * gx**2 / (3 * w)
        T.append([gx, w, gzk, al2, w / KAP, gx / w, k2 / KAP, chi * a2 / KAP, pc, k1, k2, k1 / k2 * gzk**2])
    return np.array(sorted(T, key=lambda r: r[5]))


def main():
    T = cargar()
    np.savetxt(os.path.join(C.DATA, 'principal_fig2.csv'), T, delimiter=',', comments='',
               header='g_x, omega, g_z/kappa, |alpha|^2 nominal, omega/kappa, g_x/omega, kappa_2/kappa, chi|alpha_eff|^2/kappa, '
                      'P_c steady (fixed polaron code), kappa_1 (C1 inverted), kappa_2=4G^2/kappa, y=(kappa_1/kappa_2)(g_z/kappa)^2')
    ok = T[:, 8] > 0.99
    E.aplicar()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(E.COL2, 2.5), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.28))
    norm = LogNorm(3e-3, 1.2)
    mk = lambda wk: 'o' if wk < 180 else ('s' if wk < 220 else ('D' if wk < 300 else '^'))
    for i, r in enumerate(T):
        if ok[i]:
            for a in (ax, bx):
                a.scatter(r[5] if a is ax else r[6], r[11] / (5 / 72), c=[r[6]], norm=norm, cmap='viridis', marker=mk(r[4]),
                          s=22, edgecolors='k', linewidths=0.4, zorder=3)
        else:
            ax.scatter(r[5], r[11] / (5 / 72), marker='x', color='0.5', s=22, zorder=3)
    for a in (ax, bx):
        a.axhline(1, color='k', lw=0.8)
        a.set_ylabel(r'$(\kappa_1/\kappa_2)(g_z/\kappa)^2\,/\,(5/72)$' if a is ax else '')
    ax.set_xlabel(r'$g_x/\omega$'); bx.set_xlabel(r'$\kappa_2/\kappa$'); bx.set_xscale('log')
    ax.set_ylim(0.95, 1.01); bx.set_ylim(0.95, 1.01)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    cb = fig.colorbar(sm, ax=[ax, bx], pad=0.015, fraction=0.03); cb.set_label(r'$\kappa_2/\kappa$')
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker=m, ls='none', mfc='0.7', mec='k', ms=4, label=l) for m, l in
          [('o', r'$\omega/\kappa<180$'), ('s', r'$\omega/\kappa=200$'), ('D', r'$\omega/\kappa=233$–$267$'), ('^', r'$\omega/\kappa=500$')]]
    hs.append(Line2D([], [], marker='x', ls='none', color='0.5', ms=4, label=r'no stable cat ($\omega/\kappa=1000$)'))
    ax.legend(handles=hs, fontsize=5.8, loc='lower left', ncol=2)
    ax.text(0.02, 0.96, '(a)', transform=ax.transAxes, va='top'); bx.text(0.04, 0.96, '(b)', transform=bx.transAxes, va='top')
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(C.AQUI, f'principal_fig2.{ext}'))
    print(f"estables: {ok.sum()}  y/(5/72) ∈ [{(T[ok,11]/(5/72)).min():.4f}, {(T[ok,11]/(5/72)).max():.4f}]")
    print(f"corr(1−y', κ₂/κ)={np.corrcoef(1-T[ok,11]/(5/72), T[ok,6])[0,1]:.2f}  corr(1−y', g_x/ω)={np.corrcoef(1-T[ok,11]/(5/72), T[ok,5])[0,1]:.2f}  "
          f"corr(1−y', χ|α|²/κ)={np.corrcoef(1-T[ok,11]/(5/72), T[ok,7])[0,1]:.2f}")
    for r in T[~ok]:
        print(f"inestable: g_x/ω={r[5]:.3f} ω/κ={r[4]:.0f} P_c={r[8]:.3f} y'={r[11]/(5/72):.3f}")


if __name__ == '__main__':
    main()
