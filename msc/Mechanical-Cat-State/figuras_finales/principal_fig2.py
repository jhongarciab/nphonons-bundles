"""Figura principal 2 — resonancia vestida con estados. Solo lee cachés (no propaga):
  data/fig2/gx-0.212132_wp*_wq12.000000_Om0.060000_N22.npz  (estados estacionarios de Floquet, t = nT_p)
  data/principal_fig2d.npz                                     (opcional: gato transitorio, calc_principal_fig2d.py)
Paneles: (a) Wigner del oscilador (qubit trazado) en wp = 2w = 12; (b) en wp* = 11.98; (c) P_c(wp) con el código
fijo polarónico, marcando (a) y (b); (d) gato par transitorio en el máximo de fidelidad (si existe la caché).
Marco de laboratorio, t = nT_p: los lóbulos están en ±2i desplazados en +gz/w (real).
Escribe data/principal_fig2_wigner.npz (rejillas W) y data/principal_fig2c.csv.
"""
import os, glob
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import comun as C
import estilo as E

W_, GZ = 6.0, 0.3 * np.cos(np.pi / 4)
GXMA = -0.3 * np.sin(np.pi / 4)
N = 22
X = np.linspace(-3.2, 3.2, 241)


def cache(wp):
    f = os.path.join(C.DATA, 'fig2', f'gx{GXMA:.6f}_wp{wp:.6f}_wq12.000000_Om0.060000_N{N}.npz')
    return np.load(f)


def wigner_osc(M, n):
    rho = qt.Qobj(M, dims=[[n, 2], [n, 2]]).ptrace(0)
    return qt.wigner(rho, X, X, g=2)          # g=2: ejes en β = Re β + i Im β


def main():
    d = GZ / W_
    wpred = 2 * (W_ - 4 * GXMA**2 / (3 * W_))
    za, zb = cache(12.0), cache(11.98)
    Wa, Wb = wigner_osc(za['rho'], N), wigner_osc(zb['rho'], N)
    # (c) curva P_c con el código fijo desde la caché
    wps, pcs = [], []
    for f in sorted(glob.glob(os.path.join(C.DATA, 'fig2', f'gx{GXMA:.6f}_wp*_wq12.000000_Om0.060000_N{N}.npz'))):
        z = np.load(f); wps.append(float(z['wp'])); pcs.append(C.Pc(z['rho'], N, 2j, d))
    o = np.argsort(wps); wps, pcs = np.array(wps)[o], np.array(pcs)[o]
    np.savetxt(os.path.join(C.DATA, 'principal_fig2c.csv'), np.c_[wps, pcs], delimiter=',', comments='',
               header='omega_p [2pi GHz], P_c fixed polaron code D(gz/w)|+-2i> (steady state, t=nT_p, N=22)')
    fd = os.path.join(C.DATA, 'principal_fig2d.npz')
    zd = np.load(fd) if os.path.exists(fd) else None
    Wd = wigner_osc(zd['rho'], N) if zd is not None else None
    np.savez(os.path.join(C.DATA, 'principal_fig2_wigner.npz'), x=X, Wa=Wa, Wb=Wb, **({'Wd': Wd} if Wd is not None else {}))

    E.aplicar()
    npan = 4 if zd is not None else 3
    fig = plt.figure(figsize=(E.COL2, 2.05))
    gs = fig.add_gridspec(1, npan + 2, width_ratios=[1] * (npan - 1) + [0.06, 0.42, 1.35], wspace=0.12)
    vmax = max(abs(Wa).max(), abs(Wb).max())
    paneles = [(Wa, '(a)', rf'$\omega_p=2\omega$'), (Wb, '(b)', rf'$\omega_p=\omega_p^*$')]
    if zd is not None:
        paneles.append((Wd, '(d)', rf'transient, $\Gamma t={0.015 * zd["t"][int(zd["kmax"])]:.0f}$'))
    axs = []
    for i, (Wm, lab, tit) in enumerate(paneles):
        ax = fig.add_subplot(gs[i]); axs.append(ax)
        vm = vmax if lab != '(d)' else abs(Wm).max()
        im = ax.pcolormesh(X, X, Wm, cmap='RdBu_r', vmin=-vm, vmax=vm, shading='auto', rasterized=True)
        ax.set_aspect('equal'); ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
        ax.set_xlabel(r'Re $\beta$')
        if i == 0:
            ax.set_ylabel(r'Im $\beta$')
        else:
            ax.set_yticklabels([])
        ax.plot([d, d], [2, -2], 'k+', ms=4, mew=0.6)
        ax.text(0.04, 0.96, lab, transform=ax.transAxes, va='top', fontsize=9)
        ax.text(0.5, 1.03, tit, transform=ax.transAxes, ha='center', fontsize=7.5)
    cax = fig.add_subplot(gs[npan - 1])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax), cmap='RdBu_r'), cax=cax)
    cb.ax.set_title(r'$W(\beta)$', fontsize=7.5, pad=3); cb.ax.tick_params(labelsize=6.5)
    cx = fig.add_subplot(gs[npan + 1])
    x = np.linspace(wps[0], wps[-1], 600)
    cx.plot(x, PchipInterpolator(wps, pcs)(x), color=E.OKABE[0])
    cx.plot(wps, pcs, 'o', ms=2, color=E.OKABE[0])
    for wp, lab in [(12.0, '(a)'), (11.98, '(b)')]:
        p = pcs[np.argmin(abs(wps - wp))]
        cx.plot(wp, p, 's', ms=5, mfc='none', mec='k', mew=0.8)
        cx.annotate(lab, (wp, p), xytext=(4, -10), textcoords='offset points', fontsize=7.5)
    cx.axvline(wpred, color='k', ls=':', lw=0.7); cx.axvline(2 * W_, color='0.5', ls='-.', lw=0.7)
    cx.set_xlabel(r'$\omega_p/2\pi$ (GHz)'); cx.set_ylabel(r'$P_c$', labelpad=1); cx.set_ylim(0, 1.03)
    cx.text(0.04, 0.96, '(c)', transform=cx.transAxes, va='top', fontsize=9)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(C.AQUI, f'principal_fig2.{ext}'))
    for nom, z in [('(a) wp=12.00', za), ('(b) wp=11.98', zb)]:
        print(f"{nom}: P_c(fijo)={C.Pc(z['rho'], N, 2j, d):.5f}  P_e(prom)={float(z['Pe_prom']):.5f}  "
              f"α_eff²={complex(z['al2eff']):.3f}  paridad(prom)={float(z['par_prom']):+.4f}  W mín={(Wa if nom[1]=='a' else Wb).min():+.4f}  val={z['val']}")
    if zd is not None:
        k = int(zd['kmax'])
        print(f"(d) F={zd['F'][k]:.4f} Γt={0.015 * zd['t'][k]:.2f} P_c={zd['Pc'][k]:.5f} paridad={zd['par'][k]:.4f} W mín={Wd.min():+.4f} val={zd['val']}")
    else:
        print("(d) sin caché (calc_principal_fig2d.py no ha terminado): figura con 3 paneles")


if __name__ == '__main__':
    main()
