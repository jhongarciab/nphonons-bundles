"""P9: la desviación del confinamiento del modelo completo respecto al modelo mínimo frente a χ|α|²/κ,
χ = (8/3)g_x²/ω. Lee data/pfig3/ (modelo completo), data/efectivo/ (efectivo estático con y sin χn|e><e|)
y data/minimo/ (Δ). Escribe data/p9_diagnostico.csv y p9_diagnostico.pdf/png (figura de diagnóstico).
"""
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import comun as C
import estilo as E
import principal_fig3 as PF3

KAP = 0.03


def main():
    Tm = PF3.minimo(); k2, D0 = Tm[:, 0], Tm[:, 1]
    Delta = lambda x: np.exp(np.interp(np.log(x), np.log(k2), np.log(D0)))     # en unidades de κ
    efe = {}
    for f in glob.glob(os.path.join(C.DATA, 'efectivo', '*.npz')):
        z = np.load(f); efe[(float(z['gx']), float(z['w']), float(z['gzk']), int(z['chi']))] = float(z['conf'])
    filas = []
    for f in sorted(glob.glob(os.path.join(C.DATA, 'pfig3', '*.npz'))):
        z = np.load(f)
        gx, w, gzk, al2 = float(z['gx']), float(z['w']), float(z['gzk']), float(z['al2'])
        k2k = float(z['kap2']) / KAP
        chi = 8 * gx**2 / (3 * w)
        dmin = Delta(k2k) * KAP
        cf = PF3.conf_modelo_completo(z)[0]
        e1 = efe.get((gx, w, gzk, 1), np.nan); e0 = efe.get((gx, w, gzk, 0), np.nan)
        filas.append([gx, w, gzk, k2k, w / KAP, gx / w, chi * al2 / KAP, cf / dmin, e1 / dmin, e0 / dmin, cf / e1])
    T = np.array(filas)
    np.savetxt(os.path.join(C.DATA, 'p9_diagnostico.csv'), T, delimiter=',', comments='',
               header='g_x, omega, g_z/kappa, kappa_2/kappa, omega/kappa, g_x/omega, chi|alpha|^2/kappa (|alpha|^2=4), '
                      'conf_full/Delta_min, conf_eff(with chi)/Delta_min, conf_eff(without chi)/Delta_min, conf_full/conf_eff(with chi)')
    E.aplicar()
    fig, ax = plt.subplots(figsize=(E.COL1, 2.4))
    o = np.argsort(T[:, 6])
    ax.semilogx(T[:, 6], T[:, 7], 'o', color=E.OKABE[0], ms=4, label='full model')
    ax.semilogx(T[:, 6], T[:, 8], 'x', color=E.OKABE[1], ms=5, label=r'effective, with $\chi n|e\rangle\langle e|$')
    ax.semilogx(T[:, 6], T[:, 9], '+', color=E.OKABE[2], ms=6, label=r'effective, without $\chi$')
    for r in T:
        if r[4] > 300:
            ax.annotate(f"{r[4]:.0f}", (r[6], r[7]), xytext=(3, -8), textcoords='offset points', fontsize=5.5)
    ax.axhline(1, color='k', lw=0.6)
    ax.set_xlabel(r'$\chi|\alpha|^2/\kappa$'); ax.set_ylabel(r'confinement / $\Delta_{\rm min}$')
    ax.legend(fontsize=6, loc='lower left')
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(C.AQUI, f'p9_diagnostico.{ext}'))
    print("g_x/ω  ω/κ  g_z/κ  κ₂/κ  χ|α|²/κ  full/Δ  ef(χ)/Δ  ef(sin χ)/Δ  full/ef(χ)")
    for r in T[o]:
        print(f"{r[5]:.4f} {r[4]:5.0f} {r[2]:4.1f} {r[3]:.3f} {r[6]:8.4f}  {r[7]:.3f}  {r[8]:.3f}  {r[9]:.3f}  {r[10]:.3f}")


if __name__ == '__main__':
    main()
