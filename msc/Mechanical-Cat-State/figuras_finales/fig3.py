"""Fig. 3 — figura de mérito. Solo lee la caché data/fig3/*.npz (calc_fig3.py / run_fig3.sh), escribe
data/fig3a.csv, data/fig3b.csv y dibuja fig3.pdf / fig3.png.  --rerun: calcula los puntos que falten (caro).

C1: γ_pf = 2[Γ₁⁻|α|² + Γ₁⁺(|α|²+1)], |α|² = |α_eff²| = |⟨(a-d)²⟩|, d = gz/w.
C2: κ₁ = Γ₁⁻ + Γ₁⁺ se extrae de γ_pf invirtiendo C1 con r = Γ₁⁺/Γ₁⁻ = (w²+κ²/4)/(9w²+κ²/4):
    κ₁ = γ_pf (1+r) / (2[|α|²(1+r) + r]);  κ₂ = 4G²/κ;  predicción κ₁/κ₂ = (5/72)(κ/gz)².
C3: marcador relleno si κ₂/κ ≤ 0.25 (no saturado), hueco si κ₂/κ > 0.25.
"""
import sys, glob, os
import numpy as np
import matplotlib.pyplot as plt
import comun as C
import estilo as E

KAP = 0.03


def gammas(gx, w):
    gm = gx**2 * KAP / (w**2 + KAP**2 / 4); gp = gx**2 * KAP / (9 * w**2 + KAP**2 / 4)
    return gm, gp


def cargar():
    F = []
    for f in sorted(glob.glob(os.path.join(C.DATA, 'fig3', '*.npz'))):
        z = np.load(f); F.append({k: z[k] for k in z.files if k not in ('rho', 'modos')})
    return F


def main():
    if '--rerun' in sys.argv:
        import calc_fig3
        for l in open(os.path.join(C.DATA, 'fig3', 'trabajos.txt')):
            gx, w, gzk, al2, N = l.split(); calc_fig3.punto(float(gx), float(w), float(gzk), float(al2), int(N))
    F = cargar()
    filas = []
    for f in F:
        gx, w, gz = float(f['gx']), float(f['w']), float(f['gz'])
        gm, gp = gammas(gx, w); r = gp / gm
        a2 = abs(complex(f['al2eff'])); g = float(f['gpf'])
        k1 = g * (1 + r) / (2 * (a2 * (1 + r) + r))
        k2 = float(f['kap2'])
        pred_new = 2 * (gm * a2 + gp * (a2 + 1)); pred_old = 2 * a2 * (gm + gp)
        filas.append([gx, w, float(f['gzk']), float(f['al2_nom']), int(f['N']), k2 / KAP, g, a2, float(f['Pc_fijo']),
                      k1, k2, k1 / k2, 5 / 72 * (KAP / gz)**2, g / pred_new, g / pred_old, float(f['par_borde']),
                      *np.array(f['val'])])
    T = np.array(filas)
    hdr = ('g_x [2pi GHz], omega [2pi GHz], g_z/kappa, |alpha|^2 nominal, N, kappa_2/kappa, gamma_pf spectral [2pi GHz], '
           '|alpha_eff^2|=|<(a-d)^2>|, P_c fixed polaron code, kappa_1 (C1 inverted) [2pi GHz], kappa_2=4G^2/kappa [2pi GHz], '
           'kappa_1/kappa_2 measured, (5/72)(kappa/g_z)^2, gamma_pf/pred C1, gamma_pf/pred without +1, parity-mode edge weight, '
           '|Tr rho-1|, ||rho-rho^dag||, min eig rho')
    # (a): |α|²=4 y N=22 (el N=28 es solo convergencia)
    A = T[(T[:, 3] == 4) & (T[:, 4] == 22)]
    np.savetxt(os.path.join(C.DATA, 'fig3a.csv'), A, delimiter=',', header=hdr, comments='')
    Bm = (T[:, 0] == 0.05) & (T[:, 1] == 6) & (T[:, 2] == 4)
    B = T[Bm][np.argsort(T[Bm][:, 3])]
    np.savetxt(os.path.join(C.DATA, 'fig3b.csv'), B, delimiter=',', header=hdr, comments='')

    E.aplicar()
    fig = plt.figure(figsize=(E.COL1, 4.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 0.9, 1.6], hspace=0.08)
    ax = fig.add_subplot(gs[0]); rx = fig.add_subplot(gs[1], sharex=ax); bx = fig.add_subplot(gs[2])
    fig.subplots_adjust(hspace=0.08)
    ws = sorted(set(A[:, 1])); mk = dict(zip(ws, ['o', 's', 'D', '^', 'v', 'p']))
    for i, w in enumerate(ws):
        for sat in (False, True):
            m = (A[:, 1] == w) & ((A[:, 5] > 0.25) == sat)
            if not m.any():
                continue
            x = KAP / (A[m, 2] * KAP)
            kw = dict(marker=mk[w], ls='none', ms=4, color=E.OKABE[i], mec=E.OKABE[i],
                      mfc='none' if sat else E.OKABE[i], label=rf'$\omega/2\pi={w:g}$ GHz' if not sat else None)
            ax.loglog(x, A[m, 11], **kw); rx.semilogx(x, A[m, 11] / A[m, 12], **kw)
    xx = np.geomspace(0.06, 0.6, 50)
    ax.loglog(xx, 5 / 72 * xx**2, 'k-', lw=0.8, label=r'$\frac{5}{72}(\kappa/g_z)^2$', zorder=0)
    ax.set_ylabel(r'$\kappa_1/\kappa_2$'); ax.legend(loc='upper left', fontsize=6.5)
    ax.plot([], [], 'ko', mfc='none', ms=4, ls='none', label=r'$\kappa_2/\kappa>0.25$')
    ax.legend(loc='upper left', fontsize=6.3, ncol=1)
    plt.setp(ax.get_xticklabels(), visible=False)
    E.etiqueta(ax, '(a)'); ax.texts[-1].set_position((0.93, 0.1))
    rx.axhline(1, color='k', lw=0.8); rx.axhspan(0.996, 1.004, color='0.85', lw=0)
    rx.set_ylim(0.965, 1.015); rx.set_ylabel('meas./pred.', fontsize=7)
    rx.set_xlabel(r'$\kappa/g_z$')
    from matplotlib.ticker import FixedLocator, NullFormatter
    rx.xaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5])); rx.set_xticklabels(['0.1', '0.2', '0.5'])
    rx.xaxis.set_minor_formatter(NullFormatter())
    pos = bx.get_position(); bx.set_position([pos.x0, pos.y0 - 0.06, pos.width, pos.height])
    bx.plot(B[:, 7], B[:, 13], 'o-', color=E.OKABE[0], ms=4, label=r'$2[\Gamma_1^-|\alpha|^2+\Gamma_1^+(|\alpha|^2+1)]$')
    bx.plot(B[:, 7], B[:, 14], 's--', color=E.OKABE[1], ms=4, label=r'$2|\alpha|^2(\Gamma_1^-+\Gamma_1^+)$')
    aa = np.linspace(1.5, 6.5, 50)
    gm, gp = gammas(0.05, 6.0)
    bx.plot(aa, (gm * aa + gp * (aa + 1)) / (aa * (gm + gp)), ':', color=E.OKABE[1], lw=0.8)
    bx.axhline(1, color='k', lw=0.6)
    bx.set_xlabel(r'$|\alpha|^2$'); bx.set_ylabel(r'$\gamma_{\rm pf}^{\rm meas}/\gamma_{\rm pf}^{\rm pred}$')
    bx.legend(loc='upper right', fontsize=6.3); E.etiqueta(bx, '(b)'); bx.texts[-1].set_position((0.93, 0.2))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(C.AQUI, f'fig3.{ext}'))
    print("(a) g_x ω g_z/κ κ₂/κ |α_eff²| P_c κ₁/κ₂ pred razón borde")
    for r in A[np.argsort(A[:, 2])]:
        print(f"   {r[0]:.3f} {r[1]:.0f} {r[2]:4.1f} {r[5]:.3f} {r[7]:.4f} {r[8]:.5f} {r[11]:.4e} {r[12]:.4e} {r[11]/r[12]:.4f} {r[15]:.0e}")
    ns = A[:, 5] <= 0.25
    print(f"   razón no saturados: {A[ns,11].min()/1:.0f}" if False else f"   razón no saturados [{(A[ns,11]/A[ns,12]).min():.4f}, {(A[ns,11]/A[ns,12]).max():.4f}]; saturados [{(A[~ns,11]/A[~ns,12]).min():.4f}, {(A[~ns,11]/A[~ns,12]).max():.4f}]")
    print("(b) |α|²nom N |α_eff²| razón C1  razón sin +1")
    for r in B:
        print(f"   {r[3]:.0f} {r[4]:.0f} {r[7]:.4f} {r[13]:.4f} {r[14]:.4f}")
    c = T[(T[:, 0] == 0.05) & (T[:, 1] == 6) & (T[:, 2] == 12)]
    print("conv (0.05,6,12): " + "; ".join(f"N={r[4]:.0f} γ={r[6]:.6e} α²={r[7]:.4f} P_c={r[8]:.6f}" for r in c))
    print(f"validación: |Tr-1|≤{T[:,16].max():.1e} ‖ρ-ρ†‖≤{T[:,17].max():.1e} mín eig≥{T[:,18].min():.1e}; borde máx {T[:,15].max():.1e}")


if __name__ == '__main__':
    main()
