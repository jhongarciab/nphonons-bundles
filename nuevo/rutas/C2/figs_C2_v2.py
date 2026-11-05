"""Figuras en estilo de articulo de optica cuantica (referencia: mapa 2D grande + cortes anotados, etiquetas de color sobre las curvas, letras en negrita)."""
import sys, re, numpy as np
sys.path.insert(0, '../../kit')
from figstyle import *
import matplotlib.pyplot as plt, matplotlib.gridspec as gs
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.ndimage import gaussian_filter1d
from captura import pulse, optimize_G, ideal_G, simulate
ke0, ki, Gmax, Tp0 = 1e-2, 1e-3, 0.012, 100.0
BOLD = FontProperties(family='cmb10', size=12)
def lab(ax, s, x=0.03, y=0.95, color='k'): ax.text(x, y, '(%s)'%s, transform=ax.transAxes, ha='left', va='top', fontproperties=BOLD, color=color, zorder=30)
def txt(ax, x, y, s, color, fs=9, **kw): ax.text(x, y, s, color=color, fontsize=fs, ha=kw.pop('ha', 'center'), va=kw.pop('va', 'center'), **kw)
def arrow_note(ax, xy, xytext, s, color, fs=9): ax.annotate(s, xy=xy, xytext=xytext, color=color, fontsize=fs, ha='center', va='center', arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
def minibar(ax, cmap, norm, ticks, labels, title, loc=(0.74, 0.08, 0.05, 0.32)):
    ia = ax.inset_axes(loc); g = np.linspace(norm.vmax, norm.vmin, 200)[:, None]; ia.imshow(g, aspect='auto', cmap=cmap, norm=norm, extent=(0, 1, norm.vmin, norm.vmax))
    ia.set_xticks([]); ia.yaxis.tick_right(); ia.set_yticks(ticks); ia.set_yticklabels(labels, fontsize=8); ia.minorticks_off(); ia.tick_params(length=2, direction='in')
    ax.text(loc[0]-0.01, loc[1]+loc[3]/2, title, transform=ax.transAxes, ha='right', va='center', fontsize=10)
CM_ETA = LinearSegmentedColormap.from_list('eta', ['#ffffff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'])
# ---------------- datos ----------------
KE = np.geomspace(0.003, 0.06, 9); TP = np.geomspace(30, 300, 9); Z = np.full((9, 9), np.nan)
for p in (0, 1):
    for i, j, e, _ in np.load('mapa_parte%d.npy'%p): Z[int(i), int(j)] = e
def parse_canal(fn='canal2.out'):
    d = {}
    for ln in open(fn):
        m = re.match(r'(\w+)\s+(\w+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+\|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', ln)
        if m: p, k, c, T, sep, P0, P1, V, s, F = m.groups(); d[(p, k, float(c), float(T))] = (float(s), float(F), float(V))
    return d
canal = parse_canal(); chi_tab = np.load('chi_V0.npy'); gph = 4e-4
def V_of(chi, T): V0 = np.interp(chi, chi_tab[:, 0], chi_tab[:, 2]); return V0*np.exp(-(5*gph/6)*(1-chi)*T)
# ================= Figura 1: limites de la captura =================
fig = plt.figure(figsize=(7.4, 4.7)); G_ = gs.GridSpec(2, 2, width_ratios=[1, 1.5], hspace=0.40, wspace=0.42)
aa = fig.add_subplot(G_[0, 0]); ab = fig.add_subplot(G_[1, 0]); ac = fig.add_subplot(G_[:, 1])
rows = (2, 4, 6, 8); cols = [C['blue'], C['green'], C['orange'], C['mag']]
for n_, (r, col) in enumerate(zip(rows, cols)):
    ke = KE[r]; aa.plot(TP, Z[r], '-', color=col, lw=2); aa.axhline(ke/(ke+ki), xmin=0.5, color=col, ls=':', lw=1.1)
    aa.text(0.97, 0.46-0.09*n_, r'$\kappa_e=%.4f$'%ke, transform=aa.transAxes, color=col, fontsize=8.5, ha='right')
aa.text(0.97, 0.46-0.09*4, r'dotted: $\kappa_e/(\kappa_e+\kappa_i)$', transform=aa.transAxes, fontsize=8, ha='right')
aa.set_xscale('log'); aa.set_xticks([30, 50, 100, 200, 300]); aa.set_xticklabels(['30', '50', '100', '200', '300']); aa.set_xlabel(r'$T_p\,\omega_b$'); aa.set_ylabel(r'$\eta_{\rm cap}$'); aa.set_ylim(0.3, 1.05); aa.set_xlim(30, 300); lab(aa, 'a', y=0.97)
for n_, (jj, col) in enumerate(((1, C['red']), (4, C['green']), (7, C['blue']))):
    ab.plot(KE, Z[:, jj], '-', color=col, lw=2); ab.text(0.97, 0.40-0.085*n_, r'$T_p\omega_b=%.0f$'%TP[jj], transform=ab.transAxes, color=col, fontsize=8.5, ha='right')
kk = np.geomspace(0.003, 0.06, 100); ab.plot(kk, kk/(kk+ki), 'k--', lw=1.3); ab.text(0.97, 0.40-0.085*3, r'dashed: $\kappa_e/(\kappa_e+\kappa_i)$', transform=ab.transAxes, fontsize=8, ha='right')
ab.set_xscale('log'); ab.set_xticks([0.003, 0.005, 0.01, 0.02, 0.05]); ab.set_xticklabels(['0.003', '0.005', '0.01', '0.02', '0.05']); ab.set_xlabel(r'$\kappa_e/\omega_b$'); ab.set_ylabel(r'$\eta_{\rm cap}$'); ab.set_ylim(0.2, 1.05); ab.set_xlim(0.003, 0.06); lab(ab, 'b', y=0.97)
Kf, Tf = np.meshgrid(np.geomspace(0.003, 0.06, 90), np.geomspace(30, 300, 90)); from scipy.interpolate import RegularGridInterpolator as RGI
rg = RGI((np.log(KE), np.log(TP)), Z, method='cubic'); Zf = rg(np.stack([np.log(Kf), np.log(Tf)], -1))
im = ac.pcolormesh(Kf, Tf, Zf, cmap=CM_ETA, vmin=0.5, vmax=1.0, shading='auto', rasterized=True)
cs = ac.contour(Kf, Tf, Zf, levels=[0.8, 0.9, 0.95], colors='#e6550d', linewidths=1.3)
ac.clabel(cs, fmt='%.2f', fontsize=8, inline=True)
for e_ in (0.9, 0.95):
    x_ = e_/(1-e_)*ki; ac.axvline(x_, color='k', ls='--', lw=1); ac.text(x_, 1.012, r'%.2f'%e_, transform=ac.get_xaxis_transform(), ha='center', fontsize=8)
ac.text(0.5, 1.065, r'$\kappa_e/(\kappa_e+\kappa_i)=$', transform=ac.transAxes, ha='right', fontsize=8); ac.text(0.5, 1.065, '', transform=ac.transAxes)
ac.set_xscale('log'); ac.set_yscale('log'); ac.set_xticks([0.003, 0.005, 0.01, 0.02, 0.05]); ac.set_xticklabels(['0.003', '0.005', '0.01', '0.02', '0.05']); ac.set_yticks([30, 50, 100, 200, 300]); ac.set_yticklabels(['30', '50', '100', '200', '300']); ac.set_xlabel(r'$\kappa_e/\omega_b$', fontsize=12); ac.set_ylabel(r'$T_p\,\omega_b$', fontsize=12)
ac.axhline(TP[4], color='k', ls=':', lw=1.2); ac.axhline(TP[4], color='k', ls=':', lw=0)
lab(ac, 'c', y=0.97, color='k')
arrow_note(ac, (0.0036, 47), (0.0049, 37), 'bandwidth\nlimited ($G_{\\max}$)', C['red'], 8)
arrow_note(ac, (0.05, 190), (0.036, 250), r'$\kappa_i$-limited plateau', C['blue'], 8.5)
minibar(ac, CM_ETA, plt.Normalize(0.5, 1.0), [0.5, 0.75, 1.0], ['0.5', '0.75', '1'], r'$\eta_{\rm cap}$', loc=(0.12, 0.50, 0.05, 0.28))
ac.set_xlim(0.003, 0.06); ac.set_ylim(30, 300)
finish(fig); fig.savefig('figs/fig1_limites_captura.png'); plt.close(fig)
# ================= Figura 2: canal completo =================
fig = plt.figure(figsize=(7.4, 4.7)); G_ = gs.GridSpec(2, 2, width_ratios=[1, 1.5], hspace=0.40, wspace=0.40)
aa = fig.add_subplot(G_[0, 0]); ab = fig.add_subplot(G_[1, 0]); ac = fig.add_subplot(G_[:, 1])
Ts = np.array([0, 2000, 8000.])
for (p, k, col, nm, dy) in (('directa', 'gauss', C['green'], 'direct', 0.06), ('dos_etapas', 'gauss', C['blue'], 'two-stage', 0.06), ('pasivo', 'gauss', C['gray'], 'passive', 0.07)):
    y = [canal[(p, k, 1.0, T)][0] for T in Ts]; aa.plot(Ts, y, 'o-', color=col, lw=2, ms=4); txt(aa, 5600, y[-1]+dy, nm, col, fs=9)
aa.set_xlabel(r'wait $T\,\omega_b$'); aa.set_ylabel(r'recovery $s$'); aa.set_ylim(0, 0.95); lab(aa, 'a', y=0.97); aa.text(0.97, 0.52, 'Gaussian input', transform=aa.transAxes, ha='right', fontsize=8)
for chi, col in ((1.0, C['green']), (0.9, C['orange']), (0.0, C['red'])):
    ab.plot(Ts, [canal[('directa', 'gauss', chi, T)][1] for T in Ts], 'o-', color=col, lw=2, ms=4); ab.plot(Ts, [canal[('dos_etapas', 'gauss', chi, T)][1] for T in Ts], 's--', color=col, lw=1.2, ms=3, mfc='none')
    txt(ab, 8450, canal[('directa', 'gauss', chi, 8000.0)][1], r'$\chi=%.1f$'%chi, col, fs=8, ha='left')
ab.axhline(0.5, color='k', lw=0.8); ab.set_xlim(-300, 11500); ab.set_xlabel(r'wait $T\,\omega_b$'); ab.set_ylabel(r'$F_h$'); ab.set_ylim(0.40, 1.02); lab(ab, 'b', y=0.97)
ab.text(0.04, 0.22, 'solid: direct\ndashed: two-stage', transform=ab.transAxes, fontsize=7.5); ab.text(0.50, 0.06, r'$F_h=0.5$', transform=ab.transAxes, fontsize=7.5, ha='center')
chis = np.linspace(0, 1, 120); Tg = np.linspace(0, 10000, 200); CH, TT = np.meshgrid(chis, Tg); FH = (1+V_of(CH, TT))/2
cm = plt.get_cmap('RdBu'); norm = TwoSlopeNorm(vmin=0.5, vcenter=0.7, vmax=0.9)
im = ac.pcolormesh(CH, TT, FH, cmap=cm, norm=norm, shading='auto', rasterized=True)
c1 = ac.contour(CH, TT, FH, levels=[0.6, 0.8], colors='k', linewidths=[1.4, 1.4], linestyles=['-', '--']); ac.clabel(c1, fmt={0.6: '0.6', 0.8: '0.8'}, fontsize=9, manual=[(0.3, 3800), (0.9, 6500)])
ac.set_xlabel(r'noise correlation $\chi$', fontsize=12); ac.set_ylabel(r'wait $T\,\omega_b$', fontsize=12); lab(ac, 'c', y=0.97, color='w')
for chi_ in (0.9, 0.995): ac.axvline(chi_, color='k', ls=':', lw=1)
txt(ac, 0.27, 8300, r'$\Gamma_{D_1D_2}=\frac{5\gamma_\varphi}{6}(1-\chi)$', 'w', fs=10); txt(ac, 0.27, 7300, 'coherence decays', 'w', fs=9)
arrow_note(ac, (0.975, 9200), (0.68, 9500), r'protected ($\chi\to1$)', 'k', 9)
minibar(ac, cm, norm, [0.5, 0.7, 0.9], ['0.5', '0.7', '0.9'], r'$F_h$', loc=(0.50, 0.07, 0.05, 0.26))
ac.set_xlim(0, 1); ac.set_ylim(0, 10000)
finish(fig); fig.savefig('figs/fig2_canal_completo.png'); plt.close(fig)
# ================= Figura 3: dinamica y robustez =================
t = np.arange(-9*Tp0, 9*Tp0, 2.0); u = pulse('gauss', Tp0, t, 0.0); Gi, _ = ideal_G(u, t, ke0+ki); Gopt, eta = optimize_G(u, t, ke0, ki, Gmax, G0=Gi)
Gill = np.clip(Gi, -Gmax, Gmax); Gill[np.abs(t) > 350] = 0.0; Gill = gaussian_filter1d(Gill, 12); A, D = simulate(Gill, u, t, ke0, ki, ret=True)
refl = np.abs(u+np.sqrt(ke0)*A)**2; loss = ki*np.abs(A)**2
fig, ax = plt.subplots(2, 3, figsize=(7.8, 4.6)); plt.subplots_adjust(hspace=0.50, wspace=0.85, left=0.09, right=0.93)
a = ax[0, 0]; a.plot(t, u**2*1e3, color=C['blue'], lw=2); a.set_ylabel(r'$|u|^2\ (10^{-3}\omega_b)$', color=C['blue']); a.set_xlabel(r'$t\,\omega_b$'); a.set_xlim(-450, 450); lab(a, 'a')
b2 = a.twinx(); b2.plot(t, -Gill*1e3, color=C['orange'], lw=2); b2.set_ylabel(r'$|G|\ (10^{-3}\omega_b)$', color=C['orange']); b2.tick_params(axis='y', colors=C['orange']); b2.set_ylim(0, 14)
txt(a, -280, 3.3, 'input', C['blue'], 9); txt(a, 200, 0.8, 'control', C['orange'], 9)
a = ax[0, 1]; a.plot(t, np.abs(D)**2, color=C['green'], lw=2); a.set_ylim(0, 1); a.set_xlim(-450, 450); a.set_xlabel(r'$t\,\omega_b$'); a.set_ylabel(r'$|d|^2$', color=C['green']); lab(a, 'b')
b3 = a.twinx(); b3.plot(t, np.abs(A)**2*1e2, color=C['mag'], lw=2); b3.set_ylabel(r'$|a|^2\ (10^{-2})$', color=C['mag']); b3.tick_params(axis='y', colors=C['mag'])
txt(a, 300, 0.60, 'dark state', C['green'], 9); txt(a, -250, 0.5, 'cavity', C['mag'], 9)
a = ax[0, 2]; a.plot(t, np.abs(D)**2, color=C['green'], lw=2); a.plot(t, np.cumsum(refl)*2.0, color=C['red'], lw=2); a.plot(t, np.cumsum(loss)*2.0, color=C['gray'], lw=2)
a.set_xlim(-450, 650); a.set_xlabel(r'$t\,\omega_b$'); a.set_ylabel('cumulative fraction'); lab(a, 'c', y=0.65)
txt(a, 330, 0.62, 'stored', C['green'], 9); txt(a, 330, 0.17, 'lost ($\\kappa_i$)', C['gray'], 8.5); txt(a, 330, 0.0, 'reflected', C['red'], 8.5, va='top'); a.set_ylim(-0.08, 1.0)
dts = np.array([-60, -30, -15, 0, 15, 30, 60.]); sc = np.array([0.8, 0.9, 1.0, 1.1, 1.2]); dcs = np.array([-0.01, -0.005, 0, 0.005, 0.01])
G0 = Gopt
for a, x, y, xl, col, L in ((ax[1, 0], dts, [simulate(G0, pulse('gauss', Tp0, t, d), t, ke0, ki) for d in dts], r'arrival error $(\omega_b^{-1})$', C['blue'], 'd'),
                            (ax[1, 1], sc, [simulate(np.clip(G0*s_, -1, 1), u, t, ke0, ki) for s_ in sc], r'$\Omega$ scale', C['orange'], 'e'),
                            (ax[1, 2], dcs/(ke0+ki), [simulate(G0, u, t, ke0, ki, dc=d) for d in dcs], r'detuning$/\kappa$', C['mag'], 'f')):
    a.plot(x, y, 'o-', color=col, lw=2, ms=4); a.set_xlabel(xl); a.set_ylabel(r'$\eta_{\rm cap}$'); a.set_ylim(0.45, 0.95); lab(a, L, y=0.45)
arrow_note(ax[1, 2], (0.45, 0.77), (-0.1, 0.58), 'most critical', C['mag'], 8.5)
finish(fig); fig.savefig('figs/fig3_dinamica_robustez.png'); plt.close(fig)
print('ok')
