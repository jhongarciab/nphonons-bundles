"""Fig. M9: requirements on the noise. (a) F_h in the (chi, T) plane (contours); (b) F_h vs unequal Stark response delta for two patterns.
Data: dn3_chi*/dn4_chi*_T*.log (chi grid at T=500,1000,2000,4000,8000) and stark_{A,B}_*_T*.log; reduced model, rtol 1e-9."""
import json
import numpy as np
from scipy.interpolate import RectBivariateSpline
from memstyle import *
D = ROOT / "data"
last = lambda f: json.loads((D / f).read_text().strip().splitlines()[-1])
PAS = passive_sim(2000)[2]
TS = (500, 1000, 2000, 4000, 8000); CH = np.array([k / 17 for k in range(18)])
Z = np.array([[last(f"dn4_chi{c:.4f}_T{T}.log")["F_h"] for c in CH] for T in TS])
for T in TS: assert all(last(f"dn4_chi{c:.4f}_T{T}.log")["mineig"] > -1e-9 for c in CH)
sp = RectBivariateSpline(np.array(TS) / 1000, CH, Z, kx=3, ky=3)
tt, cc = np.linspace(0.5, 8, 200), np.linspace(0, 1, 200); ZZ = sp(tt, cc)
fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.1))
ax = axs[0]
cs = ax.contourf(cc, tt, ZZ, levels=[0, .4, .5, .6, .7, .8, .9, 1.0], cmap="RdYlBu", extend="neither")
ln = ax.contour(cc, tt, ZZ, levels=[.5, .8], colors="black", linewidths=0.8)
ax.clabel(ln, fmt={.5: r"$0.5$", .8: r"$0.8$"}, fontsize=FS - 1, inline=True, manual=[(0.25, 5.3), (0.75, 2.2)])
ax.set(xlim=(0, 1), ylim=(0.5, 8)); style(ax); ax.set_xlabel(r"$\chi$", fontsize=FS + 1); ax.set_ylabel(r"$T\ (10^3/\omega_b)$", fontsize=FS + 1, labelpad=1)
tag(ax, "a", x=0.03, y=0.97, color="black")
cb = fig.colorbar(cs, ax=ax, pad=0.03, fraction=0.06); cb.ax.tick_params(labelsize=FS - 1)
ax = axs[1]
dl = np.array([0.025 * k for k in range(13)])
for T, c, ls in ((2000, RED, "-"), (8000, GREEN, "-")):
    for p, mk in (("A", "sq"), ("B", "x")):
        y = np.array([last(f"stark_{p}_{d:.3f}_T{T}.log")["F_h"] for d in dl])
        curve(ax, dl, y, c, mk, label=rf"{p}, $T\omega_b={T}$", scale=0.7)
ax.axhline(PAS, color=BLACK, lw=LW)
ax.set(xlim=(-0.01, 0.31), ylim=(0.5, 1.0)); style(ax); ax.set_xlabel(r"$\delta$", fontsize=FS + 1);
tag(ax, "b", x=0.97, y=0.12, ha="right")
ax.legend(fontsize=FS - 1.5, frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.0), handlelength=1.2, borderpad=0.1, labelspacing=0.2)
fig.tight_layout(w_pad=2.4)
fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
x0 = inv.transform((cb.ax.get_tightbbox(r).x1, 0))[0]
x1 = inv.transform((min(l.get_window_extent(r).x0 for l in axs[1].get_yticklabels()), 0))[0]
yb = axs[1].get_position(); fig.text((x0 + x1) / 2, (yb.y0 + yb.y1) / 2, r"$F_h$", rotation=90, ha="center", va="center", fontsize=FS + 1)
save(fig, "figM9_requirements")
