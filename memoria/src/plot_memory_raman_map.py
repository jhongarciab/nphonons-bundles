"""Mapas de respuesta de escritura Raman F13 (datos de scan_memory_raman_map.py en pcjhon)."""
import numpy as np
from matplotlib.colors import Normalize
from memstyle import *
z = np.load(ROOT / "data" / "memory_figs" / "raman_write_map.npz")
d, t, pop = (z[k] for k in ("deltas", "times", "pop"))
assert pop.shape == (2, len(d), len(t)) and np.all(np.isfinite(pop))
fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.3), sharex=True, sharey=True)
norm = Normalize(vmin=0, vmax=max(.85, float(pop.max())))
for j, ax in enumerate(axs):
    # Filled contours remain vector paths in the PGF output.
    m = ax.contourf(d, t, pop[j].T, levels=np.linspace(0, norm.vmax, 25), cmap=CMAP, norm=norm)
    ax.contour(d, t, pop[j].T, levels=[.2, .4, .6, .8], colors="white", linewidths=.5, alpha=.8)
    ax.plot(z["opt_detuning"][j], z["opt_duration"][j], marker="*", ms=8, color="white",
            markeredgecolor="black", markeredgewidth=.5, ls="none")
    ax.set(xlim=(d.min(), d.max()), ylim=(0, t.max())); style(ax)
    ax.set_xlabel(r"$\delta_2/\omega_b$", fontsize=FSL)
    tag(ax, "ab"[j], color="white")
    ax.text(0.5, 1.03, rf"$D_{j+1}$", transform=ax.transAxes, fontsize=FS, ha="center", va="bottom", color="#333333")
axs[0].set_ylabel(r"$\omega_b\,t$", fontsize=FSL)
axs[1].tick_params(axis="y", which="both", left=False, labelleft=False)
fig.subplots_adjust(left=0.13, right=0.88, top=0.90, bottom=0.18, wspace=0.06)
p0, p1 = axs[0].get_position(), axs[1].get_position()
cax = fig.add_axes([p1.x1 + 0.02, p0.y0, 0.02, p0.y1 - p0.y0])
cb = fig.colorbar(m, cax=cax); cb.set_ticks([0, .4, .8]); cb.ax.tick_params(labelsize=FS)
cb.set_label(r"$P_{D_s}$", fontsize=FSL)
save(fig, "figM1_raman_response")
