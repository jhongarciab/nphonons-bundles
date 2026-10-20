"""Dinámica Raman de escritura, cortes del mapa remoto a la desintonía F13."""
import numpy as np
from memstyle import *
z = np.load(ROOT / "data" / "memory_figs" / "raman_write_map.npz")
d, t, p, q = (z[k] for k in ("deltas", "times", "pop", "cross"))
fig, axs = plt.subplots(1, 2, figsize=(4.8, 1.9), sharex=True, sharey=True)
for j, ax in enumerate(axs):
    i = int(np.argmin(abs(d - z["opt_detuning"][j])))
    ax.plot(t, p[j, i], color=BLUE, lw=LW, label=rf"$P_{{D_{j+1}}}$")
    ax.plot(t, q[j, i], color=RED, lw=LW, ls="--", label=rf"$P_{{D_{2-j}}}$")
    t0 = z["opt_duration"][j]
    ax.axvline(t0, color="gray", lw=.6, ls=":")
    ax.plot(t0, np.interp(t0, t, p[j, i]), "o", color=BLACK, ms=3, zorder=3)
    ax.set(xlim=(0, 250), ylim=(0, 1.05)); style(ax)
    ax.set_yticks([0, .5, 1]); ax.set_yticklabels([r"$0$", r"$0.5$", r"$1$"])
    ax.set_xlabel(r"$\omega_b\,t$", fontsize=FSL)
    tag(ax, "ab"[j], x=0.97, y=0.95, ha="right")
    ax.legend(fontsize=FS, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.99), handlelength=1.6, borderpad=0.2, labelspacing=0.25)
axs[0].set_ylabel("Populations", fontsize=FSL)
axs[1].tick_params(axis="y", which="both", left=False, labelleft=False)
fig.tight_layout(w_pad=0.5)
save(fig, "figM3_raman_dynamics")
