"""Retención de los oscuros durante la espera (scan_dark_storage_decay.py en pcjhon)."""
import numpy as np
from memstyle import *
d = np.load(ROOT / "data" / "memory_figs" / "dark_storage_decay.npz")
t = d["t"]
print("T=2000,8000:", {k: np.interp([2000, 8000], t, d[k]).round(4).tolist() for k in ("d1_chi1_pD1", "d2_chi1_pD2", "coh_chi1", "coh_chi0.9", "coh_chi0")})
fig, ax = plt.subplots(1, 2, figsize=(4.8, 1.9), sharex=True)
a = ax[0]
a.plot(t / 1000, d["d1_chi1_pD1"], color=BLUE, lw=LW, label=r"$D_1$")
a.plot(t / 1000, d["d2_chi1_pD2"], color=RED, lw=LW, ls="-.", label=r"$D_2$")
a.plot(t / 1000, np.exp(-0.001 * t), color=BLACK, lw=LW, ls="--", label=r"$e^{-\kappa t}$")
a.set_ylabel("Retained population", fontsize=FSL)
tag(a, "a", x=0.97, y=0.97, ha="right")
a = ax[1]
a.plot(t / 1000, d["coh_chi1"], color=BLUE, lw=LW, label=r"$\chi=1$")
a.plot(t / 1000, d["coh_chi0.9"], color=GOLD, lw=LW, ls="--", label=r"$\chi=0.9$")
a.plot(t / 1000, d["coh_chi0"], color=RED, lw=LW, ls="-.", label=r"$\chi=0$")
a.plot(t / 1000, np.exp(-(5 / 6) * 4e-4 * t), color=BLACK, lw=LW, ls=":", label="secular")
a.set_ylabel(r"$2|\rho_{D_1D_2}|$", fontsize=FSL)
tag(a, "b", x=0.97, y=0.97, ha="right")
for a in ax:
    a.set(xlim=(0, t[-1] / 1000), ylim=(0, 1.5)); style(a)
    a.set_yticks([0, .5, 1]); a.set_yticklabels([r"$0$", r"$0.5$", r"$1$"])
    if a is ax[0]: a.legend(fontsize=FS, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.08), handlelength=1.6, borderpad=0.2, labelspacing=0.25)
    else: a.legend(fontsize=FS - 1, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.99), ncol=2, handlelength=1.4, columnspacing=0.8, borderpad=0.2, labelspacing=0.25)
    a.set_xlabel(r"$T\ (10^3/\omega_b)$", fontsize=FSL)
fig.tight_layout(w_pad=0.8)
save(fig, "figM5_dark_storage_decay")
