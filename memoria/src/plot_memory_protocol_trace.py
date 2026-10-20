"""Dinámica por etapas de F17c a T=2000 (traza remota)."""
import numpy as np
from memstyle import *
z = np.load(ROOT / "data" / "memory_figs" / "protocol_trace.npz")
t, stages = z["t"], z["stages"]
fig, (a, b) = plt.subplots(2, 1, figsize=(4.7, 2.8), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
for ax in (a, b):
    for name, s0, s1 in stages:
        s0, s1 = float(s0), float(s1)
        if "load" in name or "emit" in name: ax.axvspan(s0, s1, color="#e6eff8", alpha=.7, lw=0)
        elif "hold" in name: ax.axvspan(s0, s1, color="#ebf5ef", alpha=.75, lw=0)
    style(ax)
a.plot(t, z["mode"], color=BLACK, lw=LW, ls=":", label=r"$\langle b^\dagger b\rangle$")
a.plot(t, z["d1"], color=BLUE, lw=LW, label=r"$P(D_1|R{=}0)$")
a.plot(t, z["d2"], color=RED, lw=LW, ls="-.", label=r"$P(D_2|R{=}1)$")
a.set(ylim=(-.03, 1.22)); a.set_yticks([0, .5, 1]); a.set_yticklabels([r"$0$", r"$0.5$", r"$1$"])
a.set_ylabel("Populations", fontsize=FSL)
b.plot(t, z["coherence"], color=GOLD, lw=LW, label=r"$2|\rho_{R0D_1,R1D_2}|$")
b.set(ylim=(-.02, 1.0), xlim=(float(t.min()), float(t.max())))
b.set_yticks([0, .4, .8]); b.set_yticklabels([r"$0$", r"$0.4$", r"$0.8$"])
b.set_ylabel("Coherence", fontsize=FSL); b.set_xlabel(r"$\omega_b\,t$", fontsize=FSL)
names = {"load 1": "L1", "write 1": "W1", "load 2": "L2", "write 2": "W2", "hold": "storage",
         "read 2": "R2", "emit 2": "O2", "read 1": "R1", "emit 1": "O1"}
for name, s0, s1 in stages:
    if name in names:
        a.text((float(s0) + float(s1)) / 2, 1.04, names[name], ha="center", va="bottom", fontsize=FS, transform=a.get_xaxis_transform())
a.legend(fontsize=FS, frameon=False, loc="center", bbox_to_anchor=(0.5, 0.45), handlelength=1.8, borderpad=0.2, labelspacing=0.25)
b.legend(fontsize=FS, frameon=False, loc="center", bbox_to_anchor=(0.5, 0.35), handlelength=1.8, borderpad=0.2)
tag(a, "a", x=0.995, y=0.97, ha="right"); tag(b, "b", x=0.995, y=0.97, ha="right")
fig.tight_layout(); fig.subplots_adjust(hspace=0.12)
save(fig, "figM4_protocol_dynamics")
