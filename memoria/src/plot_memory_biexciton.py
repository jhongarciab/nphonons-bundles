"""Biexciton figure: off-resonant transfer, resolved resonances, full protocol."""
import json
import numpy as np
from memstyle import *
D = ROOT / "data"
last = lambda f: json.loads((D / f).read_text().strip().splitlines()[-1])
red = last("f17tol9_ro_real10_kout10_T2000.log")
pts = []
for x in (-12, -9, -8, -7, -6, -5, -4, -3, -2):
    f = f"f19_full_dxxm{-x}.log"
    if (D / f).exists():
        d = last(f); assert d["mineig"] > -1e-9 and d["ok_c2"] and not d["proj"]; pts.append((x, d["s"], d["F_h"]))
fig, ax = plt.subplots(figsize=(2.88, 1.9))
if pts:
    x, s_, fh = zip(*pts)
    curve(ax, x, s_, BLUE, "sq", label=r"$s$"); curve(ax, x, fh, RED, "circ", label=r"$F_h$")
ax.axhline(red["s"], color=BLUE, lw=.6, ls="--"); ax.axhline(red["F_h"], color=RED, lw=.6, ls="--")
ax.set(xlim=(-14, -1.5), ylim=(0, 1.1)); style(ax); ax.set_xlabel(r"$\Delta_{xx}/\omega_b$", fontsize=FSL)
ax.set_ylabel(r"$s,F_h$", fontsize=FSL, labelpad=4); ax.set_yticks([0, .5, 1]); ax.set_yticklabels([r"$0$", r"$0.5$", r"$1$"]); ax.set_xticks([-12, -8, -4]); ax.set_xticklabels([r"$-12$", r"$-8$", r"$-4$"])
ax.legend(fontsize=FS, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0), handlelength=1.0, ncol=2, borderpad=0.2, columnspacing=0.8)
fig.tight_layout(pad=0.3)
save(fig, "figM8_biexciton")
