"""Barrido F17c frente al tiempo de espera (soluciones remotas en data/memory_figs)."""
import json
import numpy as np
from memstyle import *
rows = sorted((json.loads(p.read_text().splitlines()[-1]) for p in (ROOT / "data" / "memory_figs").glob("operational_T*.json")), key=lambda x: x["T"])
rows = [r for r in rows if r["T"] % 400 == 0]   # equispaced grid T = 0, 400, ..., 8000
if len(rows) < 10: raise SystemExit(f"faltan puntos remotos: {len(rows)}")
T = np.array([r["T"] for r in rows])
assert len(T) == 21 and np.allclose(np.diff(T), 400)
for r in rows:
    assert r["mode"] == "real" and r["readout"] == "real" and r["kin_frac"] == r["kout_frac"] == 10
    assert abs(r["F"] / r["s"] - r["F_h"]) < 1e-9
x = T / 1000
# modelo completo (8 estados, biexciton a -9 wb, puertos reales): data/f19_full_dxxm9[_T*].log
full = {}
for T_, f in ((500, "f19_full_dxxm9_T500.log"), (2000, "f19_full_dxxm9.log"), (8000, "f19_full_dxxm9_T8000.log")):
    q = ROOT / "data" / f
    if q.exists():
        d = json.loads(q.read_text().strip().splitlines()[-1]); assert d["T"] == T_ and not d["proj"] and d["mineig"] > -1e-9 and d["ok_c2"]
        full[T_] = d
fig, axs = plt.subplots(1, 3, figsize=(5.0, 1.9), sharex=True)
for ax, key, pk, yl, letter in zip(axs, ("F", "s", "F_h"), ("pas_F", "pas_s", "pas_Fh"),
                                   (r"$F$", r"$s$", r"$F_h$"), "abc"):
    curve(ax, x, [r[key] for r in rows], RED, "circ", scale=0.7)
    curve(ax, x, [passive_sim(r["T"])[("pas_F", "pas_s", "pas_Fh").index(pk)] for r in rows], BLACK, "circ", ls="--", marker_off=True)
    ax.set(xlim=(0, 8.1), ylim=(0, 1.05)); style(ax)
    ax.set_yticks([0, .5, 1]); ax.set_yticklabels([r"$0$", r"$0.5$", r"$1$"])
    ax.set_xticks([0, 4, 8])
    ax.set_ylabel(yl, fontsize=FSL, labelpad=3)
    ax.axvline(2, color="#c9ced1", lw=.6, ls=":")
    tag(ax, letter, x=0.97, y=0.95, ha="right")
axs[1].set_xlabel(r"$T\ (10^3/\omega_b)$", fontsize=FSL)
from matplotlib.lines import Line2D
hs = [Line2D([], [], color=RED, lw=LW, marker=MK["circ"][0], ms=MK["circ"][1] * 0.7, mfc="none", mew=0.8, label="memory"),
      Line2D([], [], color=BLACK, lw=LW, ls="--", label="passive")]
fig.legend(handles=hs, loc="lower center", ncol=2, fontsize=FS, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=1.5)
axs[2].set_ylim(0, 1.2)
fig.tight_layout(w_pad=1.2)
save(fig, "figM2_operational_sweep")
