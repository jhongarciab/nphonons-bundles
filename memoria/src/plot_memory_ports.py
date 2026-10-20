"""Fig. M7: F_h vs (a) input-port rate, (b) output-port rate, (c) noise correlation chi, (d) strain-coupling scale S.
(a-c) reduced model, rtol 1e-9, 18 equispaced points, dn3_*/dn4_*.log at T=500, 2000, 8000. (d) pulses reoptimized for each S (f18_lambda_S*.json)."""
import json, glob
import numpy as np
from memstyle import *
D = ROOT / "data"
last = lambda f: json.loads((D / f).read_text().strip().splitlines()[-1])
PAS = passive_sim(2000)[2]
TS = ((500, BLUE, "x"), (2000, RED, "circ"), (8000, GREEN, "sq"))
KIN = tuple(1.5 * k for k in range(1, 19))     # 1.5, 3, ..., 27 (18 points)
CHI = tuple(k / 17 for k in range(0, 18))      # 0, 1/17, ..., 1 (18 points)
def pts(kind, vals, T):
    out = []
    for v in vals:
        f = f"dn3_{kind}{v:.1f}_T{T}.log" if kind != "chi" else f"dn4_chi{v:.4f}_T{T}.log"
        d = last(f); assert d["mineig"] > -1e-9; out.append((float(v), d["F_h"]))
    return zip(*out)
def strain():
    best = {}
    for f in glob.glob(str(D / "f18_lambda_S*.json")) + glob.glob(str(D / "f18_prev" / "f18_lambda_S*.json")):
        j = json.load(open(f))
        if "real_io" not in j or abs(j["S"] * 20 - round(j["S"] * 20)) > 1e-9 or j["S"] < 0.15: continue
        ide = {(e["T"], e.get("chi")): e["F_h"] for e in j["ideal_io"]}
        row = (j["real_io"]["F_h"], j["real_io"]["F_h"], ide[(2000, None)], ide[(2000, 0.0)])   # several optimizations for some S: keep the pulses with the highest real-port F_h
        if j["S"] not in best or row[0] > best[j["S"]][0]: best[j["S"]] = row
    S = np.array(sorted(best)); return (S,) + tuple(np.array([best[s][k] for s in S]) for k in (1, 2, 3))
fig, axs = plt.subplots(2, 2, figsize=(4.4, 3.5))
ticks = lambda ax, xt: (ax.set_xticks(xt), ax.set_xticklabels([rf"${v:g}$" for v in xt]), ax.minorticks_off(),
                        ax.set_yticks([0, .5, 1]), ax.set_yticklabels([r"$0$", r"$0.5$", r"$1$"]), style(ax))
for ax, kind, vals, xl, letter, xt in ((axs[0, 0], "kin", KIN, r"$\kappa_{\rm in}/\kappa$", "a", [0, 10, 20]),
                                        (axs[0, 1], "kout", KIN, r"$\kappa_{\rm out}/\kappa$", "b", [0, 10, 20]),
                                        (axs[1, 0], "chi", CHI, r"$\chi$", "c", [0, 0.5, 1])):
    for T, c, mk in TS:
        x, y = pts(kind, vals, T); curve(ax, x, y, c, mk, label=rf"$T\omega_b={T}$", scale=0.7)
    if kind != "kin": ax.axhline(PAS, color=BLACK, lw=LW, label="passive")
    ax.set(ylim=(0, 1.2)); ticks(ax, xt); ax.set_xlabel(xl, fontsize=FSL); tag(ax, letter, x=0.03, y=0.97)
ax = axs[1, 1]
S, real, ideal, chi0 = strain()
curve(ax, S, ideal, BLUE, "sq", label="ideal ports", scale=0.7)
curve(ax, S, real, RED, "circ", label="real ports", scale=0.7)
curve(ax, S, chi0, GREEN, "x", label=r"ideal, $\chi=0$", scale=0.7)
ax.axhline(PAS, color=BLACK, lw=LW)
ax.set(xlim=(0, 1.05), ylim=(0, 1.2)); ticks(ax, [0, .5, 1]); ax.set_xlabel(r"$S$", fontsize=FSL); tag(ax, "d", x=0.03, y=0.97)
ax.legend(fontsize=FS - 1, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.0), handlelength=1.3, borderpad=0.2, labelspacing=0.25)
for a in axs[:, 0]: a.set_ylabel(r"$F_h$", fontsize=FSL, labelpad=1)
h, l = axs[0, 1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=FS, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.0)
fig.tight_layout(h_pad=0.9, w_pad=0.8)
save(fig, "figM7_ports_noise")
