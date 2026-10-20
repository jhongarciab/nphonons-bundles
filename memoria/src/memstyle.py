"""Shared publication style for the memory figures; PGF is the manuscript source."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"
LW, FS, FSL = 0.9, 8, 9
RED, BLACK, GOLD, BLUE, GREEN = "red", "black", "#c4942a", "#1f77b4", "C2"
CMAP = "RdYlBu_r"

rcParams.update({"pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
                 "font.family": "serif", "font.size": FS, "mathtext.fontset": "stix",
                 "pdf.fonttype": 42, "pgf.preamble": "\\usepackage{amsmath,amssymb}"})

def tag(ax, letter, x=0.03, y=0.95, color="black", ha="left"):
    ax.text(x, y, rf"$({letter})$", transform=ax.transAxes, fontsize=FSL, ha=ha, va="top", color=color)

def label(ax, x, y, s, color, ha="left", va="center", transform=True, **kw):
    ax.text(x, y, s, color=color, fontsize=FS, ha=ha, va=va,
            transform=ax.transAxes if transform else ax.transData, **kw)

def style(ax):
    ax.set_facecolor("white"); ax.grid(False); ax.tick_params(labelsize=FS)

def save(fig, name):
    fig.savefig(FIGS / f"{name}.pgf", format="pgf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGS / f"{name}.pgf")

# Dense-curve style: smooth monotone interpolant through the computed points, open markers of distinct shapes on the points.
import numpy as np
from scipy.interpolate import PchipInterpolator
MK = dict(x=("x", 4.0), plus=("+", 5.0), sq=("s", 3.6), tri=("^", 3.8), circ=("o", 3.6))

def curve(ax, x, y, color, mk, ls="-", label=None, logx=False, n=200, marker_off=False, mask=None, scale=1.0, **kw):
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = np.argsort(x); x, y = x[o], y[o]
    u = np.log(x) if logx else x
    uu = np.linspace(u[0], u[-1], n); yy = PchipInterpolator(u, y)(uu)
    ax.plot(np.exp(uu) if logx else uu, yy, color=color, lw=LW, ls=ls, **kw)
    if marker_off: return
    m, ms = MK[mk]; ms *= scale
    filled = m in ("x", "+")
    if mask is not None: x, y = x[np.asarray(mask)[o]], y[np.asarray(mask)[o]]
    ax.plot(x, y, ls="none", marker=m, ms=ms, mew=0.8, color=color, mec=color, mfc="none" if not filled else color, label=label)


def passive_ref(d, kappa=1e-3, with_tau12=True):
    """Referencia pasiva (F, s, F_h) de dos modos independientes con los mismos puertos, a partir de los campos de un log de F17:
    p1 = eta_L1 eta_R exp[-kappa (tau12 + tL2 + T + tE)], p2 = eta_L2 eta_R exp(-kappa T). Con with_tau12=False reproduce pas_F/pas_s/pas_Fh del log."""
    eL, etaR, T = d["pload"], d["eta_R"], d["T"]
    tL2 = 4.0 / (kappa * d["kin_frac"]); tE = 4.0 / (kappa * d["kout_frac"])
    p1 = eL[0] * etaR * np.exp(-kappa * ((d.get("tau12", 0.0) if with_tau12 else 0.0) + tL2 + T + tE)); p2 = eL[-1] * etaR * np.exp(-kappa * T)
    F = (p1 + p2) / 4 + np.sqrt(p1 * p2) / 2; s = (p1 + p2) / 2
    return F, s, F / s


def passive_sim(T):
    """Referencia pasiva simulada: dinámica abierta de dos modos independientes con los mismos puertos (passive_two_modes.py, tau12=100).
    Devuelve (F, s, F_h) a la espera T leída de data/passive_sim/passive_T{T}.log."""
    import json
    d = json.loads((ROOT / "data" / "passive_sim" / f"passive_T{int(T)}.log").read_text().strip().splitlines()[-1])
    assert d["mineig"] > -1e-9 and abs(d["tr"] - 1) < 1e-9 and d["tau12"] == 100.0
    return d["F"], d["s"], d["F_h"]
