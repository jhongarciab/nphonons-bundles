"""
Paneles Super-Rabi

Figura compuesta con tres paneles (a,b,c) para los tres regímenes de oscilaciones tipo Rabi
(incluyendo inset n=3 en cada panel) del modelo 2QD.
"""

import matplotlib
matplotlib.use("pgf")

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FixedLocator, LogFormatterMathtext


# -----------------------------------------------------------------------------
# Configuración general
# -----------------------------------------------------------------------------
RERUN = False  # False para recalcular, True para cargar datos guardados

# -----------------------------------------------------------------------------
# Estilo global de figura
# -----------------------------------------------------------------------------
rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.size": 12,
})


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def fix_main_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(1e1, 1.8e6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.xaxis.set_major_locator(FixedLocator([1e2, 1e4, 1e6]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.tick_params(labelsize=12)


def fix_inset_axis(axins):
    axins.set_xscale("log")
    axins.set_xlim(1e4, 7e7)
    axins.set_ylim(0, 1)
    axins.set_yticks([0, 1])
    axins.xaxis.set_major_locator(FixedLocator([1e5, 1e7]))
    axins.xaxis.set_major_formatter(LogFormatterMathtext())
    axins.tick_params(labelsize=10)


# -----------------------------------------------------------------------------
# Datos
# -----------------------------------------------------------------------------
if not RERUN:
    # Malla temporal común para los tres paneles
    omegab = 1.0

    t = np.logspace(1, 6.3, 9000)
    x = omegab * t

    t_ins = np.logspace(3, 8.3, 9000)
    x_ins = omegab * t_ins

    # -------------------------------------------------------------------------
    # PANEL (a) — RÉGIMEN I (2QD)
    # -------------------------------------------------------------------------
    lambda_omegab = 0.03
    Omega_omegab = 0.003

    n = 2
    Omega_eff = (
        np.sqrt(2) * Omega_omegab / math.sqrt(math.factorial(n))
        * (lambda_omegab / omegab) ** n
    )

    n_ins = 3
    Omega_eff_ins = (
        np.sqrt(2) * Omega_omegab / math.sqrt(math.factorial(n_ins))
        * (lambda_omegab / omegab) ** n_ins
    )

    a_Pnc = np.sin(Omega_eff * t) ** 2
    a_P0v = 1 - a_Pnc
    a_Pnc_ins = np.sin(Omega_eff_ins * t_ins) ** 2
    a_P0v_ins = 1 - a_Pnc_ins

    # -------------------------------------------------------------------------
    # PANEL (b) — RÉGIMEN II (2QD)
    # -------------------------------------------------------------------------
    lambda_omegab = 0.1
    Omega_omegab = 0.003

    n = 2
    Omega_eff = (
        np.sqrt(2) * Omega_omegab / math.sqrt(math.factorial(n))
        * (lambda_omegab / omegab) ** n
        * np.exp(-lambda_omegab**2 / (2 * omegab**2))
    )

    n_ins = 3
    Omega_eff_ins = (
        np.sqrt(2) * Omega_omegab / math.sqrt(math.factorial(n_ins))
        * (lambda_omegab / omegab) ** n_ins
        * np.exp(-lambda_omegab**2 / (2 * omegab**2))
    )

    b_Pnc = np.sin(Omega_eff * t) ** 2
    b_P0v = 1 - b_Pnc
    b_Pnc_ins = np.sin(Omega_eff_ins * t_ins) ** 2
    b_P0v_ins = 1 - b_Pnc_ins

    # -------------------------------------------------------------------------
    # PANEL (c) — RÉGIMEN III (2QD)
    # -------------------------------------------------------------------------
    lambda_omegab = 0.03
    Omega_omegab = 0.5
    J = 0.5

    n = 2
    Delta_n = -np.sqrt((n * omegab) ** 2 - 8 * Omega_omegab**2) - J
    c_minus = np.sqrt(
        4 * Omega_omegab**2 /
        (Delta_n**2 + 8 * Omega_omegab**2 - Delta_n * np.sqrt(Delta_n**2 + 8 * Omega_omegab**2))
    )
    prod = 1.0
    for k in range(1, n):
        prod *= (n * c_minus**2 - k)
    Omega_eff = abs(
        (-1) ** n * np.sqrt(2) * Omega_omegab
        * (lambda_omegab / omegab) ** n
        * prod
        / (math.factorial(n - 1) * math.sqrt(math.factorial(n)))
    )

    n = 3
    Delta_n = -np.sqrt((n * omegab) ** 2 - 8 * Omega_omegab**2) - J
    c_minus = np.sqrt(
        4 * Omega_omegab**2 /
        (Delta_n**2 + 8 * Omega_omegab**2 - Delta_n * np.sqrt(Delta_n**2 + 8 * Omega_omegab**2))
    )
    prod = 1.0
    for k in range(1, n):
        prod *= (n * c_minus**2 - k)
    Omega_eff_ins = abs(
        (-1) ** n * np.sqrt(2) * Omega_omegab
        * (lambda_omegab / omegab) ** n
        * prod
        / (math.factorial(n - 1) * math.sqrt(math.factorial(n)))
    )

    c_Pnc = np.sin(Omega_eff * t) ** 2
    c_P0v = 1 - c_Pnc
    c_Pnc_ins = np.sin(Omega_eff_ins * t_ins) ** 2
    c_P0v_ins = 1 - c_Pnc_ins

    np.savez(
        "results/data/rabi_allreg_data.npz",
        omegab=omegab,
        t=t,
        x=x,
        t_ins=t_ins,
        x_ins=x_ins,
        a_Pnc=a_Pnc,
        a_P0v=a_P0v,
        a_Pnc_ins=a_Pnc_ins,
        a_P0v_ins=a_P0v_ins,
        b_Pnc=b_Pnc,
        b_P0v=b_P0v,
        b_Pnc_ins=b_Pnc_ins,
        b_P0v_ins=b_P0v_ins,
        c_Pnc=c_Pnc,
        c_P0v=c_P0v,
        c_Pnc_ins=c_Pnc_ins,
        c_P0v_ins=c_P0v_ins,
    )

else:
    data = np.load("results/data/rabi_allreg_data.npz")
    omegab = data["omegab"]
    t = data["t"]
    x = data["x"]
    t_ins = data["t_ins"]
    x_ins = data["x_ins"]
    a_Pnc = data["a_Pnc"]
    a_P0v = data["a_P0v"]
    a_Pnc_ins = data["a_Pnc_ins"]
    a_P0v_ins = data["a_P0v_ins"]
    b_Pnc = data["b_Pnc"]
    b_P0v = data["b_P0v"]
    b_Pnc_ins = data["b_Pnc_ins"]
    b_P0v_ins = data["b_P0v_ins"]
    c_Pnc = data["c_Pnc"]
    c_P0v = data["c_P0v"]
    c_Pnc_ins = data["c_Pnc_ins"]
    c_P0v_ins = data["c_P0v_ins"]


# -----------------------------------------------------------------------------
# Lienzo y ejes
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(6.30, 4.00), sharex=True)


# -----------------------------------------------------------------------------
# PANEL (a)
# -----------------------------------------------------------------------------
ax = axes[0]
ax.plot(x, a_P0v, color="black", lw=0.9)
ax.plot(x, a_Pnc, color="blue", lw=0.9)
fix_main_axis(ax)

ax.text(0.4e5, 0.75, r"$P_{0vv}$", color="black", fontsize=12)
ax.text(0.4e5, 0.20, r"$P_{2\Psi_+}$", color="blue", fontsize=12)

axins = inset_axes(
    ax,
    width="33%",
    height="95%",
    bbox_to_anchor=(0.215, 0.30, 0.8, 0.7),
    bbox_transform=ax.transAxes,
    loc="upper left",
)
axins.plot(x_ins, a_P0v_ins, color="black", lw=0.9)
axins.plot(x_ins, a_Pnc_ins, color="green", lw=0.9)
fix_inset_axis(axins)

axins.text(0.8e6, 0.75, r"$P_{0vv}$", color="black", fontsize=10)
axins.text(0.8e6, 0.20, r"$P_{3\Psi_+}$", color="green", fontsize=10)


# -----------------------------------------------------------------------------
# PANEL (b)
# -----------------------------------------------------------------------------
ax = axes[1]
ax.plot(x, b_P0v, color="black", lw=0.9)
ax.plot(x, b_Pnc, color="blue", lw=0.9)
fix_main_axis(ax)
ax.set_ylabel("Poblaciones de los estados del sistema", fontsize=12, labelpad=10)

ax.text(5.3e3, 0.75, r"$P_{\bar{0}vv}$", color="black", fontsize=12)
ax.text(5.3e3, 0.20, r"$P_{\bar{2}\Psi_+}$", color="blue", fontsize=12)

axins = inset_axes(
    ax,
    width="33%",
    height="95%",
    bbox_to_anchor=(0.135,0.30, 0.8, 0.7),
    bbox_transform=ax.transAxes,
    loc="upper left",
)
axins.plot(x_ins, b_P0v_ins, color="black", lw=0.9)
axins.plot(x_ins, b_Pnc_ins, color="green", lw=0.9)
fix_inset_axis(axins)

axins.text(2.4e4, 0.75, r"$P_{\bar{0}vv}$", color="black", fontsize=10)
axins.text(2.4e4, 0.20, r"$P_{\bar{3}\Psi_+}$", color="green", fontsize=10)


# -----------------------------------------------------------------------------
# PANEL (c)
# -----------------------------------------------------------------------------
ax = axes[2]
ax.plot(x, c_P0v, color="black", lw=0.9)
ax.plot(x, c_Pnc, color="blue", lw=0.9)
fix_main_axis(ax)
ax.set_xlabel(r"$\omega_b\,t$", fontsize=12)

ax.text(1.5e3, 0.80, r"$P_{0+}$", color="black", fontsize=12)
ax.text(1.6e3, 0.15, r"$P_{2-}$", color="blue", fontsize=12)

axins = inset_axes(
    ax,
    width="33%",
    height="95%",
    bbox_to_anchor=(0.055, 0.30, 0.8, 0.7),
    bbox_transform=ax.transAxes,
    loc="upper left",
)
axins.plot(x_ins, c_P0v_ins, color="black", lw=0.9)
axins.plot(x_ins, c_Pnc_ins, color="green", lw=0.9)
fix_inset_axis(axins)

axins.text(1.3e4, 0.75, r"$P_{0+}$", color="black", fontsize=10)
axins.text(1.3e4, 0.20, r"$P_{3-}$", color="green", fontsize=10)


# -----------------------------------------------------------------------------
# Etiquetas de panel: (a), (b), (c)
# -----------------------------------------------------------------------------
label_positions = {
    0: (0.004, 0.20),
    1: (0.004, 0.20),
    2: (0.004, 0.20),
}

for idx, ax in enumerate(axes):
    ax.text(
        label_positions[idx][0],
        label_positions[idx][1],
        f'$({chr(96 + idx + 1)})$',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=12,
    )


# -----------------------------------------------------------------------------
# Salida
# -----------------------------------------------------------------------------
plt.tight_layout()
fig.subplots_adjust(hspace=0.16)
plt.savefig("results/oficial/rabi_allreg.pdf", bbox_inches="tight")
plt.savefig("results/oficial/pgf/rabi_allreg.pgf")
plt.close()
print("Imágenes guardadas")
