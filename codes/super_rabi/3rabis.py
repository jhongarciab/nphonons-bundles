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
# Estilo global de figura
# -----------------------------------------------------------------------------
# - pgf.texsystem: motor TeX usado por PGF.
# - text.usetex=False: evita depender del render externo TeX para texto general;
#   PGF sigue exportando correctamente la figura vectorial.
rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.size": 12,
})


# -----------------------------------------------------------------------------
# Malla temporal común para los tres paneles
# -----------------------------------------------------------------------------
omegab = 1.0

t = np.logspace(1, 6.3, 9000)      # dominio principal
x = omegab * t                      # variable adimensional del eje x

t_ins = np.logspace(3, 8.3, 9000)   # dominio para inset (más largo)
x_ins = omegab * t_ins


# -----------------------------------------------------------------------------
# Lienzo y ejes
# -----------------------------------------------------------------------------
TEXTWIDTH_IN = (16.5 / 2.54) * 0.95
fig, axes = plt.subplots(3, 1, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.65), sharex=True)
fig.subplots_adjust(hspace=0.08)


# -----------------------------------------------------------------------------
# Helpers de formato
# -----------------------------------------------------------------------------
def fix_main_axis(ax):
    """Formato común para el eje principal de cada panel."""
    ax.set_xscale("log")
    ax.set_xlim(1e1, 1.8e6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.xaxis.set_major_locator(FixedLocator([1e2, 1e4, 1e6]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.tick_params(labelsize=11)


def fix_inset_axis(axins):
    """Formato común para el inset (n=3) en cada panel."""
    axins.set_xscale("log")
    axins.set_xlim(1e4, 7e7)
    axins.set_ylim(0, 1)
    axins.set_yticks([0, 1])
    axins.xaxis.set_major_locator(FixedLocator([1e5, 1e7]))
    axins.xaxis.set_major_formatter(LogFormatterMathtext())
    axins.tick_params(labelsize=9)


# -----------------------------------------------------------------------------
# PANEL (a) — RÉGIMEN I (2QD)
# -----------------------------------------------------------------------------
lambda_omegab = 0.03
Omega_omegab = 0.003

# n = 2 (panel principal)
n = 2
Omega_eff = (
    np.sqrt(2) * Omega_omegab / math.sqrt(math.factorial(n))
    * (lambda_omegab / omegab) ** n
)

# n = 3 (inset)
n_ins = 3
Omega_eff_ins = (
    np.sqrt(2) * Omega_omegab / math.sqrt(math.factorial(n_ins))
    * (lambda_omegab / omegab) ** n_ins
)

Pnc = np.sin(Omega_eff * t) ** 2
P0v = 1 - Pnc
Pnc_ins = np.sin(Omega_eff_ins * t_ins) ** 2
P0v_ins = 1 - Pnc_ins

ax = axes[0]
ax.plot(x, P0v, color="black", lw=0.9)
ax.plot(x, Pnc, color="blue", lw=0.9)
fix_main_axis(ax)

ax.text(0.6e5, 0.75, r"$P_{0vv}$", color="black", fontsize=11)
ax.text(0.6e5, 0.20, r"$P_{2\Psi_+}$", color="blue", fontsize=11)

axins = inset_axes(
    ax,
    width="33%",
    height="95%",
    bbox_to_anchor=(0.16, 0.30, 0.8, 0.7),
    bbox_transform=ax.transAxes,
    loc="upper left",
)
axins.plot(x_ins, P0v_ins, color="black", lw=0.9)
axins.plot(x_ins, Pnc_ins, color="green", lw=0.9)
fix_inset_axis(axins)

axins.text(1.3e6, 0.75, r"$P_{0vv}$", color="black", fontsize=9)
axins.text(1.3e6, 0.20, r"$P_{3\Psi_+}$", color="green", fontsize=9)


# -----------------------------------------------------------------------------
# PANEL (b) — RÉGIMEN II (2QD)
# -----------------------------------------------------------------------------
lambda_omegab = 0.1
Omega_omegab = 0.003

# n = 2 (panel principal)
n = 2
Omega_eff = (
    np.sqrt(2) * Omega_omegab / math.sqrt(math.factorial(n))
    * (lambda_omegab / omegab) ** n
    * np.exp(-lambda_omegab**2 / (2 * omegab**2))
)

# n = 3 (inset)
n_ins = 3
Omega_eff_ins = (
    np.sqrt(2) * Omega_omegab / math.sqrt(math.factorial(n_ins))
    * (lambda_omegab / omegab) ** n_ins
    * np.exp(-lambda_omegab**2 / (2 * omegab**2))
)

Pnc = np.sin(Omega_eff * t) ** 2
P0v = 1 - Pnc
Pnc_ins = np.sin(Omega_eff_ins * t_ins) ** 2
P0v_ins = 1 - Pnc_ins

ax = axes[1]
ax.plot(x, P0v, color="black", lw=0.9)
ax.plot(x, Pnc, color="blue", lw=0.9)
fix_main_axis(ax)
ax.set_ylabel("Poblaciones de los estados del sistema", fontsize=12, labelpad=10)

ax.text(6e3, 0.75, r"$P_{0vv}$", color="black", fontsize=11)
ax.text(6e3, 0.20, r"$P_{2\Psi_+}$", color="blue", fontsize=11)

axins = inset_axes(
    ax,
    width="33%",
    height="95%",
    bbox_to_anchor=(0.11, 0.30, 0.8, 0.7),
    bbox_transform=ax.transAxes,
    loc="upper left",
)
axins.plot(x_ins, P0v_ins, color="black", lw=0.9)
axins.plot(x_ins, Pnc_ins, color="green", lw=0.9)
fix_inset_axis(axins)

axins.text(3e4, 0.75, r"$P_{0vv}$", color="black", fontsize=9)
axins.text(3e4, 0.20, r"$P_{3\Psi_+}$", color="green", fontsize=9)


# -----------------------------------------------------------------------------
# PANEL (c) — RÉGIMEN III (2QD)
# -----------------------------------------------------------------------------
lambda_omegab = 0.03
Omega_omegab = 0.5
J = 0.5  # acoplamiento Förster (J=0 recupera benchmark sin acoplamiento)

# n = 2 (panel principal)
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

# n = 3 (inset)
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

Pnc = np.sin(Omega_eff * t) ** 2
P0v = 1 - Pnc
Pnc_ins = np.sin(Omega_eff_ins * t_ins) ** 2
P0v_ins = 1 - Pnc_ins

ax = axes[2]
ax.plot(x, P0v, color="black", lw=0.9)
ax.plot(x, Pnc, color="blue", lw=0.9)
fix_main_axis(ax)
ax.set_xlabel(r"$\omega_b\,t$", fontsize=12)

ax.text(1.6e3, 0.75, r"$P_{0+}$", color="black", fontsize=11)
ax.text(1.6e3, 0.20, r"$P_{2-}$", color="blue", fontsize=11)

axins = inset_axes(
    ax,
    width="33%",
    height="95%",
    bbox_to_anchor=(0.06, 0.30, 0.8, 0.7),
    bbox_transform=ax.transAxes,
    loc="upper left",
)
axins.plot(x_ins, P0v_ins, color="black", lw=0.9)
axins.plot(x_ins, Pnc_ins, color="green", lw=0.9)
fix_inset_axis(axins)

axins.text(1.8e4, 0.75, r"$P_{0+}$", color="black", fontsize=9)
axins.text(1.8e4, 0.20, r"$P_{3-}$", color="green", fontsize=9)


# -----------------------------------------------------------------------------
# Etiquetas de panel: (a), (b), (c)
# -----------------------------------------------------------------------------
label_positions = {
    0: (0.005, 0.95),
    1: (0.005, 0.95),
    2: (0.005, 0.95),
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
fig.subplots_adjust(hspace=0.12)
#plt.show() 
plt.savefig("./figs/oficial/rabi_allreg.pdf", bbox_inches="tight")
plt.savefig("./figs/oficial/pgf/rabi_allreg.pgf")
plt.close()
