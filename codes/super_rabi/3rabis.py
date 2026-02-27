import matplotlib
matplotlib.use("pgf")

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FixedLocator, LogFormatterMathtext

rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": False,
    "pgf.rcfonts": False,
    "font.size": 13,
})

omegab = 1.0
t     = np.logspace(1, 6.3, 9000)
t_ins = np.logspace(3, 8.3, 9000)
x     = omegab * t
x_ins = omegab * t_ins

fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
fig.subplots_adjust(hspace=0.08)

# =========================
# FUNCIÓN EJE PRINCIPAL
# =========================
def fix_main_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(1e1, 1.8e6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.xaxis.set_major_locator(FixedLocator([1e2, 1e4, 1e6]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.tick_params(labelsize=15)

# =========================
# FUNCIÓN INSET
# =========================
def fix_inset_axis(axins):
    axins.set_xscale("log")
    axins.set_xlim(1e4, 7e7)
    axins.set_ylim(0, 1)
    axins.set_yticks([0, 1])
    axins.xaxis.set_major_locator(FixedLocator([1e5, 1e7]))
    axins.xaxis.set_major_formatter(LogFormatterMathtext())
    axins.tick_params(labelsize=16)

# =====================================================
# PANEL (a) — RÉGIMEN I (2QD)
# =====================================================
lambda_omegab = 0.03
Omega_omegab  = 0.003

n = 2
Omega_eff = (np.sqrt(2)*Omega_omegab/ math.sqrt(math.factorial(n))
             *(lambda_omegab/omegab)**n)

n_ins = 3
Omega_eff_ins = (np.sqrt(2)*Omega_omegab/ math.sqrt(math.factorial(n_ins))
                 *(lambda_omegab/omegab)**n_ins)

Pnc = np.sin(Omega_eff*t)**2
P0v = 1 - Pnc
Pnc_ins = np.sin(Omega_eff_ins*t_ins)**2
P0v_ins = 1 - Pnc_ins

ax = axes[0]
ax.plot(x, P0v, color="black", lw=1.6)
ax.plot(x, Pnc, color="blue", lw=1.6)
fix_main_axis(ax)

# Labels internos
ax.text(0.8e5, 0.75, r"$P_{0v}$", color="black", fontsize=16)
ax.text(0.8e5, 0.20, r"$P_{2c}$", color="blue", fontsize=16)

# Inset
axins = inset_axes(ax, width="35%", height="95%",
                   bbox_to_anchor=(0.1,0.25,0.8,0.7),
                   bbox_transform=ax.transAxes, loc="upper left")
axins.plot(x_ins, P0v_ins, color="black", lw=1.0)
axins.plot(x_ins, Pnc_ins, color="red", lw=1.0)
fix_inset_axis(axins)

axins.text(3e6, 0.75, r"$P_{0v}$", color="black", fontsize=16)
axins.text(3e6, 0.20, r"$P_{3c}$", color="red", fontsize=16)

# =====================================================
# PANEL (b) — RÉGIMEN II (2QD)
# =====================================================
lambda_omegab = 0.1
Omega_omegab  = 0.003

n = 2
Omega_eff = (np.sqrt(2)*Omega_omegab/ math.sqrt(math.factorial(n))
             *(lambda_omegab/omegab)**n
             * np.exp(-lambda_omegab**2/(2*omegab**2)))

n_ins = 3
Omega_eff_ins = (np.sqrt(2)*Omega_omegab/ math.sqrt(math.factorial(n_ins))
                 *(lambda_omegab/omegab)**n_ins
                 * np.exp(-lambda_omegab**2/(2*omegab**2)))

Pnc = np.sin(Omega_eff*t)**2
P0v = 1 - Pnc
Pnc_ins = np.sin(Omega_eff_ins*t_ins)**2
P0v_ins = 1 - Pnc_ins

ax = axes[1]
ax.plot(x, P0v, color="black", lw=1.6)
ax.plot(x, Pnc, color="blue", lw=1.6)
fix_main_axis(ax)

ax.text(0.8e4, 0.75, r"$P_{0v}$", color="black", fontsize=16)
ax.text(0.8e4, 0.20, r"$P_{2c}$", color="blue", fontsize=16)

axins = inset_axes(ax, width="35%", height="95%",
                   bbox_to_anchor=(0.08,0.25,0.8,0.7),
                   bbox_transform=ax.transAxes, loc="upper left")
axins.plot(x_ins, P0v_ins, color="black", lw=1.0)
axins.plot(x_ins, Pnc_ins, color="red", lw=1.0)
fix_inset_axis(axins)

axins.text(0.8e5, 0.75, r"$P_{0v}$", color="black", fontsize=16)
axins.text(0.8e5, 0.20, r"$P_{3c}$", color="red", fontsize=16)

# =====================================================
# PANEL (c) — RÉGIMEN III (2QD)
# =====================================================
lambda_omegab = 0.03
Omega_omegab  = 0.5   # igual que Bin Fig. 2c
J             = 0.5   # acoplamiento Förster (J=0 para benchmark con Bin)

# n=2
n = 2
# Resonancia 2QD régimen III: Delta_n = -sqrt(n²ωb² - 8Ω²) - J
Delta_n = -np.sqrt((n*omegab)**2 - 8*Omega_omegab**2) - J

# c̃₋ con Ω̃ = √2·Ω → factor 8Ω² en denominador
c_minus = np.sqrt(4*Omega_omegab**2 /
                  (Delta_n**2 + 8*Omega_omegab**2
                   - Delta_n*np.sqrt(Delta_n**2 + 8*Omega_omegab**2)))

prod = 1.0
for k in range(1, n):
    prod *= (n * c_minus**2 - k)

Omega_eff = abs((-1)**n * np.sqrt(2) * Omega_omegab
               * (lambda_omegab/omegab)**n
               * prod
               / (math.factorial(n-1) * math.sqrt(math.factorial(n))))

# n=3 (inset)
n = 3
Delta_n = -np.sqrt((n*omegab)**2 - 8*Omega_omegab**2) - J

c_minus = np.sqrt(4*Omega_omegab**2 /
                  (Delta_n**2 + 8*Omega_omegab**2
                   - Delta_n*np.sqrt(Delta_n**2 + 8*Omega_omegab**2)))

prod = 1.0
for k in range(1, n):
    prod *= (n * c_minus**2 - k)

Omega_eff_ins = abs((-1)**n * np.sqrt(2) * Omega_omegab
                   * (lambda_omegab/omegab)**n
                   * prod
                   / (math.factorial(n-1) * math.sqrt(math.factorial(n))))

Pnc = np.sin(Omega_eff*t)**2
P0v = 1 - Pnc
Pnc_ins = np.sin(Omega_eff_ins*t_ins)**2
P0v_ins = 1 - Pnc_ins

ax = axes[2]
ax.plot(x, P0v, color="black", lw=1.6)
ax.plot(x, Pnc, color="blue", lw=1.6)
fix_main_axis(ax)
ax.set_xlabel(r"$\omega_b\,t$", fontsize=24)

ax.text(1.8e3, 0.75, r"$P_{0+}$", color="black", fontsize=16)
ax.text(1.8e3, 0.20, r"$P_{2-}$", color="blue", fontsize=16)

axins = inset_axes(ax, width="35%", height="95%",
                   bbox_to_anchor=(0.06,0.25,0.8,0.7),
                   bbox_transform=ax.transAxes, loc="upper left")
axins.plot(x_ins, P0v_ins, color="black", lw=1.0)
axins.plot(x_ins, Pnc_ins, color="red", lw=1.0)
fix_inset_axis(axins)

axins.text(2e4, 0.75, r"$P_{0+}$",  color="black", fontsize=16)
axins.text(2e4, 0.20, r"$P_{3-}$", color="red", fontsize=16)

label_positions = {
    0: (0.01, 0.95),  # panel (a)
    1: (0.01, 0.95),  # panel (b)
    2: (0.01, 0.95),  # panel (c)
}

for idx, ax in enumerate(axes):
    ax.text(label_positions[idx][0], label_positions[idx][1],
            f'$({chr(96+idx+1)})$',
            transform=ax.transAxes,
            ha='left', va='top', fontsize=16)

plt.tight_layout()
#plt.show()
plt.savefig("../figs/rabi_allreg.pgf")
plt.close()