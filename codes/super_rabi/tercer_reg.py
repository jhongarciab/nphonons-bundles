import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FixedLocator, LogFormatterMathtext

# Parámetros físicos
omegab = 1.0
lambda_omegab = 0.03
Omega_omegab  = 0.8

# n = 2
n = 2

Delta_n = -np.sqrt((n * omegab)**2 - 4 * Omega_omegab**2)
c_minus = np.sqrt(2 * Omega_omegab**2 /
                  (Delta_n**2 + 4*Omega_omegab**2
                   - Delta_n * np.sqrt(Delta_n**2 + 4*Omega_omegab**2)))

prod = 1.0
for k in range(1, n):
    prod *= (n * c_minus**2 - k)

Omega_eff = abs((-1)**n
               * Omega_omegab
               * (lambda_omegab / omegab)**n
               * prod
               / (math.factorial(n - 1) * math.sqrt(math.factorial(n))))

t = np.logspace(1, 6.3, 9000)
x = omegab * t

Pnc = np.sin(Omega_eff * t)**2
P0v = 1.0 - Pnc

# n = 3
n_ins = 3

Delta_n_ins = -np.sqrt((n_ins * omegab)**2 - 4 * Omega_omegab**2)
c_minus_ins = np.sqrt(2 * Omega_omegab**2 /
                      (Delta_n_ins**2 + 4*Omega_omegab**2
                       - Delta_n_ins * np.sqrt(Delta_n_ins**2 + 4*Omega_omegab**2)))

prod_ins = 1.0
for k in range(1, n_ins):
    prod_ins *= (n_ins * c_minus_ins**2 - k)

Omega_eff_ins = abs((-1)**n_ins
                    * Omega_omegab
                    * (lambda_omegab / omegab)**n_ins
                    * prod_ins
                    / (math.factorial(n_ins - 1) * math.sqrt(math.factorial(n_ins))))

t_ins = np.logspace(3, 8.3, 9000)
x_ins = omegab * t_ins

Pnc_ins = np.sin(Omega_eff_ins * t_ins)**2
P0v_ins = 1.0 - Pnc_ins

print(f"Omega_eff(n=2) = {Omega_eff:.6e},  T = {2*np.pi/Omega_eff:.4e}")
print(f"Omega_eff(n=3) = {Omega_eff_ins:.6e},  T = {2*np.pi/Omega_eff_ins:.4e}")

# Estilo
rcParams.update({
    "font.size": 13,
    "text.usetex": True,
    "font.family": "serif"
})

fig, ax = plt.subplots(figsize=(7.0, 2.5))

# Panel principal
ax.plot(x, P0v, color="black", lw=1.6)
ax.plot(x, Pnc, color="red",   lw=1.6)

ax.set_xscale("log")
ax.set_xlim(1e1, 1.8e6)
ax.set_ylim(0.01, 1.01)
ax.set_yticks([0, 1])
ax.set_xlabel(r"$\omega_b\,t$", fontsize=18)

ax.xaxis.set_major_locator(FixedLocator([1e2, 1e4, 1e6]))
ax.xaxis.set_major_formatter(LogFormatterMathtext())

# Etiquetas
ax.text(1.9e3, 0.75, r"$P_{0+}$", color="black")
ax.text(1.9e3, 0.20, r"$P_{2-}$", color="red")

# Inset
axins = inset_axes(
    ax,
    width="40%",
    height="95%",
    bbox_to_anchor=(0.03, 0.25, 0.8, 0.7),
    bbox_transform=ax.transAxes,
    loc="upper left"
)

axins.plot(x_ins, P0v_ins, color="black", lw=1.0)
axins.plot(x_ins, Pnc_ins, color="green", lw=1.0)

axins.set_xscale("log")
axins.set_xlim(1e4, 9e7)
axins.set_ylim(0, 1)
axins.set_yticks([0, 1])

axins.xaxis.set_major_locator(FixedLocator([1e5, 1e7]))
axins.xaxis.set_major_formatter(LogFormatterMathtext())
axins.tick_params(labelsize=12)

# Etiquetas
axins.text(2e4, 0.75, r"$P_{0+}$",  color="black", fontsize=11)
axins.text(2e4, 0.20, r"$P_{3-}$", color="green", fontsize=11)

plt.tight_layout()
plt.savefig("../figs/rabi_tercer_reg.pdf", bbox_inches="tight")
plt.show()