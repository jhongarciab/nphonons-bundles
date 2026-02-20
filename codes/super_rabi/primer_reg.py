import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams

omegab = 1.0
n = 2             
lambda_omegab = 0.03   
Omega_omegab = 0.003

Omega_eff = (Omega_omegab * 
    ((lambda_omegab / omegab)**n) 
    / math.sqrt(math.factorial(n)))

T_t = 2*np.pi / Omega_eff             
t = np.linspace(0.0, 1.2*T_t, 5000)
x = omegab * t
x_label = r"$\omega_b\,t$"

Pnc = np.sin(Omega_eff * t)**2 
P0v = 1.0 - Pnc

rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(9.0, 2.3))

ax.plot(x, P0v, color="black", lw=1.6, label=r"$P_{0v}$")
ax.plot(x, Pnc, color="red",   lw=1.6, label=fr"$P_{{{n}c}}$")

ax.set_xscale("log")
ax.set_xlim(x.min(), x.max())
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel(x_label)
ax.set_ylabel("Poblaciones")
ax.legend(loc="upper right", frameon=False)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins = inset_axes(ax, width="28%", height="90%", loc="upper left", borderpad=1.2)
axins.plot(x, P0v, color="black", lw=1.0)
axins.plot(x, Pnc, color="green",   lw=1.0, alpha=0.9)

xmin_pos = np.min(x[x > 0]) if np.any(x > 0) else x[1]
axins.set_xscale("log")
axins.set_xlim(xmin_pos, x.max())
axins.set_ylim(-0.02, 1.02)

from matplotlib.ticker import ScalarFormatter, LogLocator
axins.xaxis.set_major_formatter(ScalarFormatter())
axins.xaxis.set_major_locator(LogLocator(base=10))
axins.tick_params(labelsize=9)
axins.set_title("(a)", x=0.05, y=0.90, ha="left", va="top", fontsize=11)

plt.tight_layout()
plt.savefig(
    f"../figs/rabi_primer_reg.pdf",
    bbox_inches="tight"
)
plt.show()