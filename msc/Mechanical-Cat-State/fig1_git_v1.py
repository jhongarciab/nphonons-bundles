# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 12:39:21 2025

@author: Tahir Naseem
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# ---------------------------
# System Setup
# ---------------------------

# Hilbert space dimension
N = 60

# System parameters (Hz)
gz  = 2 * np.pi * 6e6         # Longitudinal coupling
r   = 0.1                     # Transverse-to-longitudinal coupling ratio
gx  = r * gz                  # Transverse coupling
ωₘ  = 2 * np.pi * 100e6       # Mechanical frequency
ωq  = 2 * ωₘ                  # Qubit frequency (resonance condition for squeezing)
κ   = 2 * np.pi * 100e3       # Qubit decay rate
γ   = 2 * np.pi * 15          # Mechanical damping rate

# Effective couplings
g = (gz * gx) / ωₘ                  # Two-phonon coupling


# Drive parameters
ϵ  = 4 * g
χ = -2j * ϵ * g / κ                    # Squeezing drive strength

# Thermal occupation
n_th = 0

# ---------------------------
# Lindblad Rates
# ---------------------------

x = κ / 2

# Detunings
Δ1_minus = ωq - ωₘ
Δ1_plus = ωq + ωₘ
Δ2_minus = ωq - 2 * ωₘ
Δ2_plus  = ωq + 2 * ωₘ

# Single-phonon processes

Re_S1_minus = x / (x**2 + Δ1_minus**2)
Γ1_minus = 2 * gx**2 * Re_S1_minus
Γ_minus = (n_th + 1) * γ + Γ1_minus

Re_S1_plus = x / (x**2 + Δ1_minus**2)
Γ1_plus = 2 * gx**2 * Re_S1_plus
Γ_plus = n_th * γ + Γ1_plus

# Two-phonon processes
Re_S2_minus = x / (x**2 + Δ2_minus**2)
Γ2_minus = 2 * g**2 * Re_S2_minus

Re_S2_plus = x / (x**2 + Δ2_plus**2)
Γ2_plus = 2 * g**2 * Re_S2_plus


# Imaginary parts of S_{2±}
Im_S2_minus = -Δ2_minus / (x**2 + Δ2_minus**2)
Im_S2_plus  = -Δ2_plus / (x**2 + Δ2_plus**2)

# Kerr shift
δ_k = g**2 * (Im_S2_minus + Im_S2_plus)

# ---------------------------
# Operators and Hamiltonian
# ---------------------------

a = destroy(N)
ad = a.dag()

# Effective Hamiltonian
H = χ.conjugate() * ad**2 + χ * a**2 + δ_k * (ad * a)**2

# Dissipators
dissipators = [
    Γ_minus  * lindblad_dissipator(a),
    Γ_plus   * lindblad_dissipator(ad),
    Γ2_minus * lindblad_dissipator(a**2),
    Γ2_plus  * lindblad_dissipator(ad**2),
]

# ---------------------------
# Initial state and time evolution
# ---------------------------

ρ0 = thermal_dm(N, n_th)
tlist = np.linspace(0, 3, 31)/Γ2_minus
print(tlist*Γ2_minus)

options = Options(nsteps=100000)

result = mesolve(H, ρ0, tlist, dissipators, [], options=options)

# ---------------------------
# Wigner plots
# ---------------------------

xvec = np.linspace(-4, 4, 500)

for j, state in enumerate(result.states):
    W = wigner(state, xvec, xvec)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=600)

    # Symmetric color limits for Wigner negativity
    vmax = np.max(np.abs(W))
    vmin = -vmax

    im = ax.imshow(W, extent=[-4, 4, -4, 4], origin='lower',
                   cmap='RdBu_r', vmin=vmin, vmax=vmax)

    # Axis labels and ticks
    ax.set_xlabel(r'$x$', fontsize=26)
    ax.set_ylabel(r'$p$', fontsize=26)
    ax.set_xticks([-4, 0, 4])
    ax.set_yticks([-4, 0, 4])
    ax.tick_params(axis='both', labelsize=18)

    # Colorbar: 5 ticks with 1-digit precision
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in ticks])
    cbar.ax.tick_params(labelsize=14)

    # Save and display
    fig.savefig(f"wigner_panel_{j+1}.pdf", bbox_inches='tight')
    plt.show()



