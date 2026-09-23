# -*- coding: utf-8 -*-
"""
Cooling vs g_z for Appendix E:

- Full effective ME Eq. (17) solved with QuTiP steady state.
- Coherent-closure estimate n_ss ≃ 2|chi| / Gamma2_minus = Omega / (2 g)
  with epsilon = Omega / 2 (Omega fixed, independent of g_z).

We use a moderate bare thermal occupation n_th = 5 and scan g_z
to show that n_ss decreases as g_z increases (cooling).
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import destroy, steadystate, expect

# ---------------------------
# Global parameters
# ---------------------------

N = 80                         # oscillator Hilbert space dimension
r = 0.1                        # g_x / g_z
omega_m = 2 * np.pi * 100e6    # mechanical frequency (rad/s)
omega_q = 2 * omega_m          # qubit frequency (two-phonon resonance)
kappa   = 2 * np.pi * 100e3    # qubit linewidth (rad/s)
gamma   = 2 * np.pi * 15.0     # mechanical single-phonon damping (rad/s)
n_th    = 5.0                  # thermal occupation of mechanical bath

# Drive strength: Omega is independent of g_z
Omega_MHz = 0.10               # qubit drive Rabi frequency in MHz
Omega = 2 * np.pi * Omega_MHz * 1e6   # convert to rad/s
epsilon = Omega / 2.0          # epsilon = Omega / 2

# g_z scan (in MHz for plotting)
gz_list_MHz = np.linspace(3.0, 10.0, 21)          # 3, 3.62, ..., 8 MHz
gz_list     = 2 * np.pi * gz_list_MHz * 1e6     # convert to rad/s

# ---------------------------
# Operators
# ---------------------------

a    = destroy(N)
adag = a.dag()
n_op = adag * a

# ---------------------------
# Helper: compute all rates for a given g_z
# ---------------------------

def params_for_gz(gz):
    """
    Given g_z (rad/s), compute:
      chi, delta_k, Gamma_minus_tot, Gamma_plus_tot, Gamma2_minus, Gamma2_plus, g
    all in rad/s.
    """
    gx = r * gz
    g  = (gz * gx) / omega_m   # effective two-phonon coupling

    # Squeezing strength from epsilon = Omega/2
    chi = -2j * epsilon * g / kappa    # = -i Omega g / kappa

    x = kappa / 2.0

    # Detunings
    Delta1_minus = omega_q - omega_m
    Delta1_plus  = omega_q + omega_m
    Delta2_minus = omega_q - 2 * omega_m   # = 0 at exact two-phonon resonance
    Delta2_plus  = omega_q + 2 * omega_m

    # Real parts of response functions
    Re_S1_minus = x / (x**2 + Delta1_minus**2)
    Re_S1_plus  = x / (x**2 + Delta1_plus**2)
    Re_S2_minus = x / (x**2 + Delta2_minus**2)
    Re_S2_plus  = x / (x**2 + Delta2_plus**2)

    # One-phonon rates (qubit-induced)
    Gamma1_minus = 2 * gx**2 * Re_S1_minus
    Gamma1_plus  = 2 * gx**2 * Re_S1_plus

    # Mechanical bath one-phonon rates
    gamma_minus = (n_th + 1.0) * gamma
    gamma_plus  = n_th * gamma

    # Total single-phonon rates appearing in (A1)
    Gamma_minus_tot = Gamma1_minus + gamma_minus
    Gamma_plus_tot  = Gamma1_plus  + gamma_plus

    # Two-phonon rates
    Gamma2_minus = 2 * g**2 * Re_S2_minus
    Gamma2_plus  = 2 * g**2 * Re_S2_plus

    # Imaginary parts for Kerr shift
    Im_S2_minus = -Delta2_minus / (x**2 + Delta2_minus**2)
    Im_S2_plus  = -Delta2_plus  / (x**2 + Delta2_plus**2)
    delta_k = g**2 * (Im_S2_minus + Im_S2_plus)

    return chi, delta_k, Gamma_minus_tot, Gamma_plus_tot, Gamma2_minus, Gamma2_plus, g


# ---------------------------
# Main loop over g_z
# ---------------------------

n_ss_full = []
n_ss_coh  = []   # coherent-closure: n_ss ≃ 2 |chi| / Gamma2_minus

for gz, gz_MHz in zip(gz_list, gz_list_MHz):

    chi, delta_k, Gamma_minus, Gamma_plus, Gamma2_minus, Gamma2_plus, g = params_for_gz(gz)

    # --- Effective Hamiltonian (A2) in a frame without the bare mechanical term ---
    H_eff = chi * a**2 + chi.conjugate() * adag**2 + delta_k * (n_op**2)

    # --- Collapse operators for full ME (A1) ---
    c_ops = [
        np.sqrt(Gamma_minus)  * a,       # (Gamma_1^- + gamma_-) D[a]
        np.sqrt(Gamma_plus)   * adag,    # (Gamma_1^+ + gamma_+) D[a^\dagger]
        np.sqrt(Gamma2_minus) * (a**2),  # Gamma_2^- D[a^2]
        np.sqrt(Gamma2_plus)  * (adag**2)
    ]

    # --- Steady state and mean phonon number from full ME (A1) ---
    rho_ss = steadystate(H_eff, c_ops)
    n_full = expect(n_op, rho_ss)
    n_ss_full.append(n_full)

    # --- Coherent-closure estimate: n_ss ≃ 2|chi| / Gamma2_minus ---
    n_coh = 2.0 * abs(chi) / Gamma2_minus
    n_ss_coh.append(n_coh)

    # Optional check: n_coh ≃ Omega / (2 g)
    n_formula = Omega / (2.0 * g)
    print(f"g_z/2π = {gz_MHz:5.2f} MHz  |  "
          f"n_ss(full) = {n_full:7.4f},  "
          f"n_ss(coh) = {n_coh:7.4f},  "
          f"Omega/(2g) = {n_formula:7.4f}")

n_ss_full = np.array(n_ss_full)
n_ss_coh  = np.array(n_ss_coh)

# ---------------------------
# Styled single-panel plot
# ---------------------------

fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.0), dpi=600)

# Full ME
ax.plot(gz_list_MHz, n_ss_full,
        '-', lw=1.2, marker='o', ms=3,
        label=r'Effective ME')

# Coherent-closure
ax.plot(gz_list_MHz, n_ss_coh,
        '--', lw=1.2, marker='s', ms=3,
        label=r'Cat-state approximation')

# Bare thermal line
#ax.axhline(n_th, linestyle=':', lw=1.0, label=r'Bare thermal $n_{\rm th}$')

# Axis labels
ax.set_xlabel(r'$g_z/2\pi\ \mathrm{(MHz)}$', fontsize=9)
ax.set_ylabel(r'$\bar{n}_{\rm ss}$', fontsize=12)

# Grid and ticks (styled like your reference)
ax.grid(True, which='both', alpha=0.12)
ax.minorticks_on()
ax.tick_params(axis='both', which='major',
               labelsize=7, length=3.0, width=0.6,
               direction='in', top=True, right=True)
ax.tick_params(axis='both', which='minor',
               labelsize=7, length=2.0, width=0.4,
               direction='in', top=True, right=True)

# Thinner spines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_linewidth(0.6)

# Legend
ax.legend(fontsize=7, frameon=False)

fig.tight_layout()

# Save if desired
fig.savefig("cooling_vs_gz.png", dpi=600, bbox_inches="tight")
fig.savefig("cooling_vs_gz.pdf", dpi=600, bbox_inches="tight")

plt.show()


