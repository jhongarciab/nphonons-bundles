# -*- coding: utf-8 -*-
"""
6-panel fidelity scans (3x2): F(t*) vs [Omega, Delta2, g_x, g_z, omega_m, gamma_m]

What this script does (in words):
---------------------------------
We have two descriptions of the same physical system:

1. "Full model":
   - A two-level system (qubit) + a harmonic oscillator (mechanical resonator).
   - Time-dependent Hamiltonian H(t) with explicit drives and couplings.
   - Dissipation on the qubit and the oscillator.

2. "Effective model":
   - Only the oscillator appears explicitly.
   - The effect of the qubit and its bath is encoded in
     effective one- and two-phonon Lindblad terms and an
     effective Hamiltonian for the oscillator.

Goal:
-----
For different choices of a parameter (Omega, Delta2, g_x, g_z, omega_m, gamma_m),
we compare:

    ρ_full(t*)  → reduced oscillator state obtained from full qubit+oscillator dynamics
    ρ_eff(t*)   → oscillator state obtained from the effective model

We then compute the quantum state fidelity F(ρ_full(t*), ρ_eff(t*))
and plot how close the two descriptions are.

The final output:
-----------------
- A CSV file with all scan data (detuning_scan_results.csv)
- A NumPy .npz file with arrays (detuning_scan_results.npz)
- A 3x2 figure showing F vs each parameter.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import QuTiP tools:
from qutip import (
    destroy,    # annihilation operator a (or b) for harmonic oscillator
    thermal_dm, # thermal density matrix
    coherent_dm,# coherent state density matrix
    mesolve,    # master equation solver
    tensor,     # tensor product
    qeye,       # identity operator
    sigmam, sigmap, sigmaz, sigmax, # Pauli operators for the qubit
    expect      # expectation value
)
from qutip.metrics import fidelity as state_fidelity
from qutip.solver import Options
import csv, time

# ============================================================
# 0) Global base parameters (units and baseline values)
# ============================================================

# We choose units such that κ (kappa) = 1.0
# This means all frequencies and rates are measured in units of κ.
kap      = 1.0            # cavity/qubit decay rate sets the time unit

# Mechanical (oscillator) frequency:
om_m0    = 120.0          # baseline ω_m / κ  (ω_m >> κ: sideband-resolved)

# Two-phonon detuning: Δ_2 = ω_q - 2 ω_m
Delta20  = 0.0            # baseline two-phonon detuning (resonant case)
om_q0    = 2 * om_m0 + Delta20  # corresponding qubit frequency (not used directly but for clarity)

# Mechanical damping rate (very small compared to κ)
gamma_m0 = 3e-3           # oscillator damping (≪ κ)

# Thermal occupation of the oscillator bath
nth      = 0.0            # cold oscillator bath (T ≈ 0, no thermal phonons)

# -----------------------------------------------------------------
# Coupling strengths (THIS IS WHERE WE INCREASE g_x AND g_z TO 1.0)
# -----------------------------------------------------------------

# Transverse coupling g_x (σ_x-type coupling between qubit and oscillator):
gx0      = 1.0            # strong coupling (previously 0.06)

# Longitudinal coupling g_z (σ_z-type coupling between qubit and oscillator):
gz0      = 1.0            # strong coupling (previously 0.08)

# Drive strength Ω (Rabi amplitude of the qubit drive, in units of κ):
Omega0   = 0.20           # baseline drive

# Initial coherent amplitude of the mechanical resonator:
alpha    = 0.25           # small coherent seed (n̄ ≈ |alpha|^2 = 0.0625)


# Time grid for the time evolution
# -------------------------------
# We simulate from t = 0 to t = t_max and divide it into n_steps points.
t_max    = 20.0           # final time
n_steps  = 101             # number of time points
tlist    = np.linspace(0.0, t_max, n_steps)

# We are particularly interested in the time t_star.
# We'll extract the state at this time index from the solution.
t_star   = 20.0                                      # "comparison" time
idx_star = int(np.argmin(np.abs(tlist - t_star)))    # index closest to t_star


# Solver options:
# ---------------
# nsteps: maximum number of internal steps in the ODE solver
# atol, rtol: absolute and relative tolerances (smaller = more accurate, but slower)
options  = Options(
    nsteps=300000,
    atol=1e-8,
    rtol=1e-7,
    store_states=True      # we want all states, not just final one
)

# Hilbert space dimensions:
# -------------------------
# Na: dimension of qubit Hilbert space = 2 (|g>, |e>)
# Nb: dimension of oscillator Hilbert space = 8 (Fock states |0>,...,|7>)
Na, Nb   = 2, 80   # small dimension to keep simulations fast


# ============================================================
# 1) Utility functions
# ============================================================

def lorentz(delta, kappa=kap):
    """
    Simple Lorentzian function used for effective rates.

    Given detuning Δ (called 'delta' here), we define a Lorentzian:

        L(Δ) = (2x) / (x^2 + Δ^2),

    where x = κ/2 (half-width).

    This roughly mimics how a driven, damped qubit responds to different
    frequencies. We use this to approximate induced cooling/heating rates.
    """
    x = kappa / 2.0
    return (2.0 * x) / (x**2 + delta**2)


def build_full(om_m, Delta2, gx, gz, Omega, gamma_m):
    """
    Build the FULL time-dependent model: H(t), collapse operators, and initial state.

    Inputs:
    -------
    om_m   : mechanical frequency ω_m
    Delta2 : two-phonon detuning Δ_2 = ω_q - 2 ω_m
    gx     : transverse coupling strength g_x
    gz     : longitudinal coupling strength g_z
    Omega  : qubit drive amplitude Ω
    gamma_m: mechanical damping rate γ_m

    Outputs:
    --------
    H_td   : list representation of the time-dependent Hamiltonian for QuTiP
    c_ops  : list of collapse operators (dissipation channels)
    rho0   : initial density matrix of the full system (qubit ⊗ oscillator)
    b      : annihilation operator of the oscillator in the full Hilbert space
    """

    # Oscillator annihilation operator in the full Hilbert space:
    # tensor(I_qubit, b_oscillator)
    b  = tensor(qeye(Na), destroy(Nb))
    bd = b.dag()

    # Qubit operators in the full Hilbert space:
    sm = tensor(sigmam(), qeye(Nb))  # lowering operator |g><e|
    sp = tensor(sigmap(), qeye(Nb))  # raising operator |e><g|
    sz = tensor(sigmaz(), qeye(Nb))  # Pauli z
    sx = tensor(sigmax(), qeye(Nb))  # Pauli x

    # Qubit frequency defined by Δ_2 = ω_q - 2 ω_m:
    om_q = 2 * om_m + Delta2

    # Static part of the Hamiltonian: (ω_q / 2) σ_z
    H0 = 0.5 * om_q * sz

    # Time-dependent Hamiltonian H(t):
    # We express this in QuTiP as a list:
    #
    #   H_td = [H0, [H1, f1(t)], [H2, f2(t)], ...]
    #
    # where each f_i(t) is a (Python) function of time, giving a complex factor.
    H_td = [
        H0,
        # g_x σ_x (b e^{-i ω_m t} + b^† e^{+i ω_m t})
        [gx * sx * b,   lambda t, _: np.exp(-1j * om_m * t)],
        [gx * sx * bd,  lambda t, _: np.exp(+1j * om_m * t)],

        # g_z σ_z (b e^{-i ω_m t} + b^† e^{+i ω_m t})
        [gz * sz * b,   lambda t, _: np.exp(-1j * om_m * t)],
        [gz * sz * bd,  lambda t, _: np.exp(+1j * om_m * t)],

        # Qubit drive Ω σ_x (e^{+i ω_q t} + e^{-i ω_q t})
        [Omega * sx,    lambda t, _: np.exp(+1j * om_q * t)],
        [Omega * sx,    lambda t, _: np.exp(-1j * om_q * t)],
    ]

    # Collapse (dissipative) operators:
    # ---------------------------------
    # 1. Qubit spontaneous emission at rate κ:
    #    c_1 = sqrt(κ) σ_-
    #
    # 2. Mechanical damping:
    #    - For a zero-temperature bath (nth = 0), we only have
    #      c_2 = sqrt(γ_m) b
    #    - For non-zero nth, there would also be heating via b^†.
    c_ops = [
        np.sqrt(kap) * sm,                            # qubit decay
        np.sqrt((nth + 1) * gamma_m) * b,             # oscillator decay (cooling)
        np.sqrt(nth * gamma_m) * bd,                  # oscillator heating (if nth > 0)
    ]

    # Initial state:
    # --------------
    # - Qubit in its ground state: thermal_dm(Na, 0) = |g><g|
    # - Oscillator in a small coherent state |α>
    rho0_qubit = thermal_dm(Na, 0)           # effectively |g><g|
    rho0_osc   = coherent_dm(Nb, alpha)      # |α><α|
    rho0       = tensor(rho0_qubit, rho0_osc)

    return H_td, c_ops, rho0, b


def build_eff(om_m, Delta2, gx, gz, Omega, gamma_m):
    """
    Build the EFFECTIVE oscillator-only model (time-independent).

    Here the qubit and its environment are "integrated out". Their influence
    is captured in:
      - effective one-phonon and two-phonon Lindblad terms,
      - an effective Hamiltonian with squeezing and Kerr nonlinearity.

    Inputs:
    -------
    same as build_full(...)

    Outputs:
    --------
    H_eff : effective Hamiltonian acting only on the oscillator
    c_ops: list of effective collapse operators for the oscillator
    rho0 : initial oscillator state (same small coherent seed as in full model)
    a    : oscillator annihilation operator (oscillator-only Hilbert space)
    """

    # Oscillator operators in its own Hilbert space:
    a  = destroy(Nb)
    ad = a.dag()

    # Qubit frequency defined by Δ_2 = ω_q - 2 ω_m:
    om_q = 2 * om_m + Delta2

    # Detunings for single-phonon processes:
    D1m, D1p = om_q - om_m,  om_q + om_m

    # Detunings for two-phonon processes:
    D2m, D2p = om_q - 2 * om_m, om_q + 2 * om_m

    # Effective two-phonon coupling strength g_2 ≈ g_x g_z / ω_m:
    g2 = (gx * gz) / om_m

    # Induced one-phonon rates (cooling and heating):
    G1m = (gx**2) * lorentz(D1m)                   # cooling via a
    G1p = (gx**2) * lorentz(D1p) + nth * gamma_m   # heating via a^†

    # Induced two-phonon rates:
    G2m = (g2**2) * lorentz(D2m)                   # two-phonon cooling via a^2
    G2p = (g2**2) * lorentz(D2p)                   # two-phonon heating via (a^†)^2

    # Frequency shifts and squeezing from virtual processes:
    # -----------------------------------------------------
    # Kerr shift δ_k ("dk" below) and squeezing χ ("chi" below).
    dk  = (g2**2) * (
        -D2m / ((kap / 2)**2 + D2m**2)
        -D2p / ((kap / 2)**2 + D2p**2)
    )

    chi = -2j * (2 * g2) * g2 / kap

    # Effective Hamiltonian:
    #   H_eff = χ* a^2 + χ a^†2 + δ_k (a^† a)^2
    H_eff  = (chi.conjugate() * ad**2 + chi * a**2) + dk * (ad * a)**2

    # Effective collapse operators (Lindblad terms):
    #   - sqrt(G1m) a
    #   - sqrt(G1p) a^†
    #   - sqrt(G2m) a^2
    #   - sqrt(G2p) (a^†)^2
    c_ops  = [
        np.sqrt(G1m) * a,
        np.sqrt(G1p) * ad,
        np.sqrt(G2m) * (a * a),
        np.sqrt(G2p) * (ad * ad)
    ]

    # Initial state: same coherent seed as in the full model, but now
    # only the oscillator exists.
    rho0   = coherent_dm(Nb, alpha)
    return H_eff, c_ops, rho0, a


def run_one(om_m, Delta2, gx, gz, Omega, gamma_m):
    """
    For a given parameter set (om_m, Delta2, gx, gz, Omega, gamma_m):

    1. Evolve the effective model up to t_star.
    2. Evolve the full model up to t_star.
    3. Take the partial trace of the full state over the qubit,
       to get the oscillator reduced state.
    4. Compute fidelity between:
         ρ_red_full(t*)  and  ρ_eff(t*).
    5. Also compute mean phonon number <n> from both models.

    Returns:
    --------
    F as Python floats.
    """

    # --- Effective model ---
    H_e, c_e, rho0_e, a = build_eff(om_m, Delta2, gx, gz, Omega, gamma_m)
    sol_e = mesolve(H_e, rho0_e, tlist, c_e, [], options=options)
    rho_e_t = sol_e.states[idx_star]          # effective state at t_star
    n_eff   = expect(a.dag() * a, rho_e_t)    # mean phonon number from effective model

    # --- Full model ---
    H_f, c_f, rho0_f, b = build_full(om_m, Delta2, gx, gz, Omega, gamma_m)
    sol_f = mesolve(H_f, rho0_f, tlist, c_f, [], options=options)
    rho_f_t = sol_f.states[idx_star]          # full state at t_star

    # Take partial trace over the qubit to obtain the oscillator state:
    rho_red = rho_f_t.ptrace(1)               # index 0: qubit, 1: oscillator
    n_full  = expect(b.dag() * b, rho_f_t)    # mean phonon number from full model

    # Fidelity between reduced full state and effective state:
    F = state_fidelity(rho_red, rho_e_t)

    return float(F)


# ============================================================
# 2) Define 6 scans (parameter ranges)
# ============================================================

# For each parameter, we specify an array of values to scan over.
# When scanning one parameter, the others are held at their baseline values.

scans = {
    # ----------------------------------------------------------------
    # Omega scan:  from 0.02 to 0.50  (MAXIMUM Ω = 0.5 as requested)
    # ----------------------------------------------------------------
    "Omega":   np.linspace(0.02, 0.5, 25),

    # Two-phonon detuning scan: Δ_2 from -1 to +1
    "Delta2":  np.linspace(-1.0, 1.0, 25),

    # g_x scan: here we scan up to 1.0 (strong coupling region)
    "gx":      np.linspace(0.1, 1.0, 19),

    # g_z scan: similarly scan up to 1.0
    "gz":      np.linspace(0.1, 1.0, 19),

    # ω_m scan: vary mechanical frequency around baseline (still >> κ)
    "omega_m": np.concatenate([
        np.linspace(90.0, 120.0, 7),
        np.linspace(120.0, 160.0, 9)
    ]),

    # γ_m scan: vary oscillator damping from small to moderately larger
    "gamma_m": np.concatenate([
        np.linspace(1e-3, 6e-3, 6),
        np.linspace(8e-3, 3.5e-2, 10)
    ]),
}


# ============================================================
# 3) Run scans (loop over keys and grids)
# ============================================================

# We store the results in a dictionary 'results'.
# For each scan key, we store the x-values, fidelities, n_full, n_eff.

results = {}

for key, grid in scans.items():
    

    # These lists will collect results for this particular scan:
    F_list, n_full_list, n_eff_list = [], [], []
    t0 = time.time()

    for i, val in enumerate(grid, 1):
        # Start by setting ALL parameters to their baseline values:
        om_m    = om_m0
        Delta2  = Delta20
        gx      = gx0
        gz      = gz0
        Omega   = Omega0
        gamma_m = gamma_m0

        # Then overwrite the parameter that we are currently scanning:
        if key == "Omega":
            Omega  = float(val)
        if key == "Delta2":
            Delta2 = float(val)
        if key == "gx":
            gx     = float(val)
        if key == "gz":
            gz     = float(val)
        if key == "omega_m":
            om_m   = float(val)   # note: ω_q always defined via Δ_2 and ω_m
        if key == "gamma_m":
            gamma_m= float(val)

        # Run one full vs effective comparison:
        F = run_one(om_m, Delta2, gx, gz, Omega, gamma_m)

        # Store results:
        F_list.append(F)

    # Put arrays into 'results' under this scan key:
    results[key] = {
        "x":      np.array(grid),
        "F":      np.array(F_list),
    }


# ============================================================
# 4) Save data to disk (CSV + NPZ)
# ============================================================

# CSV file: each row is (scan, x_value, F)
csv_path = "detuning_scan_results.csv"   # same filename as before
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    # Header row:
    w.writerow(["scan", "x_value", "Fidelity"])
    # Data rows:
    for key, pack in results.items():
        for xv, Fv in zip(pack["x"], pack["F"]):
            w.writerow([key, f"{xv:.12g}", f"{Fv:.12f}"])

# NPZ file: store arrays under convenient names
np.savez(
    "detuning_scan_results.npz",
    **{f"{key}_x":      pack["x"]      for key, pack in results.items()},
    **{f"{key}_F":      pack["F"]      for key, pack in results.items()},
)


# ============================================================
# 5) Plot: one figure, six panels (3x2) — fidelity only
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(8.6, 4.8), dpi=200, sharey=True)
axes = axes.ravel()  # flatten 2x3 array into a simple list

# We enforce a fixed order of scans for the 6 panels:
scan_order = ["Omega", "Delta2", "gx", "gz", "omega_m", "gamma_m"]

for ax, key in zip(axes, scan_order):
    pack = results[key]
    ax.plot(pack["x"], pack["F"], "o-", lw=1.1)
    ax.set_ylim(0.98, 1.0005)          # zoom near 1 to highlight deviations
    ax.grid(True, alpha=0.15)

    # We keep the axes unlabeled to match the formatting in the paper.
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.show()


