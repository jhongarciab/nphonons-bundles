# -*- coding: utf-8 -*-
"""
Composite figure (3 panels): (a) Wigner snapshots (left, 2×4 + colorbar),
(b) Fidelity (right-top), (c) Mean phonon number (right-bottom).

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, ticker
from qutip import *
from qutip.metrics import fidelity as state_fidelity

# ------------------------------------------------------------
# 0) Simulation settings (tweak N/timesteps if need speed)
# ------------------------------------------------------------
tau_max   = 39.0            # simulation window in units of kappa^{-1}
n_steps   = 120              # time resolution
tau       = np.linspace(0, tau_max, n_steps)
options   = Options(nsteps=1_000_000, store_states=True)

# Wigner grid
xlim   = 4.0
nx     = 400
xvec   = np.linspace(-xlim, xlim, nx)

# ------------------------------------------------------------
# 1) EFFECTIVE MODEL (oscillator only)
# ------------------------------------------------------------
N_eff     = 50
a         = destroy(N_eff)
adag      = a.dag()

# physical parameters (Hz)
gz        = 2*np.pi*6e6          # longitudinal coupling
r         = 0.1
gx        = r*gz                 # transverse coupling
omega_m   = 2*np.pi*100e6        # mechanical frequency
omega_q   = 2*omega_m            # resonance
kappa     = 2*np.pi*100e3        # qubit decay rate
gamma_m   = 2*np.pi*15           # mech. damping
n_th      = 0

g         = gz*gx/omega_m        # two-phonon coupling
ϵ         =  2 * g
chi       = -2j * ϵ * g / kappa

x         = kappa/2
D1m, D1p  = omega_q - omega_m,  omega_q + omega_m
D2m, D2p  = omega_q - 2*omega_m, omega_q + 2*omega_m

# single-phonon rates
G1m = 2*gx**2 * x / (x**2 + D1m**2)
G1p = 2*gx**2 * x / (x**2 + D1p**2)
Gp  =  n_th      *gamma_m + G1p

# two-phonon rates
G2m = 2*g**2 * x / (x**2 + D2m**2)
G2p = 2*g**2 * x / (x**2 + D2p**2)

# Kerr shift
dk  = g**2 * (-D2m/(x**2+D2m**2) - D2p/(x**2+D2p**2))

H_eff = chi.conjugate()*adag**2 + chi*a**2 + dk*(adag*a)**2
diss_eff = [G1m*lindblad_dissipator(a),
            G1p*lindblad_dissipator(adag),
            G2m*lindblad_dissipator(a**2),
            G2p*lindblad_dissipator(adag**2)]

rho0_eff = thermal_dm(N_eff, 0)
tlist_eff = tau / kappa
res_eff   = mesolve(H_eff, rho0_eff, tlist_eff, diss_eff, [adag*a], options=options)
nb_eff    = res_eff.expect[0]
states_eff = res_eff.states

# ------------------------------------------------------------
# 2) FULL MODEL (qubit + oscillator) — all in kappa units
# ------------------------------------------------------------
Na, Nb = 2, 50
b      = tensor(qeye(Na), destroy(Nb))
bd     = b.dag()

gz_org, omega_m_org, Gamma_m_org, kappa_org = 2*np.pi*6e6, 2*np.pi*100e6, 2*np.pi*15, 2*np.pi*100e3
gz_d, om_m, Gam_m = gz_org/kappa_org, omega_m_org/kappa_org, Gamma_m_org/kappa_org
kap = 1.0 # kappa/kappa

gx_dim = r*gz_d
g_dim = gz_d*gx_dim/om_m
Omega = 4 * g_dim

sm, sp, sz, sx = tensor(sigmam(), qeye(Nb)), tensor(sigmap(), qeye(Nb)), tensor(sigmaz(), qeye(Nb)), tensor(sigmax(), qeye(Nb))
omega_q_dim = 2*om_m
omega_d_dim = 2*om_m

H0 = 0.5 * omega_q_dim * sz
H = [
    H0,
    [gx_dim * sx * b,      lambda t, _: np.exp(-1j * om_m * t)],
    [gx_dim * sx * bd,     lambda t, _: np.exp(+1j * om_m * t)],
    [gz_d   * sz * b,      lambda t, _: np.exp(-1j * om_m * t)],
    [gz_d   * sz * bd,     lambda t, _: np.exp(+1j * om_m * t)],
    [Omega  * sx,          lambda t, _: np.exp(+1j * omega_d_dim * t)],
    [Omega  * sx,          lambda t, _: np.exp(-1j * omega_d_dim * t)],
]

diss_full = [kap*lindblad_dissipator(sm),
             (n_th + 1) * Gam_m * lindblad_dissipator(b),
             n_th       * Gam_m * lindblad_dissipator(bd)]

rho0_full  = tensor(thermal_dm(Na, 0), thermal_dm(Nb, 0))
tlist_full = tau
res_full   = mesolve(H, rho0_full, tlist_full, diss_full, [bd*b], options=options)
nb_full    = res_full.expect[0]
states_full = res_full.states
states_full_red = [ptrace(rho, 1) for rho in states_full]

# Fidelity vs time
fidelity_t = np.array([state_fidelity(states_full_red[i], states_eff[i]) for i in range(len(tau))])

# ------------------------------------------------------------
# 3) Wigner snapshots (choose four times)
# ------------------------------------------------------------
i0 = 0
i1 = int(np.clip(np.searchsorted(tau, 1.0), 1, len(tau)-2))
i2 = int(round(0.5 * (len(tau) - 1)))
i3 = len(tau) - 1
slots = [i0, i1, i2, i3]

W_full_list, W_eff_list = [], []
for idx in slots:
    W_full_list.append(wigner(states_full_red[idx], xvec, xvec))
    W_eff_list.append(wigner(states_eff[idx],       xvec, xvec))

# common color scale
vmax = max(np.max(np.abs(W)) for W in (W_full_list + W_eff_list))
vmin = -vmax

# ------------------------------------------------------------
# 4) Composite figure: left (Wigner 2×4), right (fidelity, <n>)
# ------------------------------------------------------------

fig = plt.figure(figsize=(6.9, 3.8), dpi=450)
gs  = fig.add_gridspec(2, 2, width_ratios=[1.75, 1.0], height_ratios=[1, 1], wspace=0.28, hspace=0.04)

# ---- Left column: a 2×4 grid + slim colorbar (all within left cell)
#     (smaller hspace + slightly shorter colorbar row to enlarge Wigners)
left = gs[:, 0].subgridspec(3, 4, height_ratios=[1.0, 1.0, 0.10], wspace=0.06, hspace=0.0)
axes_top  = [fig.add_subplot(left[0, c]) for c in range(4)]
axes_bot  = [fig.add_subplot(left[1, c]) for c in range(4)]
cax       = fig.add_subplot(left[2, :])

def style_wigner(ax, show_x=False, show_y=False, title=None):
    ax.set_xticks([-4, 0, 4]); ax.set_yticks([-4, 0, 4])
    ax.tick_params(labelsize=7, length=3, width=0.6)
    if show_x: ax.set_xlabel(r"$x$", fontsize=10, labelpad=0)  # tighter labelpad
    else:      ax.set_xticklabels([])
    if show_y: ax.set_ylabel(r"$p$", fontsize=10, labelpad=0)
    else:      ax.set_yticklabels([])
    if title:  ax.set_title(title, fontsize=8, pad=1)         # tighter title pad
    for s in ['top','bottom','left','right']:
        ax.spines[s].set_linewidth(0.6)

im = None
for c, idx in enumerate(slots):
    im = axes_top[c].imshow(W_full_list[c], extent=[-xlim, xlim, -xlim, xlim],
                            origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax,
                            interpolation='none', rasterized=True)
    axes_top[c].set_xlim(-xlim, xlim); axes_top[c].set_ylim(-xlim, xlim)
    style_wigner(axes_top[c], show_x=False, show_y=(c==0),
                 title=rf"$\kappa t={tau[idx]:.2f}$")

for c, idx in enumerate(slots):
    axes_bot[c].imshow(W_eff_list[c], extent=[-xlim, xlim, -xlim, xlim],
                       origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax,
                       interpolation='none', rasterized=True)
    axes_bot[c].set_xlim(-xlim, xlim); axes_bot[c].set_ylim(-xlim, xlim)
    style_wigner(axes_bot[c], show_x=True, show_y=(c==0))

# (2) Remove the +4 tick on x for the first three bottom Wigners only
for c in range(3):
    axes_bot[c].set_xticks([-4, 0])   # keep -4 and 0; drop +4

# Unified colorbar
cb = fig.colorbar(im, cax=cax, orientation='horizontal')
ticks = np.linspace(vmin, vmax, 5)
cb.set_ticks(ticks)
cb.ax.set_xticklabels([f"{t:.2f}" for t in ticks])
cb.ax.tick_params(labelsize=7, length=2, width=0.5)

# ---- Right column: top = fidelity, bottom = <n> (share x)
ax_fid = fig.add_subplot(gs[0, 1])
ax_n   = fig.add_subplot(gs[1, 1], sharex=ax_fid)

# Fidelity
ax_fid.plot(tau, fidelity_t, '-', lw=1.2, color='tab:blue')
ax_fid.set_ylabel('Fidelity', fontsize=10)
ax_fid.legend(fontsize=7, frameon=False, loc='lower right')
ax_fid.grid(True, alpha=0.12)

# Mean phonon number
ax_n.plot(tau, nb_full, '-',  lw=1.2, color='black', label='Full')
ax_n.plot(tau, nb_eff,  '--',  lw=1.2, color='red',   label='Effective')
ax_n.set_xlabel(r'$\kappa t$', fontsize=10)
ax_n.set_ylabel(r'$\langle n \rangle$', fontsize=10)
ax_n.legend(fontsize=7, frameon=False, loc='lower right')
ax_n.grid(True, alpha=0.12)

# (1) Remove panel labels (no calls to ax.text/axes_top[...] now)
for ax in (ax_fid, ax_n):
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=7, length=3.0, width=0.6,
                   direction='in', top=True, right=True)
    ax.tick_params(axis='both', which='minor', labelsize=7, length=2.0, width=0.4,
                   direction='in', top=True, right=True)
    for s in ['top','bottom','left','right']:
        ax.spines[s].set_linewidth(0.6)

fig.tight_layout()
fig.savefig('appendix_full_vs_effective_composite.png', dpi=600, bbox_inches='tight')
fig.savefig('appendix_full_vs_effective_composite.pdf',  dpi=600, bbox_inches='tight')
plt.show()
