#!/usr/bin/env python3
"""
Fig. S6 — Correlación de bundles g₂⁽²⁾(τ) para 2QD + Förster
===============================================================

Extensión de la Fig. S6 de Bin et al. (PRL 124, 053601, 2020)
al sistema de molécula excitónica (2 QDs acoplados por Förster).

g₂⁽²⁾(τ) = ⟨b†²(0) b†²(τ) b²(τ) b²(0)⟩ / ⟨b†²b²⟩²

Método: ρ' = b² ρ_ss b†², evolucionar con mesolve,
        g₂⁽²⁾(τ) = Tr[b†²b² ρ'(τ)] / ⟨b†²b²⟩²

Hamiltoniano 2QD (marco rotante):
  H = ω_b b†b + Δ(σ₁†σ₁+σ₂†σ₂) + λ(σ₁†σ₁+σ₂†σ₂)(b+b†)
      + Ω(σ₁+σ₁†+σ₂+σ₂†) + J(σ₁†σ₂+σ₂†σ₁)

Resonancia de 2 fonones (2QD, Régimen III + Lamb + Förster):
  Δ₂ = -√(4ω_b² - 8Ω²) + λ²/ω_b - J

Paneles:
  (a) κ/ω_b = 0.001  — régimen de cavidad alta Q
  (b) κ/ω_b = 0.01   — régimen de cavidad baja Q

Parámetros: Ω/ω_b=0.2, λ/ω_b=0.1, γ_φ/ω_b=0.0004, J/ω_b=0.5

Autor: Jhon S. García B. — Tesis UQ 2025
"""

import matplotlib
matplotlib.use("pgf")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import rcParams

rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.size": 8,
})

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
RERUN = True   # True para cargar datos guardados, False para recalcular

# =============================================================================
# PARÁMETROS FÍSICOS
# =============================================================================
omega_b   = 1.0
Omega     = 0.2
lam       = 0.1
gamma_phi = 0.0004
gamma     = 0.0002
J         = 0.5
Ncut      = 15

# Resonancia de 2 fonones — 2QD (factor 8Ω² por √2-enhancement)
Delta_2ph = -np.sqrt((2 * omega_b)**2 - 8 * Omega**2) + lam**2 / omega_b - J
print(f"Δ₂(2QD) = {Delta_2ph:.5f}")

# Grillas
n_gamma = 60          # Producción: 50-60
n_tau   = 300         # Producción: 300
gamma_arr = np.logspace(-5, 0, n_gamma)

kappa_list = [0.001, 0.01]
tau_max    = 10000.0

# Diccionario para almacenar grilla + resultados
panels = {}
for kap in kappa_list:
    tau_arr = np.linspace(0, tau_max, n_tau)
    TAU, GAMMA = np.meshgrid(tau_arr, gamma_arr)
    LOG_GAMMA  = np.log10(GAMMA)

    panels[kap] = {
        'tau_arr':   tau_arr,
        'TAU':       TAU,
        'GAMMA':     GAMMA,
        'LOG_GAMMA': LOG_GAMMA,
        'g2_map':    None,
    }

# =============================================================================
# CÓMPUTO
# =============================================================================
if not RERUN:
    import qutip as qt
    import time

    # ── Operadores 2QD: QD₁ ⊗ QD₂ ⊗ Fock(Ncut) ────────────────
    b  = qt.destroy(Ncut)
    nb = b.dag() * b
    Ib = qt.qeye(Ncut)
    sm = qt.sigmam()
    sp = sm.dag()
    Iq = qt.qeye(2)

    B   = qt.tensor(Iq, Iq, b)
    NB  = qt.tensor(Iq, Iq, nb)
    SM1 = qt.tensor(sm, Iq, Ib)
    SP1 = SM1.dag()
    SM2 = qt.tensor(Iq, sm, Ib)
    SP2 = SM2.dag()
    PE1 = SP1 * SM1
    PE2 = SP2 * SM2
    NE  = PE1 + PE2     # n̂_e = σ₁†σ₁ + σ₂†σ₂

    # Operador de bundle: A = b², A† = b†²
    A     = B * B
    Adag  = A.dag()
    AdagA = Adag * A    # b†²b²

    # Hamiltoniano 2QD
    H = (omega_b * NB
         + Delta_2ph * NE
         + lam * NE * (B + B.dag())
         + Omega * (SM1 + SP1 + SM2 + SP2)
         + J * (SP1 * SM2 + SP2 * SM1))

    solver_opts = {'nsteps': 100000, 'atol': 1e-12, 'rtol': 1e-10}

    def compute_g2_bundle(gamma_val, kappa_val, tau_arr):
        """
        g₂⁽²⁾(τ) para un (γ, κ) dado.

        Método:
          1. ρ_ss = steadystate
          2. ⟨b†²b²⟩_ss
          3. ρ' = b² ρ_ss b†² (estado condicionado)
          4. Evolucionar ρ' con mesolve
          5. g₂⁽²⁾(τ) = Tr[b†²b² ρ'(τ)] / ⟨b†²b²⟩²
        """
        c_ops = [np.sqrt(kappa_val) * B,
                 np.sqrt(gamma_val) * SM1,
                 np.sqrt(gamma_val) * SM2,
                 np.sqrt(gamma_phi) * PE1,
                 np.sqrt(gamma_phi) * PE2]

        print(f"    → steadystate γ={gamma_val:.2e} κ={kappa_val}...", end='', flush=True)
        rho_ss = qt.steadystate(H, c_ops, method='direct')
        print(f" OK. mesolve...", end='', flush=True)
        b2b2_ss = np.real(qt.expect(AdagA, rho_ss))
        if b2b2_ss < 1e-30:
            return np.ones(len(tau_arr)) * np.nan

        rho_prime = A * rho_ss * Adag
        result = qt.mesolve(H, rho_prime, tau_arr, c_ops,
                            e_ops=[AdagA], options=solver_opts)
        return np.real(result.expect[0]) / b2b2_ss**2

    # ── Loop de cómputo ─────────────────────────────────────────
    for kap in kappa_list:
        p = panels[kap]
        g2_map = np.zeros((n_gamma, n_tau))
        t0 = time.time()
        print(f"\n=== Panel κ={kap} ===")
        for i, gam in enumerate(gamma_arr):
            g2_map[i, :] = compute_g2_bundle(gam, kap, p['tau_arr'])
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_gamma - i - 1)
            pct = 100 * (i + 1) / n_gamma
            print(f"  κ={kap}  [{i+1:3d}/{n_gamma}] {pct:5.1f}%  "
                  f"γ={gam:.2e}  ({elapsed:.0f}s, ETA ~{eta:.0f}s)", flush=True)

        p['g2_map'] = g2_map

    # Guardar datos
    np.savez('results/data/g2_decay_2qds_data.npz',
             gamma_arr=gamma_arr,
             Delta_2ph=Delta_2ph,
             J=J,
             **{f'g2_kap{k}': panels[k]['g2_map'] for k in kappa_list},
             **{f'tau_kap{k}': panels[k]['tau_arr'] for k in kappa_list})
    print("✓ Datos guardados en results/data/g2_decay_2qds_data.npz")

else:
    data = np.load('results/data/g2_decay_2qds_data.npz')
    gamma_arr = data['gamma_arr']

    panels = {}
    for kap in kappa_list:
        tau_arr = data[f'tau_kap{kap}']
        TAU, GAMMA = np.meshgrid(tau_arr, gamma_arr)
        LOG_GAMMA = np.log10(GAMMA)
        panels[kap] = {
            'tau_arr': tau_arr,
            'TAU': TAU,
            'GAMMA': GAMMA,
            'LOG_GAMMA': LOG_GAMMA,
            'g2_map': data[f'g2_kap{kap}'],
        }

# =============================================================================
# VISUALIZACIÓN 3D
# =============================================================================
fig = plt.figure(figsize=(5.50, 2.40))

for col, kap in enumerate(kappa_list):
    ax = fig.add_subplot(1, 2, col + 1, projection='3d')
    p = panels[kap]

    # Fondo transparente
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
    ax.yaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
    ax.zaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
    ax.xaxis._axinfo['grid']['color'] = (0.95, 0.95, 0.95, 0.8)
    ax.yaxis._axinfo['grid']['color'] = (0.95, 0.95, 0.95, 0.8)
    ax.zaxis._axinfo['grid']['color'] = (0.95, 0.95, 0.95, 0.8)

    # log10(g2), clippeado
    with np.errstate(divide='ignore', invalid='ignore'):
        log_g2 = np.log10(np.clip(p['g2_map'], 1e-3, 1e3))

    # Colores
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    colors = plt.cm.jet(norm(log_g2))

    # Surface: X=log(γ), Y=τ — interpolada para suavidad
    from scipy.ndimage import zoom

    LOG_GAM_2D = np.log10(p['GAMMA'])
    TAU_2D     = p['TAU']

    log_g2_smooth  = zoom(log_g2,    (6, 3), order=3)
    LOG_GAM_smooth = zoom(LOG_GAM_2D,(6, 3), order=1)
    TAU_smooth     = zoom(TAU_2D,    (6, 3), order=1)
    colors_smooth  = zoom(colors,    (6, 3, 1), order=1)

    ax.plot_surface(
        LOG_GAM_smooth, TAU_smooth, log_g2_smooth,
        facecolors=colors_smooth, rstride=3, cstride=3,
        shade=False, antialiased=True
    )

    # Eje X: γ/ωb (log)
    ax.set_xlabel(r'$\gamma/\omega_b$', fontsize=9, labelpad=8)
    x_ticks = [0, -2, -4]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([r'$10^{%d}$' % v for v in x_ticks], fontsize=8)
    ax.set_xlim(np.log10(gamma_arr[-1]), np.log10(gamma_arr[0]))

    # Eje Y: ωb·τ
    ax.set_ylabel(r'$\omega_b \tau\;(\times 10^3)$', fontsize=9, labelpad=8)
    y_tick_vals = [0, 2000, 4000, 6000, 8000, 10000]
    ax.set_yticks(y_tick_vals)
    ax.set_yticklabels([r'$0$', r'$2$', r'$4$', r'$6$', r'$8$', r'$10$'],
                       fontsize=8)
    ax.set_ylim(0, p['tau_arr'][-1])

    # Eje Z
    ax.set_zlabel('')
    ax.set_zlim(-2, 2)
    ax.set_zticklabels([r'$-2$', r'$-1$', r'$\phantom{-}0$', r'$\phantom{-}1$', r'$\phantom{-}2$'])
    ax.zaxis.set_tick_params(pad=-2, labelsize=8)
    ax.zaxis._axinfo['juggled'] = (1, 2, 0)
    if col == 0:
        ax.text2D(-0.22, 0.50, r'$g_2^{(2)}(\tau)$',
                  transform=ax.transAxes, fontsize=9,
                  rotation=90, va='center', ha='center')

    # Vista
    ax.view_init(elev=12, azim=-40)
    ax.set_box_aspect([2.4, 2.4, 1.0])

    # ── Título ──────────────────────────────────────────────────
    label = '(a)' if col == 0 else '(b)'
    ax.text2D(0.50, 0.90, label,
              transform=ax.transAxes, fontsize=9,
              ha='center', va='bottom')

    # Colorbar
    sm_cb = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm_cb.set_array([])
    cbar = fig.colorbar(sm_cb, ax=ax, shrink=0.21, pad=0.01, aspect=7)
    cbar.set_ticklabels([r'$-1$', r'$\phantom{-}0$', r'$\phantom{-}1$'], fontsize=8)
    cbar.ax.tick_params(labelsize=8)

print(ax.get_position())
fig.subplots_adjust(left=0.15, right=0.97, top=0.98, bottom=-0.10, wspace=0.20)
plt.savefig("results/oficial/g2_decay_2qds_paper.pdf", bbox_inches='tight')
plt.savefig("results/oficial/pgf/g2_decay_2qds_paper.pgf")
print("✓ Figura guardada: results/oficial/g2_decay_2qds_paper.pdf")
plt.close()