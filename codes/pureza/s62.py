#!/usr/bin/env python3
"""
Fig. S6 (c)-(d) — Bin et al. (PRL 124, 053601, 2020)
======================================================

g_2^(3)(τ₁, τ₂) — correlación de tercer orden de 2-phonon bundles.

(c) κ/ω_b = 0.001, γ/ω_b = 0.0006  (Poissonian para κ=0.001)
(d) κ/ω_b = 0.01,  γ/ω_b = 0.009   (Poissonian para κ=0.01)

Definición (Eq. S31b):
  g_2^(3)(τ₁,τ₂) = ⟨A†(0) A†(τ₁) A†(τ₂) A(τ₂) A(τ₁) A(0)⟩ / ⟨A†A⟩³
  con A = b², τ₂ ≥ τ₁ ≥ 0.

Cómputo en dos pasos:
  1. ρ₁(τ₁) = e^{Lτ₁}(A ρ_ss A†)
  2. Para cada τ₁:
       ρ₂ = A ρ₁(τ₁) A†
       Evolucionar ρ₂ por (τ₂ − τ₁):
       g_2^(3)(τ₁,τ₂) = Tr[A†A · e^{L(τ₂−τ₁)}(ρ₂)] / ⟨A†A⟩³

Parámetros: Ω/ω_b=0.2, λ/ω_b=0.1, γ_φ/ω_b=0.0004.
Resonancia de Stokes de 2 fonones.

Autor: Jhon S. García B. — Tesis UQ 2025
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import time

# ─── Flag de control ─────────────────────────────────────────────
COMPUTE_REAL = False   # False → datos dummy para validar visual
# ─────────────────────────────────────────────────────────────────

# =================================================================
# 1. PARÁMETROS
# =================================================================
omega_b   = 1.0
Omega     = 0.2
lam       = 0.1
gamma_phi = 0.0004
Ncut      = 15

# Resonancia de 2 fonones
Delta_2ph = -np.sqrt((2 * omega_b)**2 - 4 * Omega**2) + lam**2 / omega_b

# Configuración de cada panel
panels_cfg = {
    '(c)': {'kappa': 0.001, 'gamma': 0.0006, 'tau_max': 2000},
    '(d)': {'kappa': 0.01,  'gamma': 0.009,  'tau_max': 2000},
}

n_tau = 50   # Producción: 80-100

# Opciones del solver
solver_opts = {'nsteps': 100000, 'atol': 1e-12, 'rtol': 1e-10}


# =================================================================
# 2. CÓMPUTO
# =================================================================
panels = {}

if COMPUTE_REAL:
    b  = qt.destroy(Ncut)
    Ib = qt.qeye(Ncut)
    sm = qt.sigmam()
    Iq = qt.qeye(2)

    B   = qt.tensor(Iq, b)
    NB  = qt.tensor(Iq, b.dag() * b)
    SM  = qt.tensor(sm, Ib)
    SP  = qt.tensor(sm.dag(), Ib)
    PE  = qt.tensor(sm.dag() * sm, Ib)

    A     = B * B
    Adag  = A.dag()
    AdagA = Adag * A

    for label, cfg in panels_cfg.items():
        kap = cfg['kappa']
        gam = cfg['gamma']
        tau_max = cfg['tau_max']
        tau_arr = np.linspace(0, tau_max, n_tau)

        print(f"\n{'='*65}")
        print(f"  Panel {label}: κ={kap}, γ={gam}, τ_max={tau_max}")
        print(f"  Grid: {n_tau} × {n_tau}")
        print(f"{'='*65}")

        H = (omega_b * NB + Delta_2ph * PE
             + lam * PE * (B + B.dag()) + Omega * (SM + SP))
        c_ops = [np.sqrt(kap) * B,
                 np.sqrt(gam) * SM,
                 np.sqrt(gamma_phi) * PE]

        # Steadystate
        rho_ss = qt.steadystate(H, c_ops, method='direct')
        b2b2_ss = np.real(qt.expect(AdagA, rho_ss))
        nbar = np.real(qt.expect(NB, rho_ss))
        print(f"  ⟨n̂⟩ = {nbar:.4e}, ⟨b†²b²⟩ = {b2b2_ss:.4e}")

        # Paso 1: ρ'(0) = A ρ_ss A†, evolucionar hasta cada τ₁
        rho_prime = A * rho_ss * Adag
        result1 = qt.mesolve(H, rho_prime, tau_arr, c_ops,
                             options={**solver_opts, 'store_states': True})

        # Paso 2: para cada τ₁, aplicar segundo colapso y evolucionar
        g3 = np.full((n_tau, n_tau), np.nan)
        t0 = time.time()

        for i in range(n_tau):
            rho_tau1 = result1.states[i]

            # Segundo colapso: ρ₂ = A ρ(τ₁) A†
            rho2 = A * rho_tau1 * Adag

            # Solo τ₂ ≥ τ₁ (causalidad)
            j_start = i  # índice donde τ₂ = τ₁
            dt_arr = tau_arr[j_start:] - tau_arr[i]

            if len(dt_arr) < 2:
                # Último punto: calcular directamente
                val = np.real(qt.expect(AdagA, rho2))
                g3[i, j_start] = val / b2b2_ss**3
                continue

            result2 = qt.mesolve(H, rho2, dt_arr, c_ops,
                                 e_ops=[AdagA], options=solver_opts)
            g3[i, j_start:] = np.real(result2.expect[0]) / b2b2_ss**3

            if (i + 1) % 10 == 0 or i == n_tau - 1:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_tau - i - 1)
                print(f"  τ₁ [{i+1:3d}/{n_tau}]  ({elapsed:.0f}s, ETA ~{eta:.0f}s)")

        panels[label] = {
            'g3': g3, 'tau_arr': tau_arr,
            'kappa': kap, 'gamma': gam,
        }

    # Guardar datos
    np.savez('fig_S6_cd_data.npz',
             **{f'g3_{l}': panels[l]['g3'] for l in panels},
             **{f'tau_{l}': panels[l]['tau_arr'] for l in panels})
    print("\n✓ Datos guardados en fig_S6_cd_data.npz")

else:
    print("⚠ Usando datos DUMMY para validar visual")
    for label, cfg in panels_cfg.items():
        tau_arr = np.linspace(0, cfg['tau_max'], n_tau)
        g3 = np.full((n_tau, n_tau), np.nan)
        kap = cfg['kappa']
        for i in range(n_tau):
            for j in range(i, n_tau):
                dt = tau_arr[j] - tau_arr[i]
                g3[i, j] = 1.0 + 5.0 * np.exp(-kap * tau_arr[i]) * np.exp(-kap * dt)
        panels[label] = {
            'g3': g3, 'tau_arr': tau_arr,
            'kappa': kap, 'gamma': cfg['gamma'],
        }


# =================================================================
# 3. VISUALIZACIÓN 3D
# =================================================================
fig = plt.figure(figsize=(14, 6))

for col, (label, p) in enumerate(panels.items()):
    ax = fig.add_subplot(1, 2, col + 1, projection='3d')

    g3      = p['g3']
    tau_arr = p['tau_arr']
    kap     = p['kappa']

    # ── Fondo transparente ──────────────────────────────────────
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
    ax.yaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
    ax.zaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
    ax.xaxis._axinfo['grid']['color'] = (0.95, 0.95, 0.95, 0.8)
    ax.yaxis._axinfo['grid']['color'] = (0.95, 0.95, 0.95, 0.8)
    ax.zaxis._axinfo['grid']['color'] = (0.95, 0.95, 0.95, 0.8)

    # ── Preparar datos para surface ─────────────────────────────
    # g3[i,j] está definido solo para j ≥ i (τ₂ ≥ τ₁).
    # Para el plot 3D, rellenamos la parte j < i con NaN.
    # Meshgrid: X = τ₁, Y = τ₂
    TAU1, TAU2 = np.meshgrid(tau_arr, tau_arr, indexing='ij')

    # log10(g3), clippeado
    with np.errstate(divide='ignore', invalid='ignore'):
        log_g3 = np.log10(np.clip(g3, 1e-3, 1e3))

    # Reemplazar NaN por un valor bajo para que el surface no se rompa
    log_g3_plot = np.where(np.isnan(log_g3), np.nan, log_g3)

    # Colores
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    # Para NaN, usar color neutro
    log_g3_for_color = np.nan_to_num(log_g3_plot, nan=0.0)
    colors = plt.cm.jet(norm(log_g3_for_color))
    # Hacer transparente la zona τ₂ < τ₁
    mask_invalid = np.isnan(log_g3_plot)
    colors[mask_invalid, 3] = 0.0  # alpha = 0

    # ── Surface ─────────────────────────────────────────────────
    # Reemplazar NaN en Z por 0 para que plot_surface no falle
    log_g3_surface = np.nan_to_num(log_g3_plot, nan=-2.0)

    ax.plot_surface(
        TAU1, TAU2, log_g3_surface,
        facecolors=colors, rstride=1, cstride=1,
        shade=False, antialiased=True
    )

    # ── Líneas dashed τ₁ = 1/κ y τ₂ = 1/κ ─────────────────────
    tau_kappa = 1.0 / kap
    # No dibujamos si 1/κ está fuera del rango
    if tau_kappa <= tau_arr[-1]:
        # τ₁ = 1/κ (línea vertical en el piso)
        ax.plot([tau_kappa, tau_kappa], [0, tau_arr[-1]], [-2, -2],
                'k--', lw=0.8, alpha=0.5)
        # τ₂ = 1/κ (línea horizontal en el piso)
        ax.plot([0, tau_arr[-1]], [tau_kappa, tau_kappa], [-2, -2],
                'k--', lw=0.8, alpha=0.5)

    # ── Ejes ────────────────────────────────────────────────────
    ax.set_xlabel(r'$\omega_b \tau_1$', fontsize=12, labelpad=10)
    ax.set_ylabel(r'$\omega_b \tau_2$', fontsize=12, labelpad=10)
    ax.set_zlabel('')
    ax.set_zlim(-2, 2)
    ax.set_zticks([-2, -1, 0, 1, 2])
    ax.zaxis.set_tick_params(pad=-1)
    ax.zaxis._axinfo['juggled'] = (1, 2, 0)

    # Label z manual
    ax.text2D(-0.12, 0.50, r'$\log_{10}\,g_2^{(3)}(\tau_1, \tau_2)$',
              transform=ax.transAxes, fontsize=11,
              rotation=90, va='center', ha='center')

    ax.set_xlim(0, tau_arr[-1])
    ax.set_ylim(0, tau_arr[-1])

    # ── Vista ───────────────────────────────────────────────────
    ax.view_init(elev=20, azim=-55)
    ax.set_box_aspect([2, 2, 1])

    # ── Título ──────────────────────────────────────────────────
    ax.set_title(rf'{label}', fontsize=13, pad=10)

    # ── Colorbar ────────────────────────────────────────────────
    sm_cb = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm_cb.set_array([])
    cbar = fig.colorbar(sm_cb, ax=ax, shrink=0.55, pad=0.08, aspect=15)
    cbar.set_ticks([-1, 0, 1])

fig.subplots_adjust(wspace=0.25)
plt.savefig("fig_S6_cd.pdf", bbox_inches='tight')
plt.savefig("fig_S6_cd.png", dpi=150, bbox_inches='tight')
print("\n✓ Figuras guardadas: fig_S6_cd.{pdf,png}")
plt.show()