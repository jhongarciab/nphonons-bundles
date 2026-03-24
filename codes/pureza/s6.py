#!/usr/bin/env python3
"""
Scaffold — Fig. S6 (a)-(b), Bin et al. (PRL 124, 053601, 2020)
================================================================

Solo construye la grilla 3D y genera un plot con datos sintéticos
para validar ejes, colorbar, vista y etiquetas ANTES de lanzar
el cómputo real con mesolve.

Uso:
  1) Ejecutar tal cual → genera plot con datos dummy.
  2) Cuando la visualización esté correcta, descomentar la línea
     COMPUTE_REAL = True y ejecutar de nuevo.

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
    "font.size": 12,
})

# -----------------------------------------------------------------------------
# Configuración general
# -----------------------------------------------------------------------------
RERUN = True  # False para recalcular, True para cargar datos guardados

# -----------------------------------------------------------------------------
# 1. PARÁMETROS DE GRILLA (idénticos al cómputo real)
# -----------------------------------------------------------------------------
omega_b   = 1.0
Omega     = 0.2
lam       = 0.1
gamma_phi = 0.0004
Ncut      = 15

# Resonancia de 2 fonones
Delta_2ph = -np.sqrt((2 * omega_b)**2 - 4 * Omega**2) + lam**2 / omega_b

# Grillas
n_gamma = 25          # Producción: 50-60
n_tau   = 200         # Producción: 300
gamma_arr = np.logspace(-5, 0, n_gamma)

kappa_list = [0.001, 0.01]
tau_max    = 10000.0

# Diccionario: almacena grilla + resultados por panel
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
        'g2_map':    None,       # se llena abajo
    }


# -----------------------------------------------------------------------------
# 2. DATOS: dummy o reales
# -----------------------------------------------------------------------------
if not RERUN:
    # ── Importar QuTiP solo cuando se necesita ──────────────────
    import qutip as qt
    import time

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

    solver_opts = {'nsteps': 100000, 'atol': 1e-12, 'rtol': 1e-10}

    def compute_g2_bundle(gamma_val, kappa_val, tau_arr):
        H = (omega_b * NB + Delta_2ph * PE
             + lam * PE * (B + B.dag()) + Omega * (SM + SP))
        c_ops = [np.sqrt(kappa_val) * B,
                 np.sqrt(gamma_val) * SM,
                 np.sqrt(gamma_phi) * PE]

        rho_ss = qt.steadystate(H, c_ops, method='direct')
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

        for i, gam in enumerate(gamma_arr):
            g2_map[i, :] = compute_g2_bundle(gam, kap, p['tau_arr'])
            if (i + 1) % 5 == 0 or i == n_gamma - 1:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_gamma - i - 1)
                print(f"  κ={kap}  [{i+1:3d}/{n_gamma}]  "
                      f"γ={gam:.2e}  ({elapsed:.0f}s, ETA ~{eta:.0f}s)")

        p['g2_map'] = g2_map

    # Guardar para no recalcular
    np.savez('results/data/fig_S6_data.npz',
             gamma_arr=gamma_arr,
             **{f'g2_kap{k}': panels[k]['g2_map'] for k in kappa_list},
             **{f'tau_kap{k}': panels[k]['tau_arr'] for k in kappa_list})
    print("✓ Datos guardados en results/data/fig_S6_data.npz")

else:
    data = np.load('results/data/fig_S6_data.npz')
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


# -----------------------------------------------------------------------------
# 3. VISUALIZACIÓN 3D
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(6.17, 3.10))

for col, kap in enumerate(kappa_list):
    ax = fig.add_subplot(1, 2, col + 1, projection='3d')
    p = panels[kap]

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

    # ── log10(g2), clippeado ────────────────────────────────────
    with np.errstate(divide='ignore', invalid='ignore'):
        log_g2 = np.log10(np.clip(p['g2_map'], 1e-3, 1e3))

    # ── Colores ─────────────────────────────────────────────────
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    colors = plt.cm.jet(norm(log_g2))

    # ── Surface: X=log(γ), Y=τ ──────────────────────────────────
    LOG_GAM_2D = np.log10(p['GAMMA'])
    TAU_2D     = p['TAU']

    ax.plot_surface(
        LOG_GAM_2D, TAU_2D, log_g2,
        facecolors=colors, rstride=1, cstride=1,
        shade=False, antialiased=True
    )

    # ── Línea τ = 1/κ en el piso ───────────────────────────────
    tau_kappa = 1.0 / kap
    x_line = np.log10(gamma_arr)
    z_floor = np.full_like(x_line, -2.5)

    # ── Eje X: γ/ωb (frente, log) ──────────────────────────────
    ax.set_xlabel(r'$\gamma/\omega_b$', fontsize=12, labelpad=10)
    x_ticks = [0, -1, -2, -3, -4, -5]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([r'$10^{%d}$' % v for v in x_ticks], fontsize=10)
    ax.set_xlim(np.log10(gamma_arr[-1]), np.log10(gamma_arr[0]))

    # ── Eje Y: ωb·τ (fondo) ────────────────────────────────────
    ax.set_ylabel(r'$\omega_b \tau\;(\times 10^3)$', fontsize=12, labelpad=10)
    y_tick_vals = [0, 2000, 4000, 6000, 8000, 10000]
    ax.set_yticks(y_tick_vals)
    ax.set_yticklabels([r'$0$', r'$2$', r'$4$', r'$6$', r'$8$', r'$10$'],
                       fontsize=10)
    ax.set_ylim(0, p['tau_arr'][-1])

    # ── Eje Z: ticks a la IZQUIERDA + label manual ─────────────
    ax.set_zlabel('')
    ax.set_zlim(-2, 2)
    ax.set_zticks([-2, -1, 0, 1, 2])
    ax.zaxis.set_tick_params(pad=-1)
    ax.zaxis._axinfo['juggled'] = (1, 2, 0)
    # Label más separado de los ticks (x=-0.12)
    ax.text2D(-0.12, 0.50, r'$\log_{10}\,g_2^{(2)}(\tau)$',
              transform=ax.transAxes, fontsize=10,
              rotation=90, va='center', ha='center')

    # ── Vista ───────────────────────────────────────────────────
    ax.view_init(elev=15, azim=-35)
    ax.set_box_aspect([2, 2, 1])

    # ── Título ──────────────────────────────────────────────────
    label = '(a)' if col == 0 else '(b)'
    ax.set_title(rf'{label}  $\kappa/\omega_b = {kap}$', fontsize=12, pad=8)

    # ── Colorbar con solo ticks -1, 0, 1 (sin label) ───────────
    sm_cb = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm_cb.set_array([])
    cbar = fig.colorbar(sm_cb, ax=ax, shrink=0.55, pad=0.08, aspect=15)
    cbar.set_ticks([-1, 0, 1])
    cbar.ax.tick_params(labelsize=10)

fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08, wspace=0.16)
plt.savefig("results/oficial/fig_S6_scaffold.pdf", bbox_inches='tight')
plt.savefig("results/oficial/pgf/fig_S6_scaffold.pgf")
print("✓ Figura guardada: results/oficial/fig_S6_scaffold.pdf")
plt.close()