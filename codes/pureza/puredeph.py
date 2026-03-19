#!/usr/bin/env python3
"""
Réplica de la Fig. S5 — Suplementario de Bin et al. (PRL 124, 053601, 2020)
============================================================================

Panel (a): Π_2 y Π_3 vs γ_φ/ω_b
  Parámetros fijos: Ω/ω_b=0.2, λ/ω_b=0.1, γ/ω_b=0.0002, κ/ω_b=0.002

Panel (b): Π_2 vs Ω_eff^(2)/ω_b para κ/ω_b ∈ {0.002, 0.004, 0.006}
  Parámetros fijos: λ/ω_b=0.1, γ/ω_b=0.0002, γ_φ/ω_b=0.0004
  Se varía Ω (no λ), con Ω_eff^(2) = Ω·(λ/ω_b)²/√2

Corrección v3: El ancho de la resonancia de Stokes es ~κ, así que el
barrido de Δ necesita paso << κ. Se usa esquema de dos pasadas:
  1) Pasada gruesa: ventana ±0.1, ~15 puntos (paso ~0.013)
  2) Pasada fina:   ventana ±2κ alrededor del mejor Δ grueso, ~15 puntos

Autor: Jhon S. García B. — Tesis UQ 2025
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import time

# =============================================================================
# PARÁMETROS GLOBALES
# =============================================================================
omega_b = 1.0
gamma   = 0.0002     # γ/ω_b
Ncut    = 20         # Truncamiento Fock completo


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================
def build_operators(Nc):
    """Construir operadores del sistema QD ⊗ Fock(Nc)."""
    b  = qt.destroy(Nc)
    nb = b.dag() * b
    Ib = qt.qeye(Nc)
    sm = qt.sigmam()
    Iq = qt.qeye(2)

    ops = {
        'b':   qt.tensor(Iq, b),
        'nb':  qt.tensor(Iq, nb),
        'sm':  qt.tensor(sm, Ib),
        'sp':  qt.tensor(sm.dag(), Ib),
        'pe':  qt.tensor(sm.dag() * sm, Ib),
        'Nc':  Nc,
    }
    ops['H_phonon'] = omega_b * ops['nb']
    return ops


def validate_rho(rho, tol=1e-8):
    """Validar traza, hermiticidad y positividad de ρ."""
    if abs(rho.tr() - 1.0) > 1e-6:
        return False
    if not rho.isherm:
        return False
    evals = np.real(rho.eigenstates()[0])
    if np.min(evals) < -tol:
        return False
    return True


def solve_ss(H, c_ops):
    """Resolver estado estacionario con fallback de métodos."""
    for method in ('direct', 'eigen', 'svd'):
        try:
            rho = qt.steadystate(H, c_ops, method=method, use_rcm=True)
            if validate_rho(rho):
                return rho
        except Exception:
            pass
    return None


def resonance_estimate(n_bundle, lam, Omega_val):
    """
    Estimación de la resonancia de Stokes corregida.
    Combina Régimen II (Lamb shift) y III (driving):
      Δ_n ≈ -√((n·ω_b)² - 4Ω²) + λ²/ω_b
    """
    arg = (n_bundle * omega_b)**2 - 4 * Omega_val**2
    if arg < 0:
        return -n_bundle * omega_b
    Delta_III = -np.sqrt(arg)
    Lamb_shift = lam**2 / omega_b
    return Delta_III + Lamb_shift


def purity_at_Delta(ops_f, ops_t, lam, kappa, gamma_phi_val, Omega_val, Delta):
    """Calcular π_n para un Δ dado. Retorna pureza escalar."""
    H_drive_f = Omega_val * (ops_f['sm'] + ops_f['sp'])
    H_drive_t = Omega_val * (ops_t['sm'] + ops_t['sp'])

    # Modelo completo
    H_f = (ops_f['H_phonon']
           + Delta * ops_f['pe']
           + lam * ops_f['pe'] * (ops_f['b'] + ops_f['b'].dag())
           + H_drive_f)
    c_f = [
        np.sqrt(kappa)         * ops_f['b'],
        np.sqrt(gamma)         * ops_f['sm'],
        np.sqrt(gamma_phi_val) * ops_f['pe'],
    ]
    rho_f = solve_ss(H_f, c_f)
    if rho_f is None:
        return np.nan

    n_a = np.real(qt.expect(ops_f['nb'], rho_f))

    # Modelo truncado
    H_t = (ops_t['H_phonon']
           + Delta * ops_t['pe']
           + lam * ops_t['pe'] * (ops_t['b'] + ops_t['b'].dag())
           + H_drive_t)
    c_t = [
        np.sqrt(kappa)         * ops_t['b'],
        np.sqrt(gamma)         * ops_t['sm'],
        np.sqrt(gamma_phi_val) * ops_t['pe'],
    ]
    rho_t = solve_ss(H_t, c_t)
    if rho_t is None:
        return np.nan

    na1 = np.real(qt.expect(ops_t['nb'], rho_t))
    na_n = max(n_a - na1, 0.0)
    return na_n / n_a if n_a > 1e-30 else 0.0


def compute_purity_optimized(ops_f, ops_t, lam, kappa, gamma_phi_val,
                              Omega_val, n_bundle):
    """
    Calcular π_n con optimización de Δ en dos pasadas:
      1) Gruesa: ventana ±0.1, 15 puntos
      2) Fina:   ventana ±3κ alrededor del mejor, 15 puntos
    
    El paso fino es ~6κ/15 = 0.4κ, suficiente para resolver la resonancia.
    """
    Delta_center = resonance_estimate(n_bundle, lam, Omega_val)

    # --- Pasada 1: gruesa ---
    n_coarse = 15
    window_coarse = 0.1
    Delta_coarse = np.linspace(Delta_center - window_coarse,
                               Delta_center + window_coarse,
                               n_coarse)
    
    best_pi = -1.0
    best_Delta = Delta_center

    for Delta in Delta_coarse:
        pi = purity_at_Delta(ops_f, ops_t, lam, kappa, gamma_phi_val,
                             Omega_val, Delta)
        if not np.isnan(pi) and pi > best_pi:
            best_pi = pi
            best_Delta = Delta

    # --- Pasada 2: fina alrededor del mejor punto grueso ---
    n_fine = 15
    window_fine = max(3.0 * kappa, 0.002)  # ±3κ mínimo, piso de 0.002
    Delta_fine = np.linspace(best_Delta - window_fine,
                             best_Delta + window_fine,
                             n_fine)
    
    for Delta in Delta_fine:
        pi = purity_at_Delta(ops_f, ops_t, lam, kappa, gamma_phi_val,
                             Omega_val, Delta)
        if not np.isnan(pi) and pi > best_pi:
            best_pi = pi
            best_Delta = Delta

    return best_pi if best_pi >= 0 else np.nan


# =============================================================================
# PRECONSTRUIR OPERADORES
# =============================================================================
print("Construyendo operadores...")
ops_full = build_operators(Ncut)
ops_trunc = {n: build_operators(n) for n in [2, 3]}


# =============================================================================
# PANEL (a): Π_2 y Π_3 vs γ_φ/ω_b
# =============================================================================
print("\n" + "="*65)
print("  PANEL (a): Π_n vs γ_φ/ω_b")
print("  Parámetros: Ω=0.2, λ=0.1, γ=0.0002, κ=0.002")
print("="*65)

Omega_a = 0.2
lam_a   = 0.1
kappa_a = 0.002
gamma_phi_arr_a = np.logspace(-5, -1, 40)  # Producción: 60-80

purity_a = {2: np.zeros(len(gamma_phi_arr_a)),
            3: np.zeros(len(gamma_phi_arr_a))}

t0 = time.time()
for idx, gp in enumerate(gamma_phi_arr_a):
    for n_b in [2, 3]:
        pi_n = compute_purity_optimized(
            ops_full, ops_trunc[n_b], lam_a, kappa_a, gp, Omega_a, n_b)
        purity_a[n_b][idx] = pi_n

    if (idx + 1) % 5 == 0 or idx == len(gamma_phi_arr_a) - 1:
        elapsed = time.time() - t0
        print(f"  [{idx+1}/{len(gamma_phi_arr_a)}]  γ_φ={gp:.2e}  "
              f"Π_2={purity_a[2][idx]:.4f}  Π_3={purity_a[3][idx]:.4f}  "
              f"({elapsed:.0f}s)")

print(f"  ✓ Panel (a) completado en {time.time()-t0:.1f}s")


# =============================================================================
# PANEL (b): Π_2 vs Ω_eff^(2)/ω_b — BARRIENDO Ω
# =============================================================================
print("\n" + "="*65)
print("  PANEL (b): Π_2 vs Ω_eff^(2)/ω_b  [variando Ω]")
print("  Parámetros fijos: λ=0.1, γ=0.0002, γ_φ=0.0004")
print("="*65)

lam_b       = 0.1
gamma_phi_b = 0.0004
kappa_list  = [0.002, 0.004, 0.006]

# Ω_eff^(2) = Ω·(λ/ω_b)²/√2
factor_eff = (lam_b / omega_b)**2 / np.sqrt(2)

# Rango de Bin: Ω_eff ∈ [~5.6e-5, ~2.5e-3] → Ω ∈ [~0.008, ~0.354]
Omega_arr_b = np.linspace(0.008, 0.36, 50)  # Producción: 80-100
Omega_eff_arr = Omega_arr_b * factor_eff

purity_b = {kap: np.zeros(len(Omega_arr_b)) for kap in kappa_list}

t0 = time.time()
for kap in kappa_list:
    print(f"\n  κ/ω_b = {kap}")
    for idx, Om in enumerate(Omega_arr_b):
        pi_2 = compute_purity_optimized(
            ops_full, ops_trunc[2], lam_b, kap, gamma_phi_b, Om, 2)
        purity_b[kap][idx] = pi_2

        if (idx + 1) % 10 == 0 or idx == len(Omega_arr_b) - 1:
            elapsed = time.time() - t0
            print(f"    [{idx+1}/{len(Omega_arr_b)}]  Ω={Om:.4f}  "
                  f"Ω_eff={Omega_eff_arr[idx]:.2e}  "
                  f"Π_2={pi_2:.4f}  ({elapsed:.0f}s)")

print(f"\n  ✓ Panel (b) completado en {time.time()-t0:.1f}s")


# =============================================================================
# VISUALIZACIÓN — Fig. S5
# =============================================================================
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))

# --- Panel (a) ---
ax_a.plot(gamma_phi_arr_a, purity_a[2], 'k-', lw=2, label=r'$\Pi_2$')
ax_a.plot(gamma_phi_arr_a, purity_a[3], 'r--', lw=2, label=r'$\Pi_3$')
ax_a.set_xscale('log')
ax_a.set_xlabel(r'$\gamma_\varphi/\omega_b$', fontsize=14)
ax_a.set_ylabel(r'$\Pi_n$', fontsize=14)
ax_a.set_xlim(1e-5, 1e-1)
ax_a.set_ylim(0.35, 1.05)
ax_a.legend(fontsize=12, loc='lower left')
ax_a.tick_params(labelsize=11)
ax_a.text(0.05, 0.95, '(a)', transform=ax_a.transAxes,
          fontsize=14, va='top', ha='left', fontweight='bold')

# --- Panel (b) ---
styles = [
    {'color': 'k', 'marker': 'o', 'mfc': 'none', 'label': r'$\kappa/\omega_b = 0.002$'},
    {'color': 'r', 'marker': '*', 'mfc': 'r',    'label': r'$\kappa/\omega_b = 0.004$'},
    {'color': 'b', 'marker': 's', 'mfc': 'none', 'label': r'$\kappa/\omega_b = 0.006$'},
]
for i, kap in enumerate(kappa_list):
    ax_b.plot(Omega_eff_arr, purity_b[kap],
              color=styles[i]['color'],
              marker=styles[i]['marker'],
              markerfacecolor=styles[i]['mfc'],
              markersize=5, lw=1.2,
              label=styles[i]['label'])

ax_b.set_xlabel(r'$\Omega_{\mathrm{eff}}^{(2)}/\omega_b$', fontsize=14)
ax_b.set_ylabel(r'$\Pi_2$', fontsize=14)
ax_b.ticklabel_format(axis='x', style='scientific', scilimits=(-3,-3))
ax_b.set_ylim(0.97, 1.001)
ax_b.legend(fontsize=10, loc='lower right')
ax_b.tick_params(labelsize=11)
ax_b.text(0.05, 0.95, '(b)', transform=ax_b.transAxes,
          fontsize=14, va='top', ha='left', fontweight='bold')

plt.tight_layout()
plt.savefig("fig_S5_replica.pdf", bbox_inches='tight')
plt.savefig("fig_S5_replica.png", dpi=200, bbox_inches='tight')
print("\n✓ Figuras guardadas: fig_S5_replica.{pdf,png}")
plt.show()


# =============================================================================
# DIAGNÓSTICO
# =============================================================================
print("\n" + "="*65)
print("  DIAGNÓSTICO")
print("="*65)

# Verificar resolución del barrido fino
print(f"\n  Paso grueso: {2*0.1/15:.5f} = {2*0.1/15/kappa_a:.1f}κ")
print(f"  Paso fino (κ=0.002): {2*3*0.002/15:.6f} = {2*3*0.002/15/0.002:.2f}κ")
print(f"  Paso fino (κ=0.004): {2*3*0.004/15:.6f} = {2*3*0.004/15/0.004:.2f}κ")

# Sweet spots
factor = (lam_b / omega_b)**2 / np.sqrt(2)
for kap in kappa_list:
    Omega_sweet = kap / (10 * factor)
    print(f"\n  κ={kap}: sweet spot Ω ≈ {Omega_sweet:.3f}, "
          f"Ω_eff ≈ {Omega_sweet*factor:.2e}")

print("\n" + "="*65)