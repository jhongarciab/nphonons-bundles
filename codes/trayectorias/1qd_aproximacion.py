#!/usr/bin/env python3
"""
=============================================================================
Réplica de la Figura 5 de Bin et al., PRL 124, 053601 (2020)
"N-Phonon Bundle Emission via the Stokes Process"

Mapa de pureza Π_N(λ/ωb, κ/ωb) para N = 2 y N = 3.

SISTEMA: 1 QD acoplado a cavidad acústica (modo fonónico)
ESPACIO:  QD ⊗ Fock(Ncut)  — QD en primer subespacio

HAMILTONIANO (frame rotante con el láser):
   H = ωb b†b + Δ σ†σ + λ σ†σ (b + b†) + Ω (σ + σ†)

ECUACIÓN MAESTRA (Lindblad):
   dρ/dt = -i[H,ρ] + κ D[b] + γ D[σ] + γ_φ D[σ†σ]

MÉTRICA DE PUREZA — Indicador T_N (Muñoz et al., Optica 5, 14, 2018, Eq. 20):

   T_N = N · ⟨(b†)^N b^N⟩_ss / ((N-1)! · ⟨b†b⟩_ss)

   T_N es un indicador del régimen de emisión de N-fonones calculable
   directamente desde ρ_ss. Es una aproximación de Π_N; puede exceder 1
   fuera del régimen ideal, por lo que se satura en [0, 1].
   Muñoz et al. muestran que T_N ≈ Π_N en el régimen de alta pureza.

PARÁMETROS (Fig. 5 de Bin):
   Ω/ωb = 0.2,  γ/ωb = 0.0002,  γ_φ/ωb = 0.0004

CORRECCIÓN RESPECTO AL CÓDIGO ORIGINAL:
   El Delta óptimo se busca para cada par (λ, κ), no con un κ_ref fijo.
   Esto evita el sesgo en la resonancia para κ grande.

INSTRUCCIONES DE EJECUCIÓN:
   python purity_map_bin_fig5.py

   Tiempo estimado:
     Resolución TEST  (20×20): ~3 min
     Resolución MEDIA (40×40): ~12 min
     Resolución PROD  (80×80): ~50 min

   Los datos se guardan en .npy antes de graficar (seguridad ante cortes).
=============================================================================
"""

import numpy as np
import qutip as qt
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from math import factorial
import time
import sys

print(f"QuTiP v{qt.__version__}  |  NumPy v{np.__version__}")

# =============================================================================
# PARÁMETROS FÍSICOS — Fig. 5 de Bin et al. (2020)
# =============================================================================
omega_b   = 1.0      # frecuencia del fonón (unidad de energía)
Omega     = 0.2      # amplitud de driving  Ω/ωb
gamma     = 0.0002   # tasa de decaimiento QD   γ/ωb
gamma_phi = 0.0004   # tasa de dephasing puro   γ_φ/ωb

# =============================================================================
# PARÁMETROS NUMÉRICOS — ajustar según tiempo disponible
# =============================================================================
Ncut    = 15    # truncamiento Fock (suficiente para N≤3, λ/ωb≤0.14)
n_lam   = 20    # puntos en λ/ωb  (TEST: 20 | PROD: 80)
n_kappa = 20    # puntos en κ/ωb  (TEST: 20 | PROD: 80)
n_delta = 20    # puntos en el barrido de Delta para buscar resonancia

lambda_arr = np.linspace(0.02, 0.14, n_lam)
kappa_arr  = np.logspace(-3, 0, n_kappa)

# =============================================================================
# ESPACIO DE HILBERT:  QD ⊗ Fock(Ncut)
#   subespacio 0 → QD     (dim 2)
#   subespacio 1 → fonón  (dim Ncut)
# =============================================================================
b    = qt.destroy(Ncut)
nb   = b.dag() * b
I_b  = qt.qeye(Ncut)
I_q  = qt.qeye(2)

# Operadores QD
sm   = qt.sigmam()          # σ = |v⟩⟨c|
sp   = sm.dag()
sz   = sp * sm              # σ†σ = |c⟩⟨c|  (proyector estado excitado)

# Operadores del sistema completo (QD ⊗ Fock)
b_sys    = qt.tensor(I_q, b)          # aniquilación fonónica
nb_sys   = qt.tensor(I_q, nb)         # número fonónico
sm_sys   = qt.tensor(sm, I_b)         # de-excitación QD
sp_sys   = sm_sys.dag()
proj_exc = qt.tensor(sz, I_b)         # |c⟩⟨c| ⊗ 1_b  (σ†σ)

# Partes FIJAS del Hamiltoniano (no dependen de Δ ni κ)
H_phon  = omega_b * nb_sys
H_drive = Omega * (sm_sys + sp_sys)

# Operadores para calcular T_N
bdagb2 = b_sys.dag()**2 * b_sys**2    # (b†)²b²
bdagb3 = b_sys.dag()**3 * b_sys**3    # (b†)³b³

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def validate_rho(rho, tol=1e-6):
    """Verificaciones básicas de validez de la matriz densidad."""
    if abs(rho.tr() - 1.0) > 1e-4:
        return False
    if not rho.isherm:
        return False
    evals = np.real(rho.eigenstates()[0])
    if np.min(evals) < -tol:
        return False
    return True


def solve_ss(H, c_ops):
    """Estado estacionario con métodos de respaldo (direct → eigen → svd)."""
    for method in ('direct', 'eigen', 'svd'):
        try:
            rho = qt.steadystate(H, c_ops, method=method)
            if validate_rho(rho):
                return rho
        except Exception:
            pass
    return None


def build_H(Delta, lam):
    """Construye el Hamiltoniano completo dado Δ y λ."""
    H_eph     = lam * proj_exc * (b_sys + b_sys.dag())
    H_detuning = Delta * proj_exc
    return H_phon + H_detuning + H_eph + H_drive


def build_c_ops(kappa):
    """Lista de operadores de colapso dado κ."""
    return [
        np.sqrt(kappa)     * b_sys,
        np.sqrt(gamma)     * sm_sys,
        np.sqrt(gamma_phi) * proj_exc,
    ]


def T_N(n, rho):
    """
    Indicador de pureza T_N (Muñoz et al. 2018, Ec. 20, generalizado a N):

       T_N = N · ⟨(b†)^N b^N⟩ / ((N-1)! · ⟨b†b⟩)

    Devuelve valor saturado en [0, 1].
    Si ⟨b†b⟩ < 1e-18, devuelve 0.
    """
    nbar = np.real(qt.expect(nb_sys, rho))
    if nbar < 1e-18:
        return 0.0
    if n == 2:
        exp_n = np.real(qt.expect(bdagb2, rho))
    elif n == 3:
        exp_n = np.real(qt.expect(bdagb3, rho))
    else:
        op = b_sys.dag()**n * b_sys**n
        exp_n = np.real(qt.expect(op, rho))
    val = n * exp_n / (factorial(n - 1) * nbar)
    return float(np.clip(val, 0.0, 1.0))


# =============================================================================
# BARRIDO 2D (λ, κ) con búsqueda de Delta óptimo para cada par
# =============================================================================
# Estimación inicial de resonancia Stokes de orden n (Régimen I):
#   Δ_n ≈ -n·ωb + n·λ²/ωb
#   (el término Ω²/(n·ωb) es corrección pequeña para Ω=0.2, n≥2)

def find_optimal_delta(n_bundle, lam, kappa):
    """
    Busca el Δ que maximiza ⟨(b†)^n b^n⟩_ss para los parámetros dados.
    Usa un barrido fino alrededor de la estimación analítica.
    """
    Delta_est = -n_bundle * omega_b + n_bundle * lam**2 / omega_b
    Delta_scan = np.linspace(Delta_est - 0.3, Delta_est + 0.3, n_delta)

    c_ops_loc = build_c_ops(kappa)
    op_n = b_sys.dag()**n_bundle * b_sys**n_bundle

    best_val   = -1.0
    best_delta = Delta_est

    for D in Delta_scan:
        rho = solve_ss(build_H(D, lam), c_ops_loc)
        if rho is None:
            continue
        val = np.real(qt.expect(op_n, rho))
        if val > best_val:
            best_val   = val
            best_delta = D

    return best_delta


def run_sweep(n_bundle, label):
    """
    Barrido 2D completo sobre (λ, κ).
    Para cada par (λ, κ) busca el Delta óptimo y calcula T_N.
    Retorna mapa shape (n_kappa, n_lam).
    """
    purity_map = np.full((n_kappa, n_lam), np.nan)
    t0 = time.time()

    for j, lam in enumerate(lambda_arr):
        for i, kappa in enumerate(kappa_arr):

            delta_opt = find_optimal_delta(n_bundle, lam, kappa)
            H         = build_H(delta_opt, lam)
            c_ops_loc = build_c_ops(kappa)
            rho       = solve_ss(H, c_ops_loc)

            if rho is not None:
                purity_map[i, j] = T_N(n_bundle, rho)

        # Progreso
        elapsed = time.time() - t0
        eta     = elapsed / (j + 1) * (n_lam - j - 1)
        frac    = 100 * (j + 1) / n_lam
        print(f"  [{label}]  λ={lam:.3f}  {j+1}/{n_lam}  "
              f"({frac:.0f}%)  t={elapsed:.0f}s  ETA≈{eta:.0f}s",
              flush=True)

    print(f"  [{label}] ✓ Completado en {time.time()-t0:.0f}s\n")
    return purity_map


# =============================================================================
# EJECUCIÓN
# =============================================================================
print("=" * 65)
print("Réplica Fig. 5 — Bin et al. PRL 124, 053601 (2020)")
print("=" * 65)
print(f"  Ω/ωb={Omega}  γ/ωb={gamma}  γ_φ/ωb={gamma_phi}  Ncut={Ncut}")
print(f"  Grilla: {n_lam}×{n_kappa}  |  n_delta={n_delta}")
print(f"  λ ∈ [{lambda_arr[0]:.2f}, {lambda_arr[-1]:.2f}]")
print(f"  κ ∈ [{kappa_arr[0]:.1e}, {kappa_arr[-1]:.1e}] (log)\n")

print("Panel (a): Π_2 — barrido en curso...")
purity_2 = run_sweep(2, "Π₂")

print("Panel (b): Π_3 — barrido en curso...")
purity_3 = run_sweep(3, "Π₃")

# Guardar datos antes de graficar
np.save('purity_2ph.npy',    purity_2)
np.save('purity_3ph.npy',    purity_3)
np.save('lambda_arr.npy',    lambda_arr)
np.save('kappa_arr.npy',     kappa_arr)
print("✓ Datos guardados (.npy)\n")

# =============================================================================
# FIGURA — Réplica del estilo de Bin Fig. 5
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5), sharey=True)
fig.subplots_adjust(wspace=0.06, right=0.88)

norm = Normalize(vmin=0, vmax=1)
cmap = plt.cm.YlOrRd_r     # amarillo (alta pureza) → rojo (baja)

for ax, purity, n_b, panel_label in [
    (axes[0], purity_2, 2, '(a)'),
    (axes[1], purity_3, 3, '(b)'),
]:
    p = np.clip(np.nan_to_num(purity, nan=0.0), 0.0, 1.0)

    im = ax.pcolormesh(
        lambda_arr, kappa_arr, p,
        norm=norm, cmap=cmap, shading='auto'
    )

    # Contornos de alta pureza
    try:
        levels = [0.90, 0.99] if n_b == 2 else [0.85, 0.95]
        cs = ax.contour(
            lambda_arr, kappa_arr, p,
            levels=levels,
            colors='black', linewidths=1.0, linestyles='dashed'
        )
        ax.clabel(cs, inline=True, fontsize=9, fmt='%.2f')
    except Exception:
        pass

    ax.set_yscale('log')
    ax.set_xlim(lambda_arr[0], lambda_arr[-1])
    ax.set_ylim(kappa_arr[0], kappa_arr[-1])
    ax.set_xlabel(r'$\lambda/\omega_b$', fontsize=13)
    ax.tick_params(labelsize=11)

    ax.text(0.05, 0.95, panel_label,
            transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top', color='white',
            bbox=dict(boxstyle='round,pad=0.2',
                      facecolor='black', alpha=0.5))

    ax.set_title(rf'$\Pi_{n_b}$', fontsize=13)

axes[0].set_ylabel(r'$\kappa/\omega_b$', fontsize=13)

# Colorbar compartida
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_ticks([0, 0.5, 1.0])
cbar.set_ticklabels(['0', '0.5', '1'])
cbar.set_label(r'$\Pi_N$', fontsize=13)
cbar.ax.tick_params(labelsize=11)

plt.show()