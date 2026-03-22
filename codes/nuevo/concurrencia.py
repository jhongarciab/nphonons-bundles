#!/usr/bin/env python3
"""
Entanglement entre QDs vs Δ — Molécula excitónica (2QD + Förster)
==================================================================

Calcula la concurrencia de Wootters C(ρ_elec) del estado reducido
electrónico de los dos QDs en función del detuning Δ/ω_b.

ρ_elec = Tr_fonón(ρ_ss)  →  matriz densidad de 4×4 (2 qubits)

La concurrencia mide el entanglement entre QD₁ y QD₂ en el estado
estacionario. La pregunta es si las resonancias de Stokes (emisión
de bundles de n fonones) generan entanglement entre los emisores.

Concurrencia de Wootters:
  C(ρ) = max(0, λ₁ - λ₂ - λ₃ - λ₄)
  donde λᵢ son los valores propios (decrecientes) de √(√ρ ρ̃ √ρ)
  con ρ̃ = (σ_y ⊗ σ_y) ρ* (σ_y ⊗ σ_y)

Figura: dos paneles apilados
  (a) g⁽²⁾(0) vs Δ — resonancias de Stokes
  (b) C(ρ_elec) vs Δ — entanglement

Parámetros: idénticos a forster_2qds.py

Autor: Jhon S. García B. — Tesis UQ 2025
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import time

# =============================================================================
# PARÁMETROS
# =============================================================================
omega_b       = 1.0
lam_over_ob   = 0.08
Omega_over_ob = 0.01
kappa_over_ob = 0.003
gamma_over_ob = 0.0002
gphi_over_ob  = 0.0004
J_over_ob     = 0.5

Ncut = 8
Delta_list = np.linspace(0.0, -6.0, 501)

# =============================================================================
# OPERADORES — QD₁ ⊗ QD₂ ⊗ Fock
# =============================================================================
b    = qt.destroy(Ncut)
num_b = b.dag() * b
I_b  = qt.qeye(Ncut)
sm   = qt.sigmam()
sp   = sm.dag()
I_q  = qt.qeye(2)

b_sys   = qt.tensor(I_q, I_q, b)
num_sys = qt.tensor(I_q, I_q, num_b)
sm1     = qt.tensor(sm, I_q, I_b)
sp1     = sm1.dag()
sm2     = qt.tensor(I_q, sm, I_b)
sp2     = sm2.dag()
proj_e1 = sp1 * sm1
proj_e2 = sp2 * sm2
I_sys   = qt.tensor(I_q, I_q, I_b)

# Operadores de colapso
c_ops = [
    np.sqrt(kappa_over_ob) * b_sys,
    np.sqrt(gamma_over_ob) * sm1,
    np.sqrt(gamma_over_ob) * sm2,
    np.sqrt(gphi_over_ob)  * proj_e1,
    np.sqrt(gphi_over_ob)  * proj_e2,
]

# Hamiltonianos parciales
H_phonon      = omega_b * num_sys
H_interaction = lam_over_ob * (proj_e1 + proj_e2) * (b_sys + b_sys.dag())
H_drive       = Omega_over_ob * (sm1 + sp1 + sm2 + sp2)
H_Forster     = J_over_ob * (sp1 * sm2 + sp2 * sm1)

# g⁽²⁾ operador
def factorial_number_operator(num_op, n, I_op):
    op = I_op
    for k in range(n):
        op = op * (num_op - k * I_op)
    return op

bdagb_2 = factorial_number_operator(num_sys, 2, I_sys)

# =============================================================================
# CONCURRENCIA DE WOOTTERS
# =============================================================================
def concurrence(rho_2qubit):
    """
    Concurrencia de Wootters para un estado de 2 qubits (4×4).

    C(ρ) = max(0, λ₁ - λ₂ - λ₃ - λ₄)
    donde λᵢ son los eigenvalues en orden decreciente de
    R = √(√ρ · ρ̃ · √ρ), con ρ̃ = (σ_y⊗σ_y) ρ* (σ_y⊗σ_y).
    """
    # σ_y ⊗ σ_y
    sy = qt.sigmay()
    sysy = qt.tensor(sy, sy)

    # ρ̃ = (σ_y⊗σ_y) ρ* (σ_y⊗σ_y)
    rho_conj = qt.Qobj(rho_2qubit.full().conj(), dims=rho_2qubit.dims)
    rho_tilde = sysy * rho_conj * sysy

    # R = ρ · ρ̃ (no necesitamos √ρ explícitamente)
    # Los eigenvalues de C se obtienen de: R = ρ · ρ̃
    # λᵢ = √(eigenvalues de R), ordenados decreciente
    R = rho_2qubit * rho_tilde
    eigenvalues = np.sort(np.real(R.eigenenergies()))[::-1]

    # Corregir posibles negativos numéricos
    eigenvalues = np.maximum(eigenvalues, 0.0)
    lambdas = np.sqrt(eigenvalues)

    C = max(0.0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3])
    return C


def partial_trace_phonon(rho_full):
    """
    Traza parcial sobre el modo fonónico.

    ρ_full vive en QD₁(2) ⊗ QD₂(2) ⊗ Fock(Ncut).
    Retorna ρ_elec en QD₁(2) ⊗ QD₂(2) = 4×4.
    """
    # ptrace([0,1]) traza sobre los subsistemas 0 y 1 (QD₁, QD₂)
    # y descarta el subsistema 2 (Fock).
    # En QuTiP, ptrace(sel) RETIENE los subsistemas en sel.
    return rho_full.ptrace([0, 1])


# =============================================================================
# SOLVER
# =============================================================================
def validate_rho(rho, tol=1e-8):
    if abs(rho.tr() - 1.0) > 1e-6:
        return False
    if not rho.isherm:
        return False
    evals = np.real(rho.eigenstates()[0])
    if np.min(evals) < -tol:
        return False
    return True

def solve_ss(H, c_ops):
    for method in ('direct', 'eigen', 'svd'):
        try:
            rho = qt.steadystate(H, c_ops, method=method, use_rcm=True)
            if validate_rho(rho):
                return rho
        except Exception:
            pass
    return None

# =============================================================================
# BARRIDO EN Δ
# =============================================================================
print("="*65)
print("  Concurrencia entre QDs vs Δ")
print(f"  Parámetros: λ={lam_over_ob}, Ω={Omega_over_ob}, "
      f"κ={kappa_over_ob}, γ={gamma_over_ob}, J={J_over_ob}")
print(f"  Ncut={Ncut}, {len(Delta_list)} puntos")
print("="*65)

conc_vals  = np.full_like(Delta_list, np.nan)
g2_vals    = np.full_like(Delta_list, np.nan)
nbar_vals  = np.full_like(Delta_list, np.nan)
purity_vals = np.full_like(Delta_list, np.nan)

t0 = time.time()
for i, Delta in enumerate(Delta_list):
    H = (Delta * (proj_e1 + proj_e2)
         + H_phonon + H_interaction + H_drive + H_Forster)

    rho_ss = solve_ss(H, c_ops)
    if rho_ss is None:
        continue

    # ⟨n̂⟩ y g⁽²⁾(0)
    nbar = np.real(qt.expect(num_sys, rho_ss))
    nbar_vals[i] = nbar
    if nbar > 1e-12:
        g2_vals[i] = np.real(qt.expect(bdagb_2, rho_ss)) / nbar**2

    # Traza parcial → ρ electrónico de 2 qubits
    rho_elec = partial_trace_phonon(rho_ss)

    # Concurrencia
    conc_vals[i] = concurrence(rho_elec)

    # Pureza del estado electrónico: Tr(ρ²)
    purity_vals[i] = np.real((rho_elec * rho_elec).tr())

    if (i + 1) % 100 == 0 or i == len(Delta_list) - 1:
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(Delta_list)}]  Δ={Delta:.2f}  "
              f"C={conc_vals[i]:.4e}  g²={g2_vals[i]:.2e}  "
              f"({elapsed:.0f}s)")

print(f"\n  ✓ Completado en {time.time()-t0:.1f}s")

# =============================================================================
# EXPORTAR DATOS
# =============================================================================
np.savez('concurrence_2qd_data.npz',
         Delta=Delta_list, concurrence=conc_vals,
         g2=g2_vals, nbar=nbar_vals, purity_elec=purity_vals)
print("✓ Datos exportados: concurrence_2qd_data.npz")

# =============================================================================
# DIAGNÓSTICO
# =============================================================================
print(f"\n{'='*65}")
print("  Concurrencia en resonancias de Stokes")
print(f"{'='*65}")

for n in range(1, 6):
    Delta_res = -n * omega_b - J_over_ob
    idx = np.argmin(np.abs(Delta_list - Delta_res))
    if not np.isnan(conc_vals[idx]):
        print(f"  n={n}, Δ={Delta_list[idx]:.3f}: "
              f"C={conc_vals[idx]:.6e}, g²={g2_vals[idx]:.2e}, "
              f"Tr(ρ²)={purity_vals[idx]:.6f}")

print(f"\n  C máximo: {np.nanmax(conc_vals):.6e} "
      f"en Δ={Delta_list[np.nanargmax(conc_vals)]:.3f}")
print(f"  C promedio: {np.nanmean(conc_vals):.6e}")

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                gridspec_kw={'height_ratios': [1, 1]})

# Panel (a): g⁽²⁾(0)
ax1.plot(Delta_list, g2_vals, 'b-', lw=1.2)
ax1.set_yscale('log')
ax1.set_ylabel(r'$g^{(2)}(0)$', fontsize=13)
ax1.set_ylim(1e-1, 1e10)
ax1.tick_params(labelsize=10)

for n in range(1, 6):
    Delta_res = -n * omega_b - J_over_ob
    ax1.axvline(Delta_res, ls=':', color='gray', lw=0.8, alpha=0.6)

ax1.text(0.03, 0.92, '(a)', transform=ax1.transAxes,
         fontsize=13, va='top', fontweight='bold')

# Panel (b): Concurrencia
ax2.plot(Delta_list, conc_vals, 'r-', lw=1.5)
ax2.set_xlabel(r'$\Delta/\omega_b$', fontsize=13)
ax2.set_ylabel(r'Concurrencia $\mathcal{C}$', fontsize=13)
ax2.set_xlim(0.0, -6.0)
ax2.tick_params(labelsize=10)

for n in range(1, 6):
    Delta_res = -n * omega_b - J_over_ob
    ax2.axvline(Delta_res, ls=':', color='gray', lw=0.8, alpha=0.6)

ax2.text(0.03, 0.92, '(b)', transform=ax2.transAxes,
         fontsize=13, va='top', fontweight='bold')

plt.tight_layout()
plt.savefig("concurrence_2qd.pdf", bbox_inches='tight')
plt.savefig("concurrence_2qd.png", dpi=200, bbox_inches='tight')
print("\n✓ Figuras guardadas: concurrence_2qd.{pdf,png}")
plt.show()