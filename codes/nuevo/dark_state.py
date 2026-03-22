#!/usr/bin/env python3
"""
Población del estado oscuro |Ψ₋⟩ vs Δ — Molécula excitónica (2QD)
===================================================================

Resultado original: demostración numérica de que el estado oscuro de Dicke
|Ψ₋⟩ = (|cv⟩ − |vc⟩)/√2 permanece completamente desacoplado del láser
en todo el rango de desintonía Δ, incluyendo las resonancias de Stokes.

Se plotea:
  - Panel superior: g⁽²⁾(0) vs Δ (para contexto, mostrando las resonancias)
  - Panel inferior: ⟨Ψ₋|ρ_ss|Ψ₋⟩ vs Δ (población del estado oscuro)

El contraste es el resultado: las resonancias de Stokes producen picos
enormes en g⁽²⁾, pero la población del estado oscuro permanece en cero.
Solo el subespacio brillante |Ψ₊⟩ = (|cv⟩ + |vc⟩)/√2 participa.

Parámetros: los mismos de las figuras de g⁽ⁿ⁾ del 2QD.

Autor: Jhon S. García B. — Tesis UQ 2025
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import time

# =============================================================================
# PARÁMETROS — idénticos a forster_2qds.py
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
# OPERADORES — espacio QD₁ ⊗ QD₂ ⊗ Fock
# =============================================================================
b    = qt.destroy(Ncut)
num_b = b.dag() * b
I_b  = qt.qeye(Ncut)
sm   = qt.sigmam()
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

# =============================================================================
# PROYECTOR DEL ESTADO OSCURO
# =============================================================================
# Estados electrónicos de base: |v⟩ = basis(2,1), |c⟩ = basis(2,0)
ket_v = qt.basis(2, 1)
ket_c = qt.basis(2, 0)

# |Ψ₋⟩ = (|cv⟩ − |vc⟩)/√2  (estado oscuro de Dicke)
psi_minus_elec = (qt.tensor(ket_c, ket_v) - qt.tensor(ket_v, ket_c)).unit()

# |Ψ₊⟩ = (|cv⟩ + |vc⟩)/√2  (estado brillante)
psi_plus_elec = (qt.tensor(ket_c, ket_v) + qt.tensor(ket_v, ket_c)).unit()

# |vv⟩ (ground state)
psi_vv = qt.tensor(ket_v, ket_v)

# |cc⟩ (doble excitación)
psi_cc = qt.tensor(ket_c, ket_c)

# Proyectores electrónicos (sumados sobre Fock)
# P_{Ψ₋} = |Ψ₋⟩⟨Ψ₋| ⊗ I_b = Σ_m |Ψ₋,m⟩⟨Ψ₋,m|
proj_dark = qt.tensor(psi_minus_elec * psi_minus_elec.dag(), I_b)
proj_bright = qt.tensor(psi_plus_elec * psi_plus_elec.dag(), I_b)
proj_vv = qt.tensor(psi_vv * psi_vv.dag(), I_b)
proj_cc = qt.tensor(psi_cc * psi_cc.dag(), I_b)

# g⁽²⁾ = ⟨b†²b²⟩/⟨n̂⟩²
def factorial_number_operator(num_op, n, I_op):
    op = I_op
    for k in range(n):
        op = op * (num_op - k * I_op)
    return op

bdagb_2 = factorial_number_operator(num_sys, 2, I_sys)

# =============================================================================
# VALIDACIÓN Y SOLVER
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
print("  Población del estado oscuro |Ψ₋⟩ vs Δ")
print(f"  Parámetros: λ={lam_over_ob}, Ω={Omega_over_ob}, "
      f"κ={kappa_over_ob}, γ={gamma_over_ob}, J={J_over_ob}")
print(f"  Ncut={Ncut}, {len(Delta_list)} puntos de Δ")
print("="*65)

# Arrays de resultados
pop_dark    = np.full_like(Delta_list, np.nan)
pop_bright  = np.full_like(Delta_list, np.nan)
pop_vv      = np.full_like(Delta_list, np.nan)
pop_cc      = np.full_like(Delta_list, np.nan)
g2_vals     = np.full_like(Delta_list, np.nan)
nbar_vals   = np.full_like(Delta_list, np.nan)

t0 = time.time()
for i, Delta in enumerate(Delta_list):
    H = (Delta * (proj_e1 + proj_e2)
         + H_phonon + H_interaction + H_drive + H_Forster)

    rho_ss = solve_ss(H, c_ops)
    if rho_ss is None:
        continue

    # Poblaciones de los subespacios electrónicos
    pop_dark[i]   = np.real(qt.expect(proj_dark, rho_ss))
    pop_bright[i] = np.real(qt.expect(proj_bright, rho_ss))
    pop_vv[i]     = np.real(qt.expect(proj_vv, rho_ss))
    pop_cc[i]     = np.real(qt.expect(proj_cc, rho_ss))

    # g⁽²⁾(0) y ⟨n̂⟩
    nbar = np.real(qt.expect(num_sys, rho_ss))
    nbar_vals[i] = nbar
    if nbar > 1e-12:
        g2_vals[i] = np.real(qt.expect(bdagb_2, rho_ss)) / nbar**2

    if (i + 1) % 100 == 0 or i == len(Delta_list) - 1:
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(Delta_list)}]  Δ={Delta:.2f}  "
              f"P(Ψ₋)={pop_dark[i]:.2e}  P(Ψ₊)={pop_bright[i]:.2e}  "
              f"g²={g2_vals[i]:.2e}  ({elapsed:.0f}s)")

print(f"\n  ✓ Completado en {time.time()-t0:.1f}s")

# =============================================================================
# DIAGNÓSTICO
# =============================================================================
print("\n" + "="*65)
print("  Diagnóstico: poblaciones en resonancias de Stokes")
print("="*65)

# Resonancias esperadas: Δ_n = -n·ω_b - J
for n in [1, 2, 3, 4, 5]:
    Delta_res = -n * omega_b - J_over_ob
    idx = np.argmin(np.abs(Delta_list - Delta_res))
    if not np.isnan(pop_dark[idx]):
        print(f"\n  n={n}, Δ_res={Delta_res:.1f} (idx={idx}, Δ={Delta_list[idx]:.3f}):")
        print(f"    P(|vv⟩)  = {pop_vv[idx]:.6f}")
        print(f"    P(|Ψ₊⟩) = {pop_bright[idx]:.6f}")
        print(f"    P(|Ψ₋⟩) = {pop_dark[idx]:.6e}")
        print(f"    P(|cc⟩)  = {pop_cc[idx]:.6e}")
        print(f"    Suma     = {pop_vv[idx]+pop_bright[idx]+pop_dark[idx]+pop_cc[idx]:.6f}")
        print(f"    g⁽²⁾     = {g2_vals[idx]:.2e}")

# =============================================================================
# VISUALIZACIÓN — 3 paneles
# =============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True,
                                     gridspec_kw={'height_ratios': [2, 2, 1]})

# ── Panel (a): g⁽²⁾(0) vs Δ ────────────────────────────────────
ax1.plot(Delta_list, g2_vals, 'b-', lw=1.2)
ax1.set_yscale('log')
ax1.set_ylabel(r'$g^{(2)}(0)$', fontsize=13)
ax1.set_ylim(1e-1, 1e10)
ax1.tick_params(labelsize=10)

for n in range(1, 6):
    Delta_res = -n * omega_b - J_over_ob
    ax1.axvline(Delta_res, ls=':', color='gray', lw=0.8, alpha=0.6)
    ax1.text(Delta_res + 0.05, 5e9, rf'$n={n}$', fontsize=8,
             color='gray', ha='left')

ax1.text(0.03, 0.92, '(a)', transform=ax1.transAxes,
         fontsize=13, va='top', fontweight='bold')

# ── Panel (b): poblaciones en escala log ────────────────────────
ax2.plot(Delta_list, pop_bright, 'r-', lw=1.5, label=r'$P(|\Psi_+\rangle)$')
ax2.plot(Delta_list, pop_dark, 'b--', lw=2, label=r'$P(|\Psi_-\rangle)$')
ax2.plot(Delta_list, pop_cc, 'g:', lw=1.2, label=r'$P(|cc\rangle)$')

ax2.set_yscale('log')
ax2.set_ylabel('Población', fontsize=13)
ax2.set_ylim(1e-8, 1e0)
ax2.tick_params(labelsize=10)
ax2.legend(fontsize=9, loc='upper right', ncol=3)

for n in range(1, 6):
    Delta_res = -n * omega_b - J_over_ob
    ax2.axvline(Delta_res, ls=':', color='gray', lw=0.8, alpha=0.6)

ax2.text(0.03, 0.92, '(b)', transform=ax2.transAxes,
         fontsize=13, va='top', fontweight='bold')

# ── Panel (c): ratio P(Ψ₋)/P(Ψ₊) ──────────────────────────────
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = np.where(pop_bright > 1e-12, pop_dark / pop_bright, np.nan)

ax3.plot(Delta_list, ratio, 'k-', lw=1.5)
ax3.axhline(0.5, ls='--', color='gray', lw=1, alpha=0.6,
            label=r'$P(\Psi_-)/P(\Psi_+) = 0.5$')

ax3.set_xlabel(r'$\Delta/\omega_b$', fontsize=13)
ax3.set_ylabel(r'$P(\Psi_-)/P(\Psi_+)$', fontsize=13)
ax3.set_xlim(0.0, -6.0)
ax3.set_ylim(-0.05, 1.05)
ax3.tick_params(labelsize=10)
ax3.legend(fontsize=9, loc='upper right')

for n in range(1, 6):
    Delta_res = -n * omega_b - J_over_ob
    ax3.axvline(Delta_res, ls=':', color='gray', lw=0.8, alpha=0.6)

ax3.text(0.03, 0.92, '(c)', transform=ax3.transAxes,
         fontsize=13, va='top', fontweight='bold')

plt.tight_layout()
plt.savefig("dark_state_population.pdf", bbox_inches='tight')
plt.savefig("dark_state_population.png", dpi=200, bbox_inches='tight')
print("\n✓ Figuras guardadas: dark_state_population.{pdf,png}")
plt.show()