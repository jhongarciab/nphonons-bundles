#!/usr/bin/env python3
"""
03_escritura_polaron.py — Pulso de escritura: |Ψ_+⟩⊗|2̃⟩ → |Ψ_-⟩⊗|2̃⟩
==========================================================================

Protocolo de escritura para la memoria fonónica de 2 fonones en base
de polarón. Parte del estado estacionario real ρ_ss del sistema 2QD
bajo resonancia de Stokes n=2, y aplica un pulso diferencial
H_w = Ω_w(σ₁⁺ - σ₂⁺ + h.c.) durante t_π = π/(2√2·Ω_w).

Este Hamiltoniano actúa exclusivamente en el subespacio excitónico:
  |Ψ_+⟩ ↔ |Ψ_-⟩  (rotación π)
sin acoplar los fonones directamente.

Figura de mérito:
  F_w(t) = ⟨Ψ_-,2̃|ρ(t)|Ψ_-,2̃⟩   fidelidad de escritura

Referencia de resonancia n=2:
  Δ₂ = λ²/ω_b - 2ω_b - J  (ec. res_II de la tesis)

Autor: Jhon S. García B. — Tesis UQ 2025
"""

import numpy as np
from pathlib import Path
import qutip as qt
from scipy.optimize import curve_fit

# =============================================================================
# PARÁMETROS
# =============================================================================
RERUN = False

omega_b   = 1.0
gamma     = 0.0002
gamma_phi = 0.0004
J         = 0.5
kappa     = 0.001
lam       = 0.22
alpha     = lam / omega_b      # = 0.22

# Detuning de resonancia Stokes n=2 (ec. res_II de la tesis)
Delta_2   = lam**2 / omega_b - 2.0 * omega_b - J   # = -2.4516

# Amplitud del pulso diferencial de escritura
Omega_w   = 0.05    # Ω_w/ω_b: cumple 1/J << t_π << 1/κ

# Tiempo del pulso π: t_π = π / (2√2 · Ω_w)
t_pi      = np.pi / (2.0 * np.sqrt(2.0) * Omega_w)  # ≈ 22.2/ω_b

# Evolución del pulso: resolución fina alrededor de t_π
n_steps_w = 500
tlist_w   = np.linspace(0.0, 2.0 * t_pi, n_steps_w)  # hasta 2*t_π

Ncut = 16

print(f"Detuning resonancia n=2:  Δ₂ = {Delta_2:.6f}/ω_b")
print(f"Ω_w/ω_b = {Omega_w},  t_π = {t_pi:.4f}/ω_b")
print(f"Condición 1/J = {1/J:.1f} << t_π = {t_pi:.1f} << 1/κ = {1/kappa:.1f}  ✓")

# =============================================================================
# CONSTRUCCIÓN DE OPERADORES
# =============================================================================
b   = qt.destroy(Ncut)
nb  = b.dag() * b
Ib  = qt.qeye(Ncut)
sm  = qt.sigmam()
Iq  = qt.qeye(2)

b_s  = qt.tensor(Iq, Iq, b)
nb_s = qt.tensor(Iq, Iq, nb)
sm1  = qt.tensor(sm, Iq, Ib)
sp1  = sm1.dag()
sm2  = qt.tensor(Iq, sm, Ib)
sp2  = sm2.dag()
pe1  = sp1 * sm1
pe2  = sp2 * sm2
ne   = pe1 + pe2

# Estados colectivos
ket_cv     = qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
ket_vc     = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))
ket_dark   = (ket_cv - ket_vc) / np.sqrt(2.0)
ket_bright = (ket_cv + ket_vc) / np.sqrt(2.0)

proj_dark_exc   = ket_dark   * ket_dark.dag()
proj_bright_exc = ket_bright * ket_bright.dag()
proj_dark_sys   = qt.tensor(proj_dark_exc,   Ib)
proj_bright_sys = qt.tensor(proj_bright_exc, Ib)

# Operador de desplazamiento D(α)
D_op          = qt.displace(Ncut, alpha)
fock2_polaron = D_op * qt.basis(Ncut, 2)    # |2̃⟩ = D(α)|2⟩

# Proyector del estado objetivo: |Ψ_-,2̃⟩⟨Ψ_-,2̃|
proj_objetivo = qt.tensor(proj_dark_exc,
                           fock2_polaron * fock2_polaron.dag())

# Proyector del estado inicial esperado: |Ψ_+,2̃⟩⟨Ψ_+,2̃|
proj_inicial  = qt.tensor(proj_bright_exc,
                           fock2_polaron * fock2_polaron.dag())

# Operadores de colapso
c_ops = [
    np.sqrt(kappa)     * b_s,
    np.sqrt(gamma)     * sm1,
    np.sqrt(gamma)     * sm2,
    np.sqrt(gamma_phi) * pe1,
    np.sqrt(gamma_phi) * pe2,
]

# =============================================================================
# PASO 1: Estado estacionario ρ_ss bajo resonancia Stokes n=2
# =============================================================================
# Hamiltoniano de producción: bombeo simétrico en resonancia Δ₂
Omega_prod = 0.2    # Ω/ω_b de producción (pureza_2qds.py)

H_prod = (omega_b * nb_s
          + Delta_2 * ne
          + lam * ne * (b_s + b_s.dag())
          + J * (sp1 * sm2 + sp2 * sm1)
          + Omega_prod * (sp1 + sm1 + sp2 + sm2))

print(f"\nCalculando ρ_ss en resonancia n=2 (Δ₂={Delta_2:.4f})...")
rho_ss = qt.steadystate(H_prod, c_ops, method='direct')

# Verificaciones de ρ_ss
assert abs(rho_ss.tr() - 1.0) < 1e-8, "Traza ρ_ss ≠ 1"
assert rho_ss.isherm, "ρ_ss no hermítica"
eigvals = rho_ss.eigenenergies()
assert eigvals.min() > -1e-8, f"ρ_ss no positiva: min eigval = {eigvals.min():.2e}"

P_bright_ss = qt.expect(proj_bright_sys, rho_ss)
P_dark_ss   = qt.expect(proj_dark_sys,   rho_ss)
F_ini_ss    = qt.expect(proj_inicial,    rho_ss)
n_bar_ss    = qt.expect(nb_s,            rho_ss)

print(f"  P_+(ρ_ss) = {P_bright_ss:.5f}")
print(f"  P_-(ρ_ss) = {P_dark_ss:.5f}")
print(f"  ⟨Ψ_+,2̃|ρ_ss|Ψ_+,2̃⟩ = {F_ini_ss:.5f}  (población en estado fuente)")
print(f"  ⟨n̂⟩(ρ_ss) = {n_bar_ss:.5f}")

# =============================================================================
# PASO 2: Evolución bajo pulso diferencial H_w
# =============================================================================
# H_w = Ω_w(σ₁⁺ - σ₂⁺ + h.c.) en la base colectiva acopla |Ψ_+⟩↔|Ψ_-⟩
# En el espacio compuesto, incluimos el Hamiltoniano libre durante el pulso
H_w = Omega_w * (sp1 - sp2 + sm1 - sm2)

# Hamiltoniano durante el pulso: libre + pulso diferencial
# (sin bombeo de producción — el láser de producción se apaga)
H_pulso = (omega_b * nb_s
           + Delta_2 * ne
           + lam * ne * (b_s + b_s.dag())
           + J * (sp1 * sm2 + sp2 * sm1)
           + H_w)

e_ops_w = [
    proj_dark_sys,    # P_-(t)
    proj_bright_sys,  # P_+(t)
    nb_s,             # ⟨n̂⟩(t)
    proj_objetivo,    # F_w(t) = ⟨Ψ_-,2̃|ρ(t)|Ψ_-,2̃⟩
    proj_inicial,     # población en estado fuente |Ψ_+,2̃⟩
]

BASE_DIR = Path(__file__).resolve().parent
CODES_DIR = BASE_DIR.parent.parent
RESULTS_DATA_DIR = CODES_DIR / "results" / "data"
RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
data_file = RESULTS_DATA_DIR / "03_escritura_polaron_data.npz"

if not RERUN:
    print(f"\nEvolucionando bajo pulso diferencial (t_π = {t_pi:.2f}/ω_b)...")

    resultado_w = qt.mesolve(
        H_pulso, rho_ss, tlist_w, c_ops, e_ops_w,
        options=qt.Options(nsteps=5000, atol=1e-9, rtol=1e-7)
    )

    P_minus_w   = np.array(resultado_w.expect[0])
    P_plus_w    = np.array(resultado_w.expect[1])
    n_bar_w     = np.array(resultado_w.expect[2])
    F_w         = np.array(resultado_w.expect[3])
    F_fuente_w  = np.array(resultado_w.expect[4])

    # Valores en t = t_π
    idx_tpi = np.argmin(np.abs(tlist_w - t_pi))
    print(f"\n  En t = t_π = {tlist_w[idx_tpi]:.2f}/ω_b:")
    print(f"    P_-(t_π)              = {P_minus_w[idx_tpi]:.5f}")
    print(f"    P_+(t_π)              = {P_plus_w[idx_tpi]:.5f}")
    print(f"    F_w(t_π) — fidelidad  = {F_w[idx_tpi]:.5f}")
    print(f"    ⟨n̂⟩(t_π)             = {n_bar_w[idx_tpi]:.5f}")

    # Máximo de F_w: tiempo óptimo del pulso
    idx_max = np.argmax(F_w)
    t_opt   = tlist_w[idx_max]
    F_max   = F_w[idx_max]
    print(f"\n  Máximo de F_w:")
    print(f"    t_opt = {t_opt:.4f}/ω_b  (cf. t_π = {t_pi:.4f}/ω_b)")
    print(f"    F_w(t_opt) = {F_max:.5f}")

    np.savez(data_file,
             tlist_w=tlist_w,
             P_minus_w=P_minus_w,
             P_plus_w=P_plus_w,
             n_bar_w=n_bar_w,
             F_w=F_w,
             F_fuente_w=F_fuente_w,
             t_pi=np.array([t_pi]),
             F_ini_ss=np.array([F_ini_ss]),
             P_bright_ss=np.array([P_bright_ss]),
             n_bar_ss=np.array([n_bar_ss]),
             Delta_2=np.array([Delta_2]),
             Omega_w=np.array([Omega_w]))
    print(f"\nDatos guardados en {data_file}")

else:
    data   = np.load(data_file)
    tlist_w     = data['tlist_w']
    P_minus_w   = data['P_minus_w']
    P_plus_w    = data['P_plus_w']
    n_bar_w     = data['n_bar_w']
    F_w         = data['F_w']
    F_fuente_w  = data['F_fuente_w']
    t_pi        = float(data['t_pi'][0])
    F_ini_ss    = float(data['F_ini_ss'][0])
    idx_max     = np.argmax(F_w)
    t_opt       = tlist_w[idx_max]
    F_max       = F_w[idx_max]
    print(f"Datos cargados desde {data_file}")

# =============================================================================
# RESUMEN
# =============================================================================
print("\n" + "="*60)
print("RESUMEN — Protocolo de escritura (n=2, base polarón)")
print("="*60)
print(f"  Δ₂/ω_b          = {Delta_2:.6f}")
print(f"  Ω_w/ω_b         = {Omega_w}")
print(f"  t_π/ω_b         = {t_pi:.4f}")
print(f"  t_opt/ω_b       = {t_opt:.4f}")
print(f"  ⟨Ψ_+,2̃|ρ_ss|Ψ_+,2̃⟩  = {F_ini_ss:.5f}  (población fuente en ρ_ss)")
print(f"  F_w(t_opt)      = {F_max:.5f}  (fidelidad escritura)")
print()
print(f"  Relación F_w / ⟨Ψ_+,2̃|ρ_ss|...⟩ = {F_max/F_ini_ss:.4f}")
print(f"  -> Si ~1: el pulso transfiere eficientemente el paquete")
print(f"  -> Si <1: pérdidas durante el pulso o mezcla excitónica")
print()
print(f"  Referencia T_mem(n=2) ~ 1/κ = {1/kappa:.1f}/ω_b")
print(f"  Referencia t_π        = {t_pi:.1f}/ω_b  << T_mem  ✓")
