#!/usr/bin/env python3
"""
Sweet spot del 2QD vs 1QD: Π_n y R_n vs κ/ω_b
================================================

Barre κ/ω_b con λ fijo para encontrar el punto de operación óptimo
de cada sistema. Si Ω_eff(2QD) = √2·Ω_eff(1QD), el sweet spot
κ ≈ 10·Ω_eff se desplaza a κ más grande para el 2QD.

Figura con 4 paneles:
  (a) Π_2 vs κ  — 1QD y 2QD
  (b) Π_3 vs κ  — 1QD y 2QD
  (c) R_2 vs κ  — 1QD y 2QD
  (d) R_3 vs κ  — 1QD y 2QD

Parámetros: Ω/ω_b=0.2, λ/ω_b=0.1, γ/ω_b=0.0002, γ_φ/ω_b=0.0004,
J/ω_b=0.5 (solo 2QD).

Autor: Jhon S. García B. — Tesis UQ 2025
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import time
from scipy.special import factorial

# =============================================================================
# PARÁMETROS
# =============================================================================
omega_b   = 1.0
Omega     = 0.2
gamma     = 0.0002
gamma_phi = 0.0004
J         = 0.5
lam       = 0.1       # Fijo

n_bundle_list = [2, 3]
kappa_arr = np.logspace(-4, -0.5, 50)  # κ de 10⁻⁴ a ~0.3

# Truncamientos
Ncut_1qd = 20
Ncut_2qd = 20

# Barrido de Δ
n_Delta_opt = 25


# =============================================================================
# OPERADORES
# =============================================================================
def build_1qd(Nc):
    b = qt.destroy(Nc); nb = b.dag()*b; Ib = qt.qeye(Nc)
    sm = qt.sigmam(); Iq = qt.qeye(2)
    return {
        'b': qt.tensor(Iq, b), 'nb': qt.tensor(Iq, nb),
        'sm': qt.tensor(sm, Ib), 'sp': qt.tensor(sm.dag(), Ib),
        'pe': qt.tensor(sm.dag()*sm, Ib),
        'H_phonon': omega_b * qt.tensor(Iq, nb),
        'H_drive': Omega * (qt.tensor(sm, Ib) + qt.tensor(sm.dag(), Ib)),
    }

def build_2qd(Nc):
    b = qt.destroy(Nc); nb = b.dag()*b; Ib = qt.qeye(Nc)
    sm = qt.sigmam(); sp = sm.dag(); Iq = qt.qeye(2)
    ops = {
        'b': qt.tensor(Iq, Iq, b), 'nb': qt.tensor(Iq, Iq, nb),
        'sm1': qt.tensor(sm, Iq, Ib), 'sp1': qt.tensor(sp, Iq, Ib),
        'sm2': qt.tensor(Iq, sm, Ib), 'sp2': qt.tensor(Iq, sp, Ib),
        'pe1': qt.tensor(sp*sm, Iq, Ib), 'pe2': qt.tensor(Iq, sp*sm, Ib),
    }
    ops['ne'] = ops['pe1'] + ops['pe2']
    ops['H_phonon'] = omega_b * ops['nb']
    ops['H_drive'] = Omega * (ops['sm1']+ops['sp1']+ops['sm2']+ops['sp2'])
    ops['H_Forster'] = J * (ops['sp1']*ops['sm2'] + ops['sp2']*ops['sm1'])
    return ops


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
    for method in ('direct',):
        try:
            rho = qt.steadystate(H, c_ops, method=method, use_rcm=True)
            if validate_rho(rho):
                return rho
        except Exception:
            pass
    return None


# =============================================================================
# RESONANCIAS
# =============================================================================
def resonance_1qd(n):
    arg = (n * omega_b)**2 - 4 * Omega**2
    return (-np.sqrt(arg) if arg > 0 else -n*omega_b) + lam**2/omega_b

def resonance_2qd(n):
    arg = (n * omega_b)**2 - 8 * Omega**2
    return (-np.sqrt(arg) if arg > 0 else -n*omega_b) + lam**2/omega_b - J


# =============================================================================
# CÁLCULO DE PUREZA Y TASA CON OPTIMIZACIÓN DE Δ
# =============================================================================
def _eval_purity(ops_f, ops_t, kappa_val, Delta, is_2qd):
    """Evaluar pureza en un Δ dado. Retorna (pi, na_f, na_t)."""
    if is_2qd:
        H_f = (ops_f['H_phonon'] + Delta*ops_f['ne']
               + lam*ops_f['ne']*(ops_f['b']+ops_f['b'].dag())
               + ops_f['H_drive'] + ops_f['H_Forster'])
        c_f = [np.sqrt(kappa_val)*ops_f['b'],
               np.sqrt(gamma)*ops_f['sm1'], np.sqrt(gamma)*ops_f['sm2'],
               np.sqrt(gamma_phi)*ops_f['pe1'], np.sqrt(gamma_phi)*ops_f['pe2']]
        H_t = (ops_t['H_phonon'] + Delta*ops_t['ne']
               + lam*ops_t['ne']*(ops_t['b']+ops_t['b'].dag())
               + ops_t['H_drive'] + ops_t['H_Forster'])
        c_t = [np.sqrt(kappa_val)*ops_t['b'],
               np.sqrt(gamma)*ops_t['sm1'], np.sqrt(gamma)*ops_t['sm2'],
               np.sqrt(gamma_phi)*ops_t['pe1'], np.sqrt(gamma_phi)*ops_t['pe2']]
    else:
        H_f = (ops_f['H_phonon'] + Delta*ops_f['pe']
               + lam*ops_f['pe']*(ops_f['b']+ops_f['b'].dag())
               + ops_f['H_drive'])
        c_f = [np.sqrt(kappa_val)*ops_f['b'],
               np.sqrt(gamma)*ops_f['sm'],
               np.sqrt(gamma_phi)*ops_f['pe']]
        H_t = (ops_t['H_phonon'] + Delta*ops_t['pe']
               + lam*ops_t['pe']*(ops_t['b']+ops_t['b'].dag())
               + ops_t['H_drive'])
        c_t = [np.sqrt(kappa_val)*ops_t['b'],
               np.sqrt(gamma)*ops_t['sm'],
               np.sqrt(gamma_phi)*ops_t['pe']]

    rho_f = solve_ss(H_f, c_f)
    if rho_f is None:
        return np.nan, np.nan, np.nan
    na_f = np.real(qt.expect(ops_f['nb'], rho_f))

    rho_t = solve_ss(H_t, c_t)
    if rho_t is None:
        return np.nan, np.nan, np.nan
    na_t = np.real(qt.expect(ops_t['nb'], rho_t))

    na_n = max(na_f - na_t, 0.0)
    pi = na_n / na_f if na_f > 1e-30 else 0.0
    return pi, na_f, na_t


def compute_point(ops_f, ops_t, kappa_val, Delta_est, is_2qd):
    """
    Doble pasada: gruesa (±0.20, 15 pts) + fina (±3κ, 15 pts).
    Retorna (purity, rate, nbar, Delta_opt).
    """
    # ── Pasada gruesa ───────────────────────────────────────
    delta_window = 0.20 if is_2qd else 0.10
    Delta_coarse = np.linspace(Delta_est - delta_window,
                               Delta_est + delta_window,
                               n_Delta_opt)

    best_pi = -1.0
    best_D = Delta_est
    best_na_f = 0.0
    best_na_t = 0.0

    for Delta in Delta_coarse:
        pi, na_f, na_t = _eval_purity(ops_f, ops_t, kappa_val, Delta, is_2qd)
        if not np.isnan(pi) and pi > best_pi:
            best_pi = pi
            best_D = Delta
            best_na_f = na_f
            best_na_t = na_t

    # ── Pasada fina ─────────────────────────────────────────
    window_fine = max(3.0 * kappa_val, 0.003)
    Delta_fine = np.linspace(best_D - window_fine,
                             best_D + window_fine,
                             n_Delta_opt)

    for Delta in Delta_fine:
        pi, na_f, na_t = _eval_purity(ops_f, ops_t, kappa_val, Delta, is_2qd)
        if not np.isnan(pi) and pi > best_pi:
            best_pi = pi
            best_D = Delta
            best_na_f = na_f
            best_na_t = na_t

    na_n = max(best_na_f - best_na_t, 0.0)
    rate = kappa_val * na_n
    return best_pi, rate, best_na_f, best_D


# =============================================================================
# PRECONSTRUIR OPERADORES
# =============================================================================
print("Construyendo operadores...")
ops_1qd_f = build_1qd(Ncut_1qd)
ops_1qd_t = {n: build_1qd(n) for n in n_bundle_list}
ops_2qd_f = build_2qd(Ncut_2qd)
ops_2qd_t = {n: build_2qd(n) for n in n_bundle_list}

# Sweet spots teóricos
for n in n_bundle_list:
    Oeff_1 = Omega * (lam/omega_b)**n / np.sqrt(factorial(n))
    Oeff_2 = np.sqrt(2) * Oeff_1
    print(f"\n  n={n}:")
    print(f"    Ω_eff(1QD) = {Oeff_1:.4e}")
    print(f"    Ω_eff(2QD) = {Oeff_2:.4e}")
    print(f"    Sweet spot 1QD: κ ≈ 10·Ω_eff = {10*Oeff_1:.4e}")
    print(f"    Sweet spot 2QD: κ ≈ 10·Ω_eff = {10*Oeff_2:.4e}")


# =============================================================================
# BARRIDO PRINCIPAL
# =============================================================================
results = {}

for n_b in n_bundle_list:
    print(f"\n{'='*65}")
    print(f"  n = {n_b}: sweet spot 1QD vs 2QD, λ={lam}")
    print(f"  {len(kappa_arr)} puntos de κ")
    print(f"{'='*65}")

    pi_1 = np.full(len(kappa_arr), np.nan)
    pi_2 = np.full(len(kappa_arr), np.nan)
    R_1  = np.full(len(kappa_arr), np.nan)
    R_2  = np.full(len(kappa_arr), np.nan)
    na_1 = np.full(len(kappa_arr), np.nan)
    na_2 = np.full(len(kappa_arr), np.nan)
    D_1  = np.full(len(kappa_arr), np.nan)
    D_2  = np.full(len(kappa_arr), np.nan)

    Delta_est_1 = resonance_1qd(n_b)
    Delta_est_2 = resonance_2qd(n_b)

    t0 = time.time()
    for i, kap in enumerate(kappa_arr):
        pi_1[i], R_1[i], na_1[i], D_1[i] = compute_point(
            ops_1qd_f, ops_1qd_t[n_b], kap, Delta_est_1, False)
        pi_2[i], R_2[i], na_2[i], D_2[i] = compute_point(
            ops_2qd_f, ops_2qd_t[n_b], kap, Delta_est_2, True)

        if (i+1) % 10 == 0 or i == len(kappa_arr)-1:
            elapsed = time.time() - t0
            eta = elapsed/(i+1) * (len(kappa_arr)-i-1)
            print(f"  [{i+1:3d}/{len(kappa_arr)}]  κ={kap:.2e}  "
                  f"π₁={pi_1[i]:.4f}  π₂={pi_2[i]:.4f}  "
                  f"R₁={R_1[i]:.2e}  R₂={R_2[i]:.2e}  "
                  f"({elapsed:.0f}s, ETA ~{eta:.0f}s)")

    results[n_b] = {
        'pi_1qd': pi_1, 'pi_2qd': pi_2,
        'R_1qd': R_1, 'R_2qd': R_2,
        'na_1qd': na_1, 'na_2qd': na_2,
        'D_1qd': D_1, 'D_2qd': D_2,
    }


# =============================================================================
# EXPORTAR
# =============================================================================
np.savez('sweet_spot_data.npz',
         kappa_arr=kappa_arr, lam=lam,
         **{f'{k}_n{n}': v for n in n_bundle_list
            for k, v in results[n].items()})
print("\n✓ Datos exportados: sweet_spot_data.npz")


# =============================================================================
# VISUALIZACIÓN
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# Sweet spots teóricos para marcar
sweet_1qd = {n: 10 * Omega * (lam/omega_b)**n / np.sqrt(factorial(n))
             for n in n_bundle_list}
sweet_2qd = {n: np.sqrt(2) * sweet_1qd[n] for n in n_bundle_list}

for col, n_b in enumerate(n_bundle_list):
    r = results[n_b]

    # ── Fila 1: Pureza ──────────────────────────────────────
    ax = axes[0, col]
    ax.semilogx(kappa_arr, r['pi_1qd'], 'b-', lw=1.5, label='1QD')
    ax.semilogx(kappa_arr, r['pi_2qd'], 'r-', lw=1.5, label='2QD')
    ax.axvline(sweet_1qd[n_b], ls=':', color='blue', lw=0.8, alpha=0.6)
    ax.axvline(sweet_2qd[n_b], ls=':', color='red', lw=0.8, alpha=0.6)
    ax.set_ylabel(rf'$\Pi_{n_b}$', fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)
    label = '(a)' if col == 0 else '(b)'
    ax.set_title(rf'{label}  Pureza $\Pi_{n_b}$, $\lambda/\omega_b={lam}$',
                 fontsize=12)

    # ── Fila 2: Tasa ────────────────────────────────────────
    ax = axes[1, col]
    ax.loglog(kappa_arr, r['R_1qd'], 'b-', lw=1.5, label='1QD')
    ax.loglog(kappa_arr, r['R_2qd'], 'r-', lw=1.5, label='2QD')
    ax.axvline(sweet_1qd[n_b], ls=':', color='blue', lw=0.8, alpha=0.6)
    ax.axvline(sweet_2qd[n_b], ls=':', color='red', lw=0.8, alpha=0.6)
    ax.set_xlabel(r'$\kappa/\omega_b$', fontsize=13)
    ax.set_ylabel(rf'$R_{n_b} = \kappa \, n_a^{{({n_b})}}$', fontsize=13)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)
    label = '(c)' if col == 0 else '(d)'
    ax.set_title(rf'{label}  Tasa $R_{n_b}$', fontsize=12)

plt.tight_layout()
plt.savefig("sweet_spot_comparison.pdf", bbox_inches='tight')
plt.savefig("sweet_spot_comparison.png", dpi=200, bbox_inches='tight')
print("\n✓ Figuras guardadas: sweet_spot_comparison.{pdf,png}")
plt.show()

# =============================================================================
# DIAGNÓSTICO
# =============================================================================
print(f"\n{'='*65}")
print("  DIAGNÓSTICO — Sweet spots numéricos")
print(f"{'='*65}")

for n_b in n_bundle_list:
    r = results[n_b]

    # Encontrar κ que maximiza tasa con pureza > 0.9
    mask_1 = r['pi_1qd'] > 0.9
    mask_2 = r['pi_2qd'] > 0.9

    if np.any(mask_1):
        idx_1 = np.argmax(r['R_1qd'][mask_1])
        real_idx_1 = np.where(mask_1)[0][idx_1]
        kap_opt_1 = kappa_arr[real_idx_1]
    else:
        kap_opt_1 = np.nan

    if np.any(mask_2):
        idx_2 = np.argmax(r['R_2qd'][mask_2])
        real_idx_2 = np.where(mask_2)[0][idx_2]
        kap_opt_2 = kappa_arr[real_idx_2]
    else:
        kap_opt_2 = np.nan

    print(f"\n  n={n_b}:")
    print(f"    Sweet spot teórico 1QD: κ = {sweet_1qd[n_b]:.4e}")
    print(f"    Sweet spot teórico 2QD: κ = {sweet_2qd[n_b]:.4e}")
    print(f"    Sweet spot numérico 1QD (max R con Π>0.9): κ = {kap_opt_1:.4e}")
    print(f"    Sweet spot numérico 2QD (max R con Π>0.9): κ = {kap_opt_2:.4e}")
    if not np.isnan(kap_opt_1) and not np.isnan(kap_opt_2):
        print(f"    Ratio κ_opt(2QD)/κ_opt(1QD) = {kap_opt_2/kap_opt_1:.3f}")
        print(f"    R_max(1QD) = {r['R_1qd'][real_idx_1]:.4e} "
              f"(Π={r['pi_1qd'][real_idx_1]:.4f})")
        print(f"    R_max(2QD) = {r['R_2qd'][real_idx_2]:.4e} "
              f"(Π={r['pi_2qd'][real_idx_2]:.4f})")
        print(f"    Ratio R_max(2QD)/R_max(1QD) = "
              f"{r['R_2qd'][real_idx_2]/r['R_1qd'][real_idx_1]:.3f}")