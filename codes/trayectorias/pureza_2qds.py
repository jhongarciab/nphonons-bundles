#!/usr/bin/env python3
"""
Pureza Π_n — Molécula excitónica (2QDs + Förster)
===================================================

Corrección respecto a v1: el modelo truncado usa Ncut = n_bundle
(no Ncut=2 fijo), de modo que suprime exactamente los procesos de
orden ≥ n y el fondo estimado es n_a^(≤n-1), no solo n_a^(1).

Descomposición de Muñoz correcta para orden n:

  ⟨n̂⟩_full  = n_a^(1) + n_a^(2) + ... + n_a^(n) + ...
  ⟨n̂⟩_trunc = n_a^(1) + ... + n_a^(n-1)   [Fock truncado en n-1 máx]

  →  n_a^(n) ≈ ⟨n̂⟩_full - ⟨n̂⟩_trunc
  →  π_n     = n_a^(n) / ⟨n̂⟩_full

El modelo truncado tiene espacio de Fock(n_bundle), cuyos estados
van de |0⟩ a |n_bundle-1⟩, bloqueando cualquier proceso que requiera
≥ n_bundle fonones simultáneos.

Hamiltoniano 2QD (marco rotante, derivaciones.tex):
  H = ω_b b†b + Δ(σ₁†σ₁+σ₂†σ₂) + λ(σ₁†σ₁+σ₂†σ₂)(b†+b)
      + J(σ₁†σ₂+σ₂†σ₁) + Ω(σ₁+σ₁†+σ₂+σ₂†)

Espacio completo: C²⊗C²⊗Fock(Ncut_full), dim = 4·Ncut_full
Espacio truncado: C²⊗C²⊗Fock(n_bundle),   dim = 4·n_bundle

Resonancias (derivaciones.tex):
  Régimen I:   Δ_n = -n·ω_b - J
  Régimen II:  Δ_n = λ²/ω_b - n·ω_b - J
  Régimen III: Δ_n = -√(n²ω_b² - 8Ω²) - J   [factor 8 por √2-enhancement 2QD]

Autor: Jhon S. García B. — Tesis UQ 2025
"""

import numpy as np
import qutip as qt
from math import factorial
import time

# =============================================================================
# PARÁMETROS — 2QD producción
# =============================================================================
omega_b   = 1.0
Omega     = 0.2       # Ω/ω_b
gamma     = 0.0002    # γ/ω_b   (decaimiento espontáneo)
gamma_phi = 0.0004    # γ_φ/ω_b (dephasing puro)
J         = 0.5       # J/ω_b   (acoplamiento de Förster)

# Ncut del modelo completo — debe ser > max(n_bundle) + margen
Ncut_full = 14        # dim = 4·14 = 56

# =============================================================================
# GRILLA 2D: λ/ω_b × κ/ω_b
# =============================================================================
n_lam   = 30          # Producción: 50-60
n_kappa = 30

lambda_arr = np.linspace(0.08, 0.26, n_lam)
kappa_arr  = np.logspace(-5, -1, n_kappa)

n_bundle_list = [3, 4]

# Puntos en el barrido de Δ para optimización
n_Delta_opt = 17


# =============================================================================
# CONSTRUCCIÓN DE OPERADORES — QD₁ ⊗ QD₂ ⊗ Fock(Nc)
# =============================================================================
def build_2qd_operators(Nc):
    """
    Construye todos los operadores del sistema 2QD+fonón para un
    espacio de Fock de dimensión Nc.

    Retorna diccionario con operadores y partes fijas del Hamiltoniano.
    """
    b  = qt.destroy(Nc)
    nb = b.dag() * b
    Ib = qt.qeye(Nc)
    sm = qt.sigmam()
    sp = sm.dag()
    Iq = qt.qeye(2)

    ops = {}
    # Modo fonónico
    ops['b']  = qt.tensor(Iq, Iq, b)
    ops['nb'] = qt.tensor(Iq, Iq, nb)

    # QD₁ (actúa en el 1er espacio de qubit)
    ops['sm1'] = qt.tensor(sm, Iq, Ib)
    ops['sp1'] = ops['sm1'].dag()
    ops['pe1'] = ops['sp1'] * ops['sm1']   # proyector |e⟩⟨e| QD₁

    # QD₂ (actúa en el 2do espacio de qubit)
    ops['sm2'] = qt.tensor(Iq, sm, Ib)
    ops['sp2'] = ops['sm2'].dag()
    ops['pe2'] = ops['sp2'] * ops['sm2']   # proyector |e⟩⟨e| QD₂

    # Operador de número electrónico total n̂_e = |e⟩⟨e|₁ + |e⟩⟨e|₂
    ops['ne_total'] = ops['pe1'] + ops['pe2']

    # Partes del Hamiltoniano independientes de (Δ, λ, κ)
    ops['H_phonon']  = omega_b * ops['nb']
    ops['H_drive']   = Omega * (ops['sm1'] + ops['sp1'] + ops['sm2'] + ops['sp2'])
    ops['H_Forster'] = J * (ops['sp1'] * ops['sm2'] + ops['sp2'] * ops['sm1'])

    ops['Nc'] = Nc
    ops['Iq'] = Iq
    ops['Ib'] = Ib

    return ops


# Modelo completo — preconstruido (reutilizado en todo el barrido)
ops_full = build_2qd_operators(Ncut_full)

# Proyectores de Fock |m⟩⟨m| en el espacio completo (para poblaciones p_m)
fock_projs_full = [
    qt.tensor(ops_full['Iq'], ops_full['Iq'], qt.fock_dm(Ncut_full, m))
    for m in range(Ncut_full)
]

# Operadores (b†)^n b^n en el espacio completo (para T̃_n)
bdagn_bn_full = {
    n: ops_full['b'].dag()**n * ops_full['b']**n
    for n in n_bundle_list
}

# Modelos truncados — uno por cada n_bundle
# Fock(n_bundle) tiene estados |0⟩ ... |n_bundle-1⟩
# → suprime procesos que requieran ≥ n_bundle fonones simultáneos
ops_trunc = {n: build_2qd_operators(n) for n in n_bundle_list}


# =============================================================================
# UTILIDADES
# =============================================================================
def validate_rho(rho, tol=1e-8):
    """Verifica traza, hermiticidad y positividad de ρ."""
    if abs(rho.tr() - 1.0) > 1e-6:
        return False
    if not rho.isherm:
        return False
    evals = np.real(rho.eigenstates()[0])
    if np.min(evals) < -tol:
        return False
    return True


def solve_ss(H, c_ops):
    """
    Resuelve la ecuación maestra de Lindblad en estado estacionario.
    Intenta tres métodos en orden de robustez numérica.
    """
    for method in ('direct', 'eigen', 'svd'):
        try:
            rho = qt.steadystate(H, c_ops, method=method, use_rcm=True)
            if validate_rho(rho):
                return rho
        except Exception:
            pass
    return None


def build_H_2qd(ops, lam, Delta):
    """
    Hamiltoniano completo del sistema 2QD para parámetros (λ, Δ).

    H = ω_b b†b + Δ·n̂_e + λ·n̂_e·(b+b†) + H_drive + H_Förster
    """
    return (ops['H_phonon']
            + Delta * ops['ne_total']
            + lam   * ops['ne_total'] * (ops['b'] + ops['b'].dag())
            + ops['H_drive']
            + ops['H_Forster'])


def build_cops_2qd(ops, kappa):
    """
    Operadores de colapso de Lindblad para un κ dado.

    L₁ = √κ · b          (decaimiento fonónico)
    L₂ = √γ · σ₁⁻        (emisión espontánea QD₁)
    L₃ = √γ · σ₂⁻        (emisión espontánea QD₂)
    L₄ = √γ_φ · σ₁†σ₁   (dephasing QD₁)
    L₅ = √γ_φ · σ₂†σ₂   (dephasing QD₂)
    """
    return [
        np.sqrt(kappa)     * ops['b'],
        np.sqrt(gamma)     * ops['sm1'],
        np.sqrt(gamma)     * ops['sm2'],
        np.sqrt(gamma_phi) * ops['pe1'],
        np.sqrt(gamma_phi) * ops['pe2'],
    ]


def resonance_estimate_2qd(n_bundle, lam):
    """
    Estimación del detuning de Stokes óptimo para el 2QD.

    Combina Régimen III + Lamb shift + desplazamiento Förster:
      Δ_n ≈ -√(n²ω_b² - 8Ω²) + λ²/ω_b - J

    El factor 8Ω² (en lugar de 2Ω² del 1QD) refleja el factor √2
    de enhancement superradiante del estado brillante del 2QD:
      Ω_eff^(n) = √2 · (λ/ω_b)^n · Ω / √(n!)
    lo que produce una frecuencia de Rabi vestida 2 veces mayor en la
    condición de resonancia del Régimen III.
    """
    arg = (n_bundle * omega_b)**2 - 8.0 * Omega**2
    if arg < 0:
        # Régimen I como fallback si Ω es demasiado grande
        return -n_bundle * omega_b - J
    Delta_III  = -np.sqrt(arg)
    Lamb_shift = lam**2 / omega_b
    return Delta_III + Lamb_shift - J


# =============================================================================
# CÁLCULO DE PUREZA EN UN PUNTO (Δ, λ, κ) — DESCOMPOSICIÓN DE MUÑOZ
# =============================================================================
def compute_purity_at_Delta_2qd(lam, kappa, Delta, n_bundle):
    """
    Calcula π_n en un (λ, κ, Δ) dado usando la descomposición de Muñoz.

    Estrategia:
      1. Resolver ρ_ss en el espacio COMPLETO → ⟨n̂⟩_full
      2. Resolver ρ_ss en el espacio TRUNCADO a Fock(n_bundle)
         → ⟨n̂⟩_trunc  (contiene solo procesos de orden 1 a n-1)
      3. n_a^(n) = ⟨n̂⟩_full - ⟨n̂⟩_trunc
      4. π_n = n_a^(n) / ⟨n̂⟩_full

    El truncado en Fock(n_bundle) bloquea exactamente los procesos de
    n o más fonones, porque el estado |n⟩ no existe en ese espacio.

    Retorna: (purity, n_a_full, n_a_trunc, rho_full)
             o (nan, nan, nan, None) si algún solver falla.
    """
    # ── Modelo completo ──────────────────────────────────────────────
    H_f   = build_H_2qd(ops_full, lam, Delta)
    c_f   = build_cops_2qd(ops_full, kappa)
    rho_f = solve_ss(H_f, c_f)
    if rho_f is None:
        return np.nan, np.nan, np.nan, None

    n_a_full = np.real(qt.expect(ops_full['nb'], rho_f))

    # ── Modelo truncado en Fock(n_bundle) ────────────────────────────
    ops_t  = ops_trunc[n_bundle]
    H_t    = build_H_2qd(ops_t, lam, Delta)
    c_t    = build_cops_2qd(ops_t, kappa)
    rho_t  = solve_ss(H_t, c_t)
    if rho_t is None:
        return np.nan, np.nan, np.nan, None

    n_a_trunc = np.real(qt.expect(ops_t['nb'], rho_t))

    # ── Fracción de n-fonones ─────────────────────────────────────────
    n_a_n = max(n_a_full - n_a_trunc, 0.0)

    if n_a_full > 1e-30:
        purity = n_a_n / n_a_full
    else:
        purity = 0.0

    return purity, n_a_full, n_a_trunc, rho_f


# =============================================================================
# OPTIMIZACIÓN SOBRE Δ PARA UN PUNTO (λ, κ)
# =============================================================================
def compute_optimized_2qd(lam, kappa, n_bundle):
    """
    Para un punto (λ, κ) y un orden n_bundle, encuentra el Δ que
    maximiza π_n mediante un barrido de n_Delta_opt puntos centrado
    en la estimación resonante.

    Además del resultado óptimo, calcula observables secundarios:
      - purity_fock: pureza basada en población p(n)
      - Tn: parámetro T̃_n de Muñoz 2018 (Eq. S25)
      - p_m: distribución de Fock p(0)...p(n+3)
    """
    Delta_center = resonance_estimate_2qd(n_bundle, lam)

    # Ventana ±0.20 — más ancha que en 1QD para absorber incertidumbre en J
    delta_window = 0.20
    Delta_scan   = np.linspace(Delta_center - delta_window,
                               Delta_center + delta_window,
                               n_Delta_opt)

    best_purity = -1.0
    best_result = None

    for Delta in Delta_scan:
        purity, n_a_full, n_a_trunc, rho_f = compute_purity_at_Delta_2qd(
            lam, kappa, Delta, n_bundle)

        if np.isnan(purity):
            continue

        if purity > best_purity:
            best_purity = purity
            best_result = {
                'purity_munoz': purity,
                'nbar':         n_a_full,
                'n_a_trunc':    n_a_trunc,
                'Delta_opt':    Delta,
                'rho_full':     rho_f,
            }

    if best_result is None:
        return None

    # ── Observables adicionales desde ρ_ss óptimo ────────────────────
    rho_f  = best_result['rho_full']
    n_a    = best_result['nbar']

    # Distribución de Fock p(m) = ⟨m|ρ_fonón|m⟩ para m=0,...,n+3
    n_show = min(n_bundle + 3, Ncut_full)
    p_m    = np.array([
        np.real(qt.expect(fock_projs_full[m], rho_f))
        for m in range(n_show)
    ])

    # Pureza de Fock: π_n^Fock = n·p(n) / ⟨n̂⟩
    # Válida bajo cascada perfecta (Eq. S24 de Muñoz 2018)
    if n_a > 1e-30 and n_bundle < Ncut_full:
        purity_fock = n_bundle * p_m[n_bundle] / n_a
    else:
        purity_fock = 0.0

    # T̃_n = n·⟨(b†)^n b^n⟩ / [(n-1)!·⟨n̂⟩]
    # Aproxima π_n vía correlaciones de Glauber de orden n (Muñoz 2018)
    if n_a > 1e-30:
        bdnbn_val = np.real(qt.expect(bdagn_bn_full[n_bundle], rho_f))
        Tn = n_bundle * bdnbn_val / (factorial(n_bundle - 1) * n_a)
    else:
        Tn = 0.0

    best_result['purity_fock'] = purity_fock
    best_result['Tn']          = Tn
    best_result['p_m']         = p_m

    del best_result['rho_full']   # liberar memoria

    return best_result


# =============================================================================
# BARRIDO PRINCIPAL 2D: λ × κ
# =============================================================================
all_results = {}

for n_b in n_bundle_list:

    # Ncut del truncado correcto para este n_b
    Nc_t = n_b   # Fock(n_b) → estados |0⟩ ... |n_b-1⟩

    print(f"\n{'='*65}")
    print(f"  Π_{n_b} — 2QD (Förster J={J})")
    print(f"  Resonancia Régimen I: Δ = -{n_b}·ω_b - J = {-n_b*omega_b - J:.4f}")
    print(f"  Grilla: {n_lam}×{n_kappa} = {n_lam*n_kappa} puntos")
    print(f"  Δ scan: {n_Delta_opt} pts/punto, ventana ±0.20")
    print(f"  Ncut_full={Ncut_full} (dim={4*Ncut_full})")
    print(f"  Ncut_trunc={Nc_t} (Fock(n_b), dim={4*Nc_t})  ← correcto para n={n_b}")
    print(f"  Ω={Omega}, λ∈[{lambda_arr[0]:.2f},{lambda_arr[-1]:.2f}]")
    print(f"  γ={gamma}, γ_φ={gamma_phi}, κ∈[{kappa_arr[0]:.1e},{kappa_arr[-1]:.1e}]")
    print(f"{'='*65}")

    pm_map    = np.full((n_lam, n_kappa), np.nan)
    pf_map    = np.full((n_lam, n_kappa), np.nan)
    tn_map    = np.full((n_lam, n_kappa), np.nan)
    nb_map    = np.full((n_lam, n_kappa), np.nan)
    delta_map = np.full((n_lam, n_kappa), np.nan)

    t0    = time.time()
    total = n_lam * n_kappa
    count = 0

    for i, lam in enumerate(lambda_arr):
        for j, kap in enumerate(kappa_arr):

            res = compute_optimized_2qd(lam, kap, n_b)
            count += 1

            if res is not None:
                pm_map[i, j]    = res['purity_munoz']
                pf_map[i, j]    = res['purity_fock']
                tn_map[i, j]    = res['Tn']
                nb_map[i, j]    = res['nbar']
                delta_map[i, j] = res['Delta_opt']

            if count % 25 == 0 or count == total:
                elapsed = time.time() - t0
                rate    = count / elapsed if elapsed > 0 else 0
                eta     = (total - count) / rate if rate > 0 else 0
                pv = pm_map[i, j]    if not np.isnan(pm_map[i, j])    else -1.0
                dv = delta_map[i, j] if not np.isnan(delta_map[i, j]) else 0.0
                print(f"  [{count:5d}/{total}]  λ={lam:.3f}  κ={kap:.1e}  "
                      f"π_{n_b}={pv:.4f}  Δ_opt={dv:.4f}  "
                      f"({elapsed:.0f}s, ETA ~{eta:.0f}s)")

    elapsed_total = time.time() - t0
    print(f"\n  ✓ Π_{n_b} completado en {elapsed_total:.1f}s")

    all_results[n_b] = {
        'munoz': pm_map,
        'fock':  pf_map,
        'Tn':    tn_map,
        'nbar':  nb_map,
        'delta': delta_map,
    }

# =============================================================================
# VISUALIZACIÓN — Heatmaps 2D
# =============================================================================
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

n_cols = len(n_bundle_list)
fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 5))
axes = axes.reshape(1, n_cols)

cmap_pur   = plt.cm.RdYlBu_r

for col, n_b in enumerate(n_bundle_list):

    # ── Fila 0: Pureza π_n (Muñoz) ───────────────────────────────────
    ax   = axes[0, col]
    pmap = all_results[n_b]['munoz']
    Z    = np.nan_to_num(np.clip(pmap, 0, 1), nan=0).T

    im1 = ax.pcolormesh(
        lambda_arr, kappa_arr, Z,
        shading='auto', cmap=cmap_pur,
        norm=Normalize(vmin=0, vmax=1)
    )
    ax.set_yscale('log')
    ax.set_xlabel(r'$\lambda/\omega_b$', fontsize=13)
    ax.set_ylabel(r'$\kappa/\omega_b$', fontsize=13)
    ax.set_title(rf'$\pi_{n_b}$ — 2QD ($J={J}$, $\Omega={Omega}$)', fontsize=13)
    ax.tick_params(labelsize=11)

    cbar1 = fig.colorbar(im1, ax=ax, fraction=0.045, pad=0.02)
    cbar1.set_label(rf'$\pi_{n_b}$', fontsize=12)

    label = chr(ord('a') + col)
    ax.text(0.03, 0.95, f'({label})', transform=ax.transAxes,
            fontsize=13, va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

plt.tight_layout()
plt.savefig("purity_2qd.png", dpi=300, bbox_inches='tight')
plt.savefig("purity_2qd.pdf", bbox_inches='tight')
print("\n✓ Heatmaps: purity_2qd.png / purity_2qd.pdf")


# =============================================================================
# VISUALIZACIÓN — Cortes 1D
# =============================================================================
fig2, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(13, 5))

lam_cuts  = [0.06, 0.10, 0.14]
kap_cuts  = [1e-4, 5e-4, 1e-3]
styles    = ['-', '--', ':']

# Cortes a λ fija: π_n vs κ
for k, lam_val in enumerate(lam_cuts):
    idx = np.argmin(np.abs(lambda_arr - lam_val))
    for n_b in n_bundle_list:
        color = 'C0' if n_b == 3 else 'C3'
        pmap  = all_results[n_b]['munoz']
        ax_c.plot(kappa_arr, pmap[idx, :],
                  ls=styles[k], color=color, lw=1.8,
                  label=rf'$\pi_{n_b}$, $\lambda={lambda_arr[idx]:.2f}$')

ax_c.set_xscale('log')
ax_c.set_xlabel(r'$\kappa/\omega_b$', fontsize=14)
ax_c.set_ylabel(r'$\pi_n$', fontsize=14)
ax_c.set_ylim(-0.05, 1.05)
ax_c.legend(fontsize=8, ncol=2, loc='lower left')
ax_c.set_title(r'2QD — $\lambda/\omega_b$ fija', fontsize=13)
ax_c.tick_params(labelsize=12)
ax_c.axhline(0.95, ls=':', color='gray', alpha=0.5)

# Cortes a κ fija: π_n vs λ
for k, kap_val in enumerate(kap_cuts):
    idx = np.argmin(np.abs(kappa_arr - kap_val))
    for n_b in n_bundle_list:
        color = 'C0' if n_b == 3 else 'C3'
        pmap  = all_results[n_b]['munoz']
        ax_d.plot(lambda_arr, pmap[:, idx],
                  ls=styles[k], color=color, lw=1.8,
                  label=rf'$\pi_{n_b}$, $\kappa={kappa_arr[idx]:.1e}$')

ax_d.set_xlabel(r'$\lambda/\omega_b$', fontsize=14)
ax_d.set_ylabel(r'$\pi_n$', fontsize=14)
ax_d.set_ylim(-0.05, 1.05)
ax_d.legend(fontsize=8, ncol=2, loc='lower right')
ax_d.set_title(r'2QD — $\kappa/\omega_b$ fija', fontsize=13)
ax_d.tick_params(labelsize=12)
ax_d.axhline(0.95, ls=':', color='gray', alpha=0.5)

plt.tight_layout()
plt.savefig("purity_2qd_cuts1.png", dpi=300, bbox_inches='tight')
print("✓ Cortes: purity_2qd_cuts.png")


# =============================================================================
# DIAGNÓSTICO — puntos de referencia
# =============================================================================
def run_diagnostico(lam_test, kap_test):
    print(f"\n{'='*65}")
    print(f"  DIAGNÓSTICO 2QD — λ={lam_test}, κ={kap_test}, J={J}")
    print(f"{'='*65}")

    for n_b in n_bundle_list:
        res = compute_optimized_2qd(lam_test, kap_test, n_b)
        if res is None:
            print(f"\n  n={n_b}: solver falló en todos los Δ")
            continue

        Delta_I   = -n_b * omega_b - J
        Delta_est = resonance_estimate_2qd(n_b, lam_test)

        # Verificar jerarquía γ ≪ κ
        flag_gamma = "⚠ κ < γ" if kap_test < gamma else "✓"

        print(f"\n  n={n_b} (Ncut_trunc={n_b}):  {flag_gamma}")
        print(f"    Δ Régimen I   = {Delta_I:.4f}")
        print(f"    Δ estimado    = {Delta_est:.4f}")
        print(f"    Δ óptimo      = {res['Delta_opt']:.4f}")
        print(f"    ⟨n̂⟩ completo  = {res['nbar']:.6e}")
        print(f"    ⟨n̂⟩ truncado  = {res['n_a_trunc']:.6e}   [fondo ≤ {n_b-1} fonones]")
        print(f"    π_{n_b} Muñoz  = {res['purity_munoz']:.6f}")
        print(f"    Π_{n_b} Fock   = {res['purity_fock']:.6f}")
        print(f"    T̃_{n_b}        = {res['Tn']:.6f}")
        print(f"    p(m):  {['%.2e' % x for x in res['p_m']]}")


run_diagnostico(lam_test=0.08, kap_test=0.003)
run_diagnostico(lam_test=0.14, kap_test=0.003)

print(f"\n{'='*65}")
print("  CÁLCULO COMPLETO")
print(f"{'='*65}")