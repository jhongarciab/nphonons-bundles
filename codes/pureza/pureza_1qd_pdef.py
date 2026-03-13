 #!/usr/bin/env python3
"""
Pureza Π_n — 1 QD (Bin et al.) — v3: Resonancia optimizada
=============================================================

Corrección principal: La resonancia de Stokes NO es Δ = -n·ω_b exactamente.
Con driving fuerte (Ω = 0.2) y acoplamiento e-ph (λ ≠ 0), hay correcciones:
  - Régimen III: Δ_n(Ω) = -√((n·ω_b)² - 4Ω²)
  - Régimen II:  Δ_n(λ) = -n·ω_b + λ²/ω_b

La resonancia es extremadamente estrecha (ancho ~ κ), así que estar fuera
por δΔ > κ mata la pureza.

Solución: Para cada punto (λ, κ), barrer Δ en una ventana estrecha
y tomar el máximo de π_n. Esto es lo que Bin hace implícitamente.

Método de pureza: Muñoz (Optica 2018) — dos steadystate:
  π_n = (n_a - n_a^(1)) / n_a

Autor: Jhon S. García B. — Tesis UQ 2025

DOCUMENTACIÓN OPERATIVA
-----------------------
Qué hace
- Calcula mapas de pureza π_n para 1QD en grilla (λ, κ), optimizando Δ por barrido local.

Entradas
- Parámetros físicos y numéricos definidos en la cabecera del script.

Salidas
- Figura principal (solo PDF) en carpeta de pruebas:
  codes/figs/pruebas/purity_1qd_maps.pdf

Convención
- Este script NO exporta PNG.
"""

import numpy as np
import qutip as qt
from math import factorial
import time


def odd(num):
    return num & 0x1

# =============================================================================
# PARÁMETROS — Bin Fig. 5
# =============================================================================
omega_b   = 1.0
Omega     = 0.2       # Ω/ω_b
gamma     = 0.0002    # γ/ω_b
gamma_phi = 0.0004    # γ_φ/ω_b

Ncut_full  = 20
pdef = -0.5 # Caso normal: 0.0 | Deformado: pdef ~ -0.49

# =============================================================================
# GRILLA 2D: λ/ω_b × κ/ω_b
# =============================================================================
n_lam   = 12          # Producción: 60-80
n_kappa = 12

lambda_arr = np.linspace(0.02, 0.14, n_lam)
kappa_arr  = np.logspace(-3, 0, n_kappa)

n_bundle_list = [2, 3]

# Número de puntos en el barrido fino de Δ para encontrar el máximo
n_Delta_opt = 15      # Producción: 25-30

# =============================================================================
# OPERADORES PRECONSTRUIDOS
# =============================================================================
def build_operators(Nc):
    """Construir operadores del sistema QD ⊗ Fock(Nc) con deformación pdef."""
    n_max = Nc - 1
    rad = np.add((np.arange(n_max) + 1), 2.0 * pdef * odd(np.arange(n_max) + 1))
    if np.any(rad < 0):
        raise ValueError(f"Radicando negativo en operador deformado para pdef={pdef}, Nc={Nc}")

    superdiag = np.sqrt(rad)
    a_np = np.diag(superdiag, 1)
    a_dag_np = a_np.T
    n_np = np.add(a_dag_np.dot(a_np), np.diag(-2.0 * pdef * odd(np.arange(n_max + 1))))

    b = qt.Qobj(a_np)
    nb = qt.Qobj(n_np)
    Ib = qt.qeye(Nc)
    sm = qt.sigmam()
    Iq = qt.qeye(2)

    ops = {
        'b':   qt.tensor(Iq, b),
        'nb':  qt.tensor(Iq, nb),
        'sm':  qt.tensor(sm, Ib),
        'sp':  qt.tensor(sm.dag(), Ib),
        'pe':  qt.tensor(sm.dag() * sm, Ib),
        'Iq':  Iq,
        'Ib':  Ib,
        'Nc':  Nc,
    }
    ops['H_phonon'] = omega_b * ops['nb']
    ops['H_drive']  = Omega * (ops['sm'] + ops['sp'])
    return ops


ops_full  = build_operators(Ncut_full)
ops_trunc = ops_trunc = {n: build_operators(n) for n in n_bundle_list}

# Proyectores de Fock
fock_projs = [
    qt.tensor(ops_full['Iq'], qt.fock_dm(Ncut_full, m))
    for m in range(Ncut_full)
]

# Operadores (b†)^n b^n
bdagn_bn = {
    n: ops_full['b'].dag()**n * ops_full['b']**n
    for n in n_bundle_list
}


# =============================================================================
# FUNCIONES
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


def resonance_estimate(n_bundle, lam):
    """
    Estimación de la resonancia de Stokes corregida.

    Combina las correcciones de Régimen II (Lamb shift) y III (driving):
      Δ_n ≈ -√((n·ω_b)² - 4Ω²) + λ²/ω_b
    """
    arg = (n_bundle * omega_b)**2 - 4 * Omega**2
    if arg < 0:
        # Fuera del régimen válido
        return -n_bundle * omega_b
    Delta_III = -np.sqrt(arg)
    Lamb_shift = lam**2 / omega_b
    return Delta_III + Lamb_shift


def compute_purity_at_Delta(ops_f, ops_t, lam, kappa, Delta):
    """
    Calcular π_n para un Delta dado. Retorna (purity, n_a, na1).
    """
    # Modelo completo
    H_f = (ops_f['H_phonon']
           + Delta * ops_f['pe']
           + lam * ops_f['pe'] * (ops_f['b'] + ops_f['b'].dag())
           + ops_f['H_drive'])
    c_f = [
        np.sqrt(kappa)     * ops_f['b'],
        np.sqrt(gamma)     * ops_f['sm'],
        np.sqrt(gamma_phi) * ops_f['pe'],
    ]
    rho_f = solve_ss(H_f, c_f)
    if rho_f is None:
        return np.nan, np.nan, np.nan, None

    n_a = np.real(qt.expect(ops_f['nb'], rho_f))

    # Modelo truncado
    H_t = (ops_t['H_phonon']
           + Delta * ops_t['pe']
           + lam * ops_t['pe'] * (ops_t['b'] + ops_t['b'].dag())
           + ops_t['H_drive'])
    c_t = [
        np.sqrt(kappa)     * ops_t['b'],
        np.sqrt(gamma)     * ops_t['sm'],
        np.sqrt(gamma_phi) * ops_t['pe'],
    ]
    rho_t = solve_ss(H_t, c_t)
    if rho_t is None:
        return np.nan, np.nan, np.nan, None

    na1 = np.real(qt.expect(ops_t['nb'], rho_t))
    na_n = max(n_a - na1, 0.0)

    if n_a > 1e-30:
        purity = na_n / n_a
    else:
        purity = 0.0

    return purity, n_a, na1, rho_f


def compute_optimized(lam, kappa, n_bundle):
    """
    Para un punto (λ, κ), encontrar la resonancia óptima barriendo Δ.

    1. Estimar Δ_n(λ, Ω) con la fórmula analítica
    2. Barrer Δ en ventana ±δ alrededor de la estimación
    3. Tomar el Δ que maximiza π_n

    Retorna dict con todos los observables en el Δ óptimo.
    """
    # Estimación central
    Delta_center = resonance_estimate(n_bundle, lam)

    # Ancho de la ventana de búsqueda: ±0.1 (>> κ típica)
    # Esto cubre variaciones por correcciones de orden superior
    delta_window = 0.1
    Delta_scan = np.linspace(Delta_center - delta_window,
                             Delta_center + delta_window,
                             n_Delta_opt)

    best_purity = -1.0
    best_result = None

    for Delta in Delta_scan:
        purity, n_a, na1, rho_f = compute_purity_at_Delta(
            ops_full, ops_trunc[n_bundle], lam, kappa, Delta)

        if np.isnan(purity):
            continue

        if purity > best_purity:
            best_purity = purity
            best_result = {
                'purity_munoz': purity,
                'nbar':         n_a,
                'na1':          na1,
                'Delta_opt':    Delta,
                'rho_full':     rho_f,
            }

    if best_result is None:
        return None

    # Con el ρ_ss óptimo, calcular los demás observables
    rho_f = best_result['rho_full']
    n_a   = best_result['nbar']

    # Poblaciones de Fock
    n_show = min(n_bundle + 3, Ncut_full)
    p_m = np.zeros(n_show)
    for m in range(n_show):
        p_m[m] = np.real(qt.expect(fock_projs[m], rho_f))

    # Pureza Fock
    if n_a > 1e-30 and n_bundle < Ncut_full:
        purity_fock = n_bundle * p_m[n_bundle] / n_a
    else:
        purity_fock = 0.0

    # T̃_n
    if n_a > 1e-30:
        bdnbn_val = np.real(qt.expect(bdagn_bn[n_bundle], rho_f))
        Tn = n_bundle * bdnbn_val / (factorial(n_bundle - 1) * n_a)
    else:
        Tn = 0.0

    best_result['purity_fock'] = purity_fock
    best_result['Tn']          = Tn
    best_result['p_m']         = p_m

    # Liberar ρ para no acumular memoria
    del best_result['rho_full']

    return best_result


# =============================================================================
# BARRIDO PRINCIPAL
# =============================================================================
all_results = {}

for n_b in n_bundle_list:
    print(f"\n{'='*65}")
    print(f"  Π_{n_b} — con optimización de resonancia")
    print(f"  Grilla: {n_lam}×{n_kappa} = {n_lam*n_kappa} puntos")
    print(f"  Δ scan: {n_Delta_opt} puntos por (λ,κ)")
    print(f"  Total steadystate: ~{n_lam*n_kappa*n_Delta_opt*2}")
    print(f"  Ncut_full={Ncut_full}, Ncut_trunc={n_b} (Fock({n_b}), dim={2*n_b})")
    print(f"  pdef={pdef}")
    print(f"{'='*65}")

    pm_map    = np.full((n_lam, n_kappa), np.nan)
    pf_map    = np.full((n_lam, n_kappa), np.nan)
    tn_map    = np.full((n_lam, n_kappa), np.nan)
    nb_map    = np.full((n_lam, n_kappa), np.nan)
    delta_map = np.full((n_lam, n_kappa), np.nan)

    t0 = time.time()
    total = n_lam * n_kappa
    count = 0

    for i, lam in enumerate(lambda_arr):
        for j, kap in enumerate(kappa_arr):
            res = compute_optimized(lam, kap, n_b)
            count += 1

            if res is not None:
                pm_map[i, j]    = res['purity_munoz']
                pf_map[i, j]    = res['purity_fock']
                tn_map[i, j]    = res['Tn']
                nb_map[i, j]    = res['nbar']
                delta_map[i, j] = res['Delta_opt']

            if count % 25 == 0 or count == total:
                elapsed = time.time() - t0
                rate = count / elapsed if elapsed > 0 else 0
                eta = (total - count) / rate if rate > 0 else 0
                pv = pm_map[i, j] if not np.isnan(pm_map[i, j]) else -1
                dv = delta_map[i, j] if not np.isnan(delta_map[i, j]) else 0
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
# VISUALIZACIÓN — 2 filas × 2 columnas
# =============================================================================
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

n_cols = len(n_bundle_list)
fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 5))
axes = axes.reshape(1, n_cols)

cmap_pur   = plt.cm.RdYlBu_r

for col, n_b in enumerate(n_bundle_list):

    # ─────────────────────────────
    # Fila 1 — Pureza π_n
    # ─────────────────────────────
    ax = axes[0, col]

    pmap = all_results[n_b]['munoz']

    Z = np.nan_to_num(np.clip(pmap,0,1),nan=0).T

    im1 = ax.pcolormesh(
        lambda_arr,
        kappa_arr,
        Z,
        shading='auto',
        cmap=cmap_pur,
        norm=Normalize(vmin=0,vmax=1)
    )

    ax.set_yscale('log')

    ax.set_xlim(0.02,0.14)
    ax.set_ylim(1e-3,1e0)

    ax.set_xlabel(r'$\lambda/\omega_b$',fontsize=13)
    ax.set_ylabel(r'$\kappa/\omega_b$',fontsize=13)

    ax.set_title(rf'$\pi_{n_b}$ (Muñoz, $\Delta$ optimizado)',fontsize=13)

    ax.tick_params(labelsize=11)

    cbar1 = fig.colorbar(im1,ax=ax,fraction=0.045,pad=0.02)
    cbar1.set_label(rf'$\pi_{n_b}$',fontsize=12)

    label = chr(ord('a')+col)

    ax.text(
        0.03,
        0.95,
        f'({label})',
        transform=ax.transAxes,
        fontsize=13,
        va='top',
        ha='left',
        bbox=dict(facecolor='white',edgecolor='none',alpha=0.7)
    )

plt.tight_layout()
plt.savefig(f"pureza{n_bundle_list}_pdef_{pdef:.2f}.png", dpi=300)

# =============================================================================
# DIAGNÓSTICO
# =============================================================================
print("\n" + "="*65)
print("  DIAGNÓSTICO — λ=0.1, κ=0.002")
print("="*65)

for n_b in n_bundle_list:
    res = compute_optimized(0.10, 0.002, n_b)
    if res is None:
        print(f"\n  n={n_b}: falló")
        continue

    Delta_naive = -n_b * omega_b
    Delta_est   = resonance_estimate(n_b, 0.10)

    print(f"\n  n={n_b}:")
    print(f"    Δ naive      = {Delta_naive:.4f}")
    print(f"    Δ estimado   = {Delta_est:.4f}")
    print(f"    Δ óptimo     = {res['Delta_opt']:.4f}")
    print(f"    ⟨n̂⟩ completo = {res['nbar']:.6e}")
    print(f"    n_a^(1)      = {res['na1']:.6e}")
    print(f"    π_{n_b} Muñoz = {res['purity_munoz']:.6f}")
    print(f"    Π_{n_b} Fock  = {res['purity_fock']:.6f}")
    print(f"    T̃_{n_b}       = {res['Tn']:.6f}")
    print(f"    p(m): {['%.2e' % x for x in res['p_m']]}")

print("\n" + "="*65)
print("  CÁLCULO COMPLETO")
print("="*65)