#!/usr/bin/env python3
"""
Pureza Π_n vs γ_φ/ω_b — 2QD (n=2 y n=3 en un solo panel)
==========================================================

Parámetros de producción consistentes con pureza_2qds.py:
  Ω/ω_b=0.2, λ/ω_b=0.1, κ/ω_b=0.002, γ/ω_b=0.0002,
  γ_φ/ω_b=0.0004 (referencia, marcada con línea vertical),
  J/ω_b=0.5.

Método: Muñoz (dos steadystate), Δ optimizado en cada γ_φ.

Autor: Jhon S. García B. — Tesis UQ 2025
"""

import numpy as np
import qutip as qt
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time

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
# PARÁMETROS (idénticos a pureza_2qds.py)
# -----------------------------------------------------------------------------
omega_b   = 1.0
Omega     = 0.2
lam       = 0.1
kappa     = 0.002
gamma     = 0.0002
gamma_phi_ref = 0.0004   # referencia marcada en la figura
J         = 0.5

n_bundle_list = [2, 3]
gphi_arr  = np.logspace(-6, -1, 60)

# Truncamientos
Ncut_full = 16   # modelo completo 2QD
n_Delta_opt = 20

# Tamaño de figura — estilo oficial, más horizontal
FIG_W = 5.20
FIG_H = 2.25


# -----------------------------------------------------------------------------
# OPERADORES 2QD
# -----------------------------------------------------------------------------
def build_2qd(Nc):
    b = qt.destroy(Nc); nb = b.dag()*b; Ib = qt.qeye(Nc)
    sm = qt.sigmam(); sp = sm.dag(); Iq = qt.qeye(2)
    ops = {
        'b':  qt.tensor(Iq, Iq, b),
        'nb': qt.tensor(Iq, Iq, nb),
        'sm1': qt.tensor(sm, Iq, Ib), 'sp1': qt.tensor(sp, Iq, Ib),
        'sm2': qt.tensor(Iq, sm, Ib), 'sp2': qt.tensor(Iq, sp, Ib),
        'pe1': qt.tensor(sp*sm, Iq, Ib),
        'pe2': qt.tensor(Iq, sp*sm, Ib),
    }
    ops['ne'] = ops['pe1'] + ops['pe2']
    ops['H_phonon'] = omega_b * ops['nb']
    ops['H_drive']  = Omega * (ops['sm1']+ops['sp1']+ops['sm2']+ops['sp2'])
    ops['H_Forster']= J * (ops['sp1']*ops['sm2'] + ops['sp2']*ops['sm1'])
    return ops


# -----------------------------------------------------------------------------
# SOLVER
# -----------------------------------------------------------------------------
def solve_ss(H, c_ops):
    try:
        rho = qt.steadystate(H, c_ops, method='direct', use_rcm=True)
        if abs(rho.tr() - 1.0) < 1e-6 and rho.isherm:
            return rho
    except Exception:
        pass
    return None


# -----------------------------------------------------------------------------
# RESONANCIA 2QD (Régimen III + Lamb + Förster)
# -----------------------------------------------------------------------------
def resonance_2qd(n):
    arg = (n * omega_b)**2 - 8.0 * Omega**2
    base = -np.sqrt(arg) if arg > 0 else -n * omega_b
    return base + lam**2 / omega_b - J


# -----------------------------------------------------------------------------
# PUREZA EN UN PUNTO (γ_φ, n_bundle) — doble pasada en Δ
# -----------------------------------------------------------------------------
def compute_purity(ops_f, ops_t, gphi_val, Delta_est, n_bundle):
    """
    Retorna (purity, Delta_opt) usando descomposición de Muñoz:
      π_n = (⟨n̂⟩_full - ⟨n̂⟩_trunc) / ⟨n̂⟩_full
    """
    def _eval(Delta):
        H_f = (ops_f['H_phonon'] + Delta*ops_f['ne']
               + lam*ops_f['ne']*(ops_f['b']+ops_f['b'].dag())
               + ops_f['H_drive'] + ops_f['H_Forster'])
        c_f = [np.sqrt(kappa)*ops_f['b'],
               np.sqrt(gamma)*ops_f['sm1'], np.sqrt(gamma)*ops_f['sm2'],
               np.sqrt(gphi_val)*ops_f['pe1'], np.sqrt(gphi_val)*ops_f['pe2']]
        H_t = (ops_t['H_phonon'] + Delta*ops_t['ne']
               + lam*ops_t['ne']*(ops_t['b']+ops_t['b'].dag())
               + ops_t['H_drive'] + ops_t['H_Forster'])
        c_t = [np.sqrt(kappa)*ops_t['b'],
               np.sqrt(gamma)*ops_t['sm1'], np.sqrt(gamma)*ops_t['sm2'],
               np.sqrt(gphi_val)*ops_t['pe1'], np.sqrt(gphi_val)*ops_t['pe2']]

        rho_f = solve_ss(H_f, c_f)
        if rho_f is None:
            return None, None
        rho_t = solve_ss(H_t, c_t)
        if rho_t is None:
            return None, None

        na_f = np.real(qt.expect(ops_f['nb'], rho_f))
        na_t = np.real(qt.expect(ops_t['nb'], rho_t))
        na_n = max(na_f - na_t, 0.0)
        pi   = na_n / na_f if na_f > 1e-30 else 0.0
        return pi, na_f

    # Pasada gruesa ±0.20
    best_pi, best_D = -1.0, Delta_est
    for D in np.linspace(Delta_est - 0.20, Delta_est + 0.20, n_Delta_opt):
        pi, _ = _eval(D)
        if pi is not None and pi > best_pi:
            best_pi, best_D = pi, D

    # Pasada fina ±3κ
    for D in np.linspace(best_D - 3*kappa, best_D + 3*kappa, n_Delta_opt):
        pi, _ = _eval(D)
        if pi is not None and pi > best_pi:
            best_pi, best_D = pi, D

    return best_pi, best_D


# -----------------------------------------------------------------------------
# CONSTRUIR OPERADORES
# -----------------------------------------------------------------------------
results = {}

if not RERUN:
    print("Construyendo operadores...")
    ops_full = build_2qd(Ncut_full)
    ops_trunc = {n: build_2qd(n) for n in n_bundle_list}   # Fock(n) por orden

    # -------------------------------------------------------------------------
    # BARRIDO
    # -------------------------------------------------------------------------
    for n_b in n_bundle_list:
        D_est = resonance_2qd(n_b)
        print(f"\n{'='*60}")
        print(f"  n={n_b}  Δ_est={D_est:.4f}  ({len(gphi_arr)} puntos)")
        print(f"{'='*60}")

        pi_arr = np.full(len(gphi_arr), np.nan)
        t0 = time.time()

        for i, gphi in enumerate(gphi_arr):
            pi_arr[i], _ = compute_purity(ops_full, ops_trunc[n_b],
                                          gphi, D_est, n_b)
            if (i+1) % 15 == 0 or i == len(gphi_arr)-1:
                el = time.time() - t0
                eta = el/(i+1) * (len(gphi_arr)-i-1)
                print(f"  [{i+1:3d}/{len(gphi_arr)}] gphi={gphi:.2e} "
                      f"Pi_{n_b}={pi_arr[i]:.4f}  ({el:.0f}s ETA~{eta:.0f}s)")

        results[n_b] = pi_arr

    # -------------------------------------------------------------------------
    # EXPORTAR
    # -------------------------------------------------------------------------
    np.savez('results/data/dephasing_vs_purity_data.npz',
             gphi_arr=gphi_arr,
             lam=lam, kappa=kappa, gamma=gamma,
             gamma_phi_ref=gamma_phi_ref, J=J, Omega=Omega,
             **{f'pi_n{n}': results[n] for n in n_bundle_list})
    print("\n✓ Datos: results/data/dephasing_vs_purity_data.npz")

else:
    data = np.load('results/data/dephasing_vs_purity_data.npz')
    gphi_arr = data['gphi_arr']
    lam = data['lam'].item()
    kappa = data['kappa'].item()
    gamma = data['gamma'].item()
    gamma_phi_ref = data['gamma_phi_ref'].item()
    J = data['J'].item()
    Omega = data['Omega'].item()
    results = {n: data[f'pi_n{n}'] for n in n_bundle_list}


# -----------------------------------------------------------------------------
# FIGURA — un solo panel, estilo tesis
# -----------------------------------------------------------------------------
colors = {2: '#1f77b4', 3: '#d62728'}   # azul / rojo
labels = {2: r'$\Pi_2$', 3: r'$\Pi_3$'}

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

for n_b in n_bundle_list:
    ax.semilogx(gphi_arr, results[n_b],
                color=colors[n_b], lw=0.9, label=labels[n_b])

# Línea de referencia γ_φ usado en mapas de pureza
ax.axvline(gamma_phi_ref, ls='--', color='gray', lw=0.9,
           label=r'$\gamma_\varphi^{\,\mathrm{ref}}$')

ax.set_xlabel(r'$\gamma_\varphi/\omega_b$', fontsize=12)
ax.set_ylabel(r'$\Pi_n$', fontsize=12)
ax.set_xlim(1e-5, gphi_arr[-1])
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0, 0.5, 1.0])
ax.set_yticklabels([r'$0$', r'$0.5$', r'$1$'])
ax.tick_params(labelsize=12)
ax.set_facecolor('white')
ax.legend(fontsize=10, frameon=False)

fig.subplots_adjust(left=0.11, right=0.98, top=0.95, bottom=0.22)
fig.savefig("results/oficial/deph_purity.pdf", bbox_inches='tight')
fig.savefig("results/oficial/pgf/deph_purity.pgf")
print("Imágenes guardadas")