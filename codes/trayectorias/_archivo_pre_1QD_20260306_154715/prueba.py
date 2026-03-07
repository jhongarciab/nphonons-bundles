#!/usr/bin/env python3
"""
PUREZA Π_N — 2QD + Förster — Monte Carlo (definición exacta de Bin)

Π_N = P̄_N / Σ_{m=1}^{N} P̄_m
P̄_m = promedio de la población fonónica |m⟩ muestreada en tiempos
aleatorios sobre múltiples trayectorias MC.

IMPORTANTE: QuTiP 5 requiere 'keep_runs_results':True para acceder
a los estados de cada trayectoria individual (runs_states).

Parámetros: Ω=0.08, γ=γ_φ=0.0004, J=0.5, Ncut=8
Ejes: λ∈[0.04,0.20], κ∈[10⁻⁵,10⁰]
Paneles: (a) Π_3, (b) Π_4
"""
import numpy as np
import qutip as qt
import matplotlib

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from math import factorial, sqrt, pi
import time

print(f"QuTiP v{qt.__version__}  |  NumPy v{np.__version__}")

# ── PARÁMETROS FÍSICOS ──
omega_b = 1.0
Omega = 0.08
gamma = 0.0004
gamma_phi = 0.0004
J = 0.5
Ncut = 8

# ── RESOLUCIÓN ── (ajustar según tiempo)
n_lam = 5  # TEST:10  MEDIA:15  PROD:25
n_kappa = 5  # TEST:10  MEDIA:15  PROD:25
ntraj = 5  # TEST:10  MEDIA:20  PROD:40
n_delta = 12  # puntos búsqueda Δ

# t_max absoluto: si Ω_eff es tan pequeña que necesita t > T_MAX, skip
T_MAX = 2e6  # cap de tiempo de simulación
# mínimo de periodos Rabi para muestrar:
MIN_RABI_PERIODS = 5

lambda_arr = np.linspace(0.04, 0.20, n_lam)
kappa_arr = np.logspace(-5, 0, n_kappa)

# ── OPERADORES ──
b = qt.destroy(Ncut);
nb = b.dag() * b;
I_b = qt.qeye(Ncut)
I_q = qt.qeye(2);
sm = qt.sigmam()
b_sys = qt.tensor(I_q, I_q, b);
nb_sys = qt.tensor(I_q, I_q, nb)
sm1 = qt.tensor(sm, I_q, I_b);
sp1 = sm1.dag()
sm2 = qt.tensor(I_q, sm, I_b);
sp2 = sm2.dag()
proj_e1 = sp1 * sm1;
proj_e2 = sp2 * sm2;
proj_sum = proj_e1 + proj_e2
H_phon = omega_b * nb_sys;
H_drive = Omega * (sm1 + sp1 + sm2 + sp2)
H_Forster = J * (sp1 * sm2 + sp2 * sm1)
psi0 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0), qt.basis(Ncut, 0))
phonon_proj = [qt.tensor(I_q, I_q, qt.fock_dm(Ncut, m)) for m in range(Ncut)]


# ── FUNCIONES ──
def solve_ss(H, c_ops):
    for m in ('direct', 'eigen', 'svd'):
        try:
            rho = qt.steadystate(H, c_ops, method=m, use_rcm=True)
            if abs(rho.tr() - 1) < 1e-4: return rho
        except:
            pass
    return None


def build_H(Delta, lam):
    return (H_phon + Delta * proj_sum +
            lam * proj_sum * (b_sys + b_sys.dag()) + H_drive + H_Forster)


def build_c_ops(kappa):
    return [np.sqrt(kappa) * b_sys, np.sqrt(gamma) * sm1, np.sqrt(gamma) * sm2,
            np.sqrt(gamma_phi) * proj_e1, np.sqrt(gamma_phi) * proj_e2]


def find_optimal_delta(n_b, lam, kappa):
    D_est = -n_b * omega_b - J + lam ** 2 / omega_b
    D_scan = np.linspace(D_est - 0.3, D_est + 0.3, n_delta)
    c_ops = build_c_ops(kappa)
    op_n = b_sys.dag() ** n_b * b_sys ** n_b
    best = (-1.0, D_est)
    for D in D_scan:
        rho = solve_ss(build_H(D, lam), c_ops)
        if rho is None: continue
        val = np.real(qt.expect(op_n, rho))
        if val > best[0]: best = (val, D)
    return best[1]


def compute_purity_mc(n_b, lam, kappa):
    """Pureza exacta por MC con muestreo de poblaciones."""
    # Escala temporal
    Oeff = sqrt(2) * Omega * (lam / omega_b) ** n_b / sqrt(factorial(n_b))
    if Oeff < 1e-10:
        return np.nan  # demasiado débil, skip
    T_Rabi = pi / Oeff

    # Si necesitamos más de T_MAX para tener MIN_RABI_PERIODS, skip
    t_needed = (MIN_RABI_PERIODS + 2) * T_Rabi  # +2 de transitorio
    if t_needed > T_MAX:
        return np.nan

    # Encontrar Δ_opt
    delta = find_optimal_delta(n_b, lam, kappa)
    H = build_H(delta, lam)
    c_ops = build_c_ops(kappa)

    # Tiempos: 2 periodos transitorio + MIN_RABI_PERIODS de muestreo
    t_trans = 2 * T_Rabi
    t_end = t_trans + MIN_RABI_PERIODS * T_Rabi
    t_end = min(t_end, T_MAX)

    n_t_trans = 10
    n_t_sample = 50  # muestras por trayectoria
    tlist_trans = np.linspace(0, t_trans, n_t_trans)
    tlist_samp = np.linspace(t_trans, t_end, n_t_sample)
    tlist = np.unique(np.concatenate([tlist_trans, tlist_samp]))

    # MC
    try:
        result = qt.mcsolve(H, psi0, tlist, c_ops, ntraj=ntraj,
                            options={'store_states': True,
                                     'keep_runs_results': True,
                                     'progress_bar': False})
    except Exception as e:
        return np.nan

    # Muestrear poblaciones fonónicas (solo en zona de muestreo)
    idx_start = n_t_trans
    pop = np.zeros(Ncut)
    ns = 0
    for traj in result.runs_states:
        for ti in range(idx_start, len(tlist)):
            if ti >= len(traj): break
            psi_t = traj[ti]
            for m in range(min(n_b + 2, Ncut)):
                pop[m] += np.real(qt.expect(phonon_proj[m], psi_t))
            ns += 1

    if ns == 0:
        return np.nan
    P_bar = pop / ns

    # Π_N = P̄_N / Σ_{m=1}^{N} P̄_m
    den = np.sum(P_bar[1:n_b + 1])
    if den < 1e-18:
        return 0.0
    return float(np.clip(P_bar[n_b] / den, 0.0, 1.0))


# ── BARRIDO ──
def run_sweep(n_b, label):
    purity = np.full((n_kappa, n_lam), np.nan)
    t0 = time.time()
    n_skip = 0
    for j, lam in enumerate(lambda_arr):
        for i, kappa in enumerate(kappa_arr):
            purity[i, j] = compute_purity_mc(n_b, lam, kappa)
            if np.isnan(purity[i, j]):
                n_skip += 1
        el = time.time() - t0
        valid = purity[:, j];
        v = valid[~np.isnan(valid)]
        mx = np.max(v) if len(v) > 0 else 0
        print(f"  [{label}] λ={lam:.3f} {j + 1}/{n_lam} "
              f"Π_max={mx:.3f} skip={n_skip} "
              f"{el:.0f}s ETA≈{el / (j + 1) * (n_lam - j - 1):.0f}s", flush=True)
    print(f"  [{label}] ✓ {time.time() - t0:.0f}s ({n_skip} skipped)\n")
    return purity


# ── EJECUCIÓN ──
print("=" * 70)
print("PUREZA Π_N — 2QD + Förster — Monte Carlo exacto")
print("=" * 70)
print(f"  Ω={Omega} γ={gamma} γ_φ={gamma_phi} J={J} Ncut={Ncut}")
print(f"  ntraj={ntraj} n_delta={n_delta} T_MAX={T_MAX:.0e}")
print(f"  {n_lam}×{n_kappa}, λ∈[{lambda_arr[0]:.2f},{lambda_arr[-1]:.2f}]"
      f" κ∈[{kappa_arr[0]:.0e},{kappa_arr[-1]:.0e}]\n")

purity_3 = run_sweep(3, "Π₃")
purity_4 = run_sweep(4, "Π₄")

np.save('purity_mc_2qd_Pi3.npy', purity_3)
np.save('purity_mc_2qd_Pi4.npy', purity_4)
np.save('purity_mc_2qd_lambda.npy', lambda_arr)
np.save('purity_mc_2qd_kappa.npy', kappa_arr)
print("✓ Datos guardados\n")

# ── FIGURA ──
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), sharey=True)
fig.subplots_adjust(wspace=0.06, right=0.88)
norm = Normalize(vmin=0, vmax=1);
cmap = plt.cm.YlGnBu_r
for ax, pur, n_b, panel in [(axes[0], purity_3, 3, '(a)'), (axes[1], purity_4, 4, '(b)')]:
    p = np.clip(np.nan_to_num(pur, nan=0), 0, 1)
    im = ax.pcolormesh(lambda_arr, kappa_arr, p, norm=norm, cmap=cmap, shading='auto')
    try:
        lvls = [0.95, 0.97, 0.99] if n_b == 3 else [0.80, 0.90, 0.95]
        cs = ax.contour(lambda_arr, kappa_arr, p, levels=lvls,
                        colors='black', linewidths=1, linestyles='dashed')
        ax.clabel(cs, inline=True, fontsize=9, fmt='%.2f')
    except:
        pass
    for la in [0.08, 0.14]:
        ax.annotate('', xy=(la, kappa_arr[-1] * 0.85), xytext=(la, kappa_arr[-1] * 1.6),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2.5),
                    annotation_clip=False)
    ax.annotate('', xy=(lambda_arr[0] * 1.05, kappa_arr[0]),
                xytext=(lambda_arr[0] - 0.01, kappa_arr[0]),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                annotation_clip=False)
    ax.set_yscale('log');
    ax.set_xlim(lambda_arr[0], lambda_arr[-1])
    ax.set_ylim(kappa_arr[0], kappa_arr[-1])
    ax.set_xlabel(r'$\lambda/\omega_b$', fontsize=13);
    ax.tick_params(labelsize=11)
    ax.text(0.05, 0.95, panel, transform=ax.transAxes, fontsize=14, fontweight='bold',
            va='top', color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
    ax.set_title(rf'$\Pi_{n_b}$', fontsize=13)
axes[0].set_ylabel(r'$\kappa/\omega_b$', fontsize=13)
cb = fig.add_axes([0.90, 0.15, 0.02, 0.70])
cbar = fig.colorbar(im, cax=cb);
cbar.set_ticks([0, 0.5, 1])
cbar.set_label(r'$\Pi_N$', fontsize=13);
cbar.ax.tick_params(labelsize=11)
plt.show()