#!/usr/bin/env python3
"""
Trayectoria cuántica — Molécula excitónica (2QD + Förster)
===========================================================

Versión galería: computa ntraj trayectorias y muestra las 10 más
activas (más clicks fonónicos) en subplots individuales con su
seed y estadísticas, para seleccionar la mejor.

Autor: Jhon S. García B. — Tesis UQ 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# =============================================================================
# PARÁMETROS
# =============================================================================
omegab = 1.0
n = 2

lam       = 0.1 * omegab
Om        = 0.2 * omegab
kappa     = 0.0005
gamma     = 0.0002
gamma_phi = 0
J         = 0.5 * omegab

# Resonancia de Stokes 2QD (Régimen III + Lamb shift + Förster)
Delta = -2.429

Nph    = 12
ntraj  = 50          # Más trayectorias para mejor selección
n_show = 10          # Mostrar las 10 más activas
x_max  = 40000.0
Nt     = 40001

tlist = np.linspace(0.0, x_max / omegab, Nt)
x = omegab * tlist

print(f"Δ/ω_b = {Delta/omegab:.5f}")
print(f"Nph = {Nph}, ntraj = {ntraj}, t_max = {x_max}")
print(f"dim = 2 × 2 × {Nph} = {4*Nph}")

# =============================================================================
# BASE TLS Y OPERADORES
# =============================================================================
ket_c = basis(2, 0)
ket_v = basis(2, 1)
Pc = ket2dm(ket_c)
sm_tls = ket_v * ket_c.dag()
sp_tls = sm_tls.dag()

b  = destroy(Nph)
Ib = qeye(Nph)
Iq = qeye(2)
nb = b.dag() * b

b_sys   = tensor(Iq, Iq, b)
nb_sys  = tensor(Iq, Iq, nb)
sm1 = tensor(sm_tls, Iq, Ib); sp1 = sm1.dag()
sm2 = tensor(Iq, sm_tls, Ib); sp2 = sm2.dag()
pe1 = tensor(Pc, Iq, Ib)
pe2 = tensor(Iq, Pc, Ib)
ne_total = pe1 + pe2

# Estados colectivos
psi_vv   = tensor(ket_v, ket_v)
psi_plus = (tensor(ket_c, ket_v) + tensor(ket_v, ket_c)).unit()

# Proyectores
P0_vv   = tensor(ket2dm(psi_vv), ket2dm(basis(Nph, 0)))
P0_plus = tensor(ket2dm(psi_plus), ket2dm(basis(Nph, 0)))
P2_plus = tensor(ket2dm(psi_plus), ket2dm(basis(Nph, 2)))

# =============================================================================
# HAMILTONIANO Y COLAPSOS
# =============================================================================
H = (omegab * nb_sys
     + Delta * ne_total
     + lam * ne_total * (b_sys + b_sys.dag())
     + Om * (sm1 + sp1 + sm2 + sp2)
     + J * (sp1 * sm2 + sp2 * sm1))

c_ops = [
    np.sqrt(kappa)     * b_sys,
    np.sqrt(gamma)     * sm1,
    np.sqrt(gamma)     * sm2,
    np.sqrt(gamma_phi) * pe1,
    np.sqrt(gamma_phi) * pe2,
]
idx_cav = 0

psi0 = tensor(ket_v, ket_v, basis(Nph, 0))

e_ops = [P0_vv, P0_plus, P2_plus, nb_sys]
e_labels = ['P(vv,0)', 'P(Ψ+,0)', 'P(Ψ+,2)', '<n_b>']

# =============================================================================
# SEMILLAS
# =============================================================================
master_seed = 1238234
rng = np.random.default_rng(master_seed)
seeds = [int(rng.integers(1e9)) for _ in range(ntraj)]

# =============================================================================
# MCSOLVE
# =============================================================================
opts = {
    "keep_runs_results": True,
    "store_states": False,
    "progress_bar": "text",
    "nsteps": 200000,
    "improved_sampling": True,
}

print(f"\nLanzando mcsolve: {ntraj} trayectorias (master_seed={master_seed})...")
res = mcsolve(H, psi0, tlist, c_ops, e_ops=e_ops, ntraj=ntraj,
              options=opts, seeds=seeds)

# =============================================================================
# CLASIFICAR TRAYECTORIAS
# =============================================================================
traj_info = []
for k in range(ntraj):
    ct = np.array(res.col_times[k] if res.col_times[k] is not None else [])
    cw = np.array(res.col_which[k] if res.col_which[k] is not None else [])
    cav_jumps = np.count_nonzero(cw == idx_cav)
    clicks_k = omegab * ct[cw == idx_cav]

    # Clasificar bundles
    bundle_window = 2.0 / kappa
    bundles = []
    if len(clicks_k) > 0:
        current = [clicks_k[0]]
        for c in clicks_k[1:]:
            if c - current[-1] < bundle_window:
                current.append(c)
            else:
                bundles.append(len(current))
                current = [c]
        bundles.append(len(current))

    n_2ph = bundles.count(2)
    n_1ph = bundles.count(1)
    n_other = len(bundles) - n_2ph - n_1ph

    traj_info.append({
        'k': k,
        'seed': seeds[k],
        'cav_jumps': cav_jumps,
        'clicks': clicks_k,
        'n_bundles': len(bundles),
        'n_2ph': n_2ph,
        'n_1ph': n_1ph,
        'n_other': n_other,
        'bundle_sizes': bundles,
    })

# Ordenar por actividad (más clicks fonónicos)
traj_info.sort(key=lambda x: x['cav_jumps'], reverse=True)

# Resumen
print(f"\n{'='*75}")
print(f"  RESUMEN DE TRAYECTORIAS (top {n_show} de {ntraj})")
print(f"{'='*75}")
print(f"{'k':>4s}  {'seed':>12s}  {'jumps':>6s}  {'bundles':>8s}  "
      f"{'2-ph':>5s}  {'1-ph':>5s}  {'other':>5s}  {'pureza':>7s}")
print("-" * 75)
for info in traj_info[:n_show]:
    purity = info['n_2ph'] / info['n_bundles'] if info['n_bundles'] > 0 else 0
    print(f"  {info['k']:3d}  {info['seed']:12d}  {info['cav_jumps']:6d}  "
          f"{info['n_bundles']:8d}  {info['n_2ph']:5d}  {info['n_1ph']:5d}  "
          f"{info['n_other']:5d}  {purity:7.1%}")

# Estadísticas globales
all_jumps = [t['cav_jumps'] for t in traj_info]
print(f"\n  Total: mean_jumps={np.mean(all_jumps):.1f}, "
      f"max={np.max(all_jumps)}, min={np.min(all_jumps)}")

# =============================================================================
# EXPORTAR
# =============================================================================
np.savez('trayectorias_2qd_gallery.npz',
         master_seed=master_seed,
         seeds=np.array(seeds),
         traj_ranking=np.array([t['k'] for t in traj_info[:n_show]]),
         traj_jumps=np.array([t['cav_jumps'] for t in traj_info[:n_show]]))
print("✓ Datos exportados: trayectorias_2qd_gallery.npz")

# =============================================================================
# GALERÍA: top n_show trayectorias
# =============================================================================
fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.5 * n_show), sharex=True)
eps = 1e-12

for row, info in enumerate(traj_info[:n_show]):
    k = info['k']
    ax = axes[row]

    # Extraer datos de esta trayectoria
    P0vv  = res.expect[0][k]
    P0psi = res.expect[1][k]

    ax.plot(x, P0vv, lw=0.8, color='k', label=r'$P_{0,vv}$')
    ax.plot(x, P0psi, lw=0.8, color='r', label=r'$P_{0,\Psi_+}$')

    # Marcar clicks
    for xc in info['clicks']:
        ax.axvline(xc, lw=0.4, alpha=0.3, color='blue')

    ax.set_ylim(0, 1)
    ax.set_ylabel('Prob.', fontsize=9)

    purity = info['n_2ph'] / info['n_bundles'] if info['n_bundles'] > 0 else 0
    ax.text(0.01, 0.85,
            f"k={k}  seed={info['seed']}  "
            f"jumps={info['cav_jumps']}  "
            f"bundles={info['n_bundles']} "
            f"(2ph:{info['n_2ph']}, 1ph:{info['n_1ph']})  "
            f"pureza={purity:.0%}",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    if row == 0:
        ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel(r'$\omega_b\, t$', fontsize=12)
fig.suptitle(rf'Galería de trayectorias 2QD — top {n_show} de {ntraj} '
             rf'($\Delta/\omega_b = {Delta/omegab:.4f}$, master_seed={master_seed})',
             fontsize=11, y=1.002)
plt.tight_layout()
plt.savefig('trayectorias_2qd_gallery.pdf', bbox_inches='tight')
plt.savefig('trayectorias_2qd_gallery.png', dpi=150, bbox_inches='tight')
print("\n✓ Galería guardada: trayectorias_2qd_gallery.{pdf,png}")
plt.show()