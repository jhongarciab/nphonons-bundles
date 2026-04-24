#!/usr/bin/env python3
"""
Trayectoria cuántica — Molécula excitónica (2QD + Förster)
===========================================================

Extensión del código de trayectorias de 1QD al sistema 2QD.
Muestra la cascada de emisión de 2-fonon bundles con los estados
colectivos de Dicke: |vv⟩, |Ψ₊⟩ = (|cv⟩+|vc⟩)/√2, |Ψ₋⟩, |cc⟩.

Espacio: QD₁ ⊗ QD₂ ⊗ Fock(Nph)
  - Base TLS QuTiP: |0⟩ = |c⟩ (excitado), |1⟩ = |v⟩ (valencia)
  - sigmam = |v⟩⟨c| = |1⟩⟨0|

Hamiltoniano (marco rotante):
  H = ω_b b†b + Δ(σ₁†σ₁ + σ₂†σ₂) + λ(σ₁†σ₁ + σ₂†σ₂)(b+b†)
      + Ω(σ₁+σ₁†+σ₂+σ₂†) + J(σ₁†σ₂+σ₂†σ₁)

Resonancia de Stokes 2QD (Régimen II): Δ_n = λ²/ω_b - n·ω_b - J

Autor: Jhon S. García B. — Tesis UQ 2025
"""

import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qutip import *

# -----------------------------------------------------------------------------
# Configuración general
# -----------------------------------------------------------------------------
RERUN = True # False para recalcular, True para cargar datos guardados

# -----------------------------------------------------------------------------
# Estilo global de figura
# -----------------------------------------------------------------------------
rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.size": 8,
})

# ————————————————————————————————————————————————————————————————
# PARÁMETROS
# ————————————————————————————————————————————————————————————————
omegab    = 1.0
n         = 2             # bundle de 2 fonones

lam       = 0.1 * omegab
Om        = 0.2 * omegab
kappa     = 0.0005
gamma     = 0.0002
gamma_phi = 0.0004
J         = 0.5 * omegab

# Resonancia de Stokes 2QD — Régimen II
Delta = lam**2 / omegab - n * omegab - J

Nph   = 12
ntraj = 25
x_max = 10000.0
Nt    = 10001

tlist = np.linspace(0.0, x_max / omegab, Nt)
x     = omegab * tlist

print(f"Δ/ω_b = {Delta/omegab:.5f}")
print(f"Nph = {Nph}, ntraj = {ntraj}, t_max = {x_max}")
print(f"dim = 2 × 2 × {Nph} = {4*Nph}")

# ————————————————————————————————————————————————————————————————
# BASE TLS
# ————————————————————————————————————————————————————————————————
ket_c  = basis(2, 0)           # |c⟩ = excitado
ket_v  = basis(2, 1)           # |v⟩ = valencia
Pc     = ket2dm(ket_c)         # |c⟩⟨c| = σ†σ
sm_tls = ket_v * ket_c.dag()   # σ⁻ = |v⟩⟨c|

# ————————————————————————————————————————————————————————————————
# OPERADORES DEL SISTEMA QD₁ ⊗ QD₂ ⊗ Fock
# ————————————————————————————————————————————————————————————————
b   = destroy(Nph)
Ib  = qeye(Nph)
Iq  = qeye(2)
nb  = b.dag() * b

b_sys  = tensor(Iq, Iq, b)
nb_sys = tensor(Iq, Iq, nb)

sm1 = tensor(sm_tls, Iq, Ib);  sp1 = sm1.dag()
sm2 = tensor(Iq, sm_tls, Ib);  sp2 = sm2.dag()
pe1 = tensor(Pc, Iq, Ib)       # σ₁†σ₁
pe2 = tensor(Iq, Pc, Ib)       # σ₂†σ₂

I_sys = tensor(Iq, Iq, Ib)

# ————————————————————————————————————————————————————————————————
# PROYECTORES PARA POBLACIONES
# ————————————————————————————————————————————————————————————————
psi_vv    = tensor(ket_v, ket_v)
psi_plus  = (tensor(ket_c, ket_v) + tensor(ket_v, ket_c)).unit()  # |Ψ₊⟩
psi_minus = (tensor(ket_c, ket_v) - tensor(ket_v, ket_c)).unit()  # |Ψ₋⟩

P_minus = tensor(ket2dm(psi_minus), Ib)

P0_vv   = tensor(ket2dm(psi_vv),   ket2dm(basis(Nph, 0)))
P0_plus = tensor(ket2dm(psi_plus), ket2dm(basis(Nph, 0)))
P1_vv   = tensor(ket2dm(psi_vv),   ket2dm(basis(Nph, 1)))
P1_plus = tensor(ket2dm(psi_plus), ket2dm(basis(Nph, 1)))
P2_vv   = tensor(ket2dm(psi_vv),   ket2dm(basis(Nph, 2)))
P2_plus = tensor(ket2dm(psi_plus), ket2dm(basis(Nph, 2)))

# ————————————————————————————————————————————————————————————————
# HAMILTONIANO
# ————————————————————————————————————————————————————————————————
ne_total = pe1 + pe2

H = (omegab * nb_sys
     + Delta  * ne_total
     + lam    * ne_total * (b_sys + b_sys.dag())
     + Om     * (sm1 + sp1 + sm2 + sp2)
     + J      * (sp1 * sm2 + sp2 * sm1))

# ————————————————————————————————————————————————————————————————
# OPERADORES DE COLAPSO
# ————————————————————————————————————————————————————————————————
c_ops = [
    np.sqrt(kappa)     * b_sys,   # canal 0: emisión fonón
    np.sqrt(gamma)     * sm1,     # canal 1: decaimiento QD₁
    np.sqrt(gamma)     * sm2,     # canal 2: decaimiento QD₂
    np.sqrt(gamma_phi) * pe1,     # canal 3: dephasing QD₁
    np.sqrt(gamma_phi) * pe2,     # canal 4: dephasing QD₂
]
idx_cav = 0

# ————————————————————————————————————————————————————————————————
# ESTADO INICIAL: |vv, 0⟩
# ————————————————————————————————————————————————————————————————
psi0 = tensor(ket_v, ket_v, basis(Nph, 0))

# ————————————————————————————————————————————————————————————————
# e_ops
# ————————————————————————————————————————————————————————————————
e_ops = [
    P0_vv,    # 0
    P0_plus,  # 1
    P1_vv,    # 2
    P1_plus,  # 3
    P2_vv,    # 4
    P2_plus,  # 5
    P_minus,  # 6
    nb_sys,   # 7
]
e_labels = [
    'P(vv,0)', 'P(Ψ+,0)', 'P(vv,1)', 'P(Ψ+,1)',
    'P(vv,2)', 'P(Ψ+,2)', 'P(Ψ-)', '<n_b>',
]

# ————————————————————————————————————————————————————————————————
# MCSOLVE o CARGA DE NPZ
# ————————————————————————————————————————————————————————————————
if not RERUN:
    seed  = 324680751
    rng   = np.random.default_rng(seed)
    seeds = [int(rng.integers(1e9)) for _ in range(ntraj)]

    opts = {
        "keep_runs_results": True,
        "store_states": False,
        "progress_bar": "text",
        "nsteps": 200000,
        "improved_sampling": True,
    }

    print(f"\nLanzando mcsolve: {ntraj} trayectorias...")
    res = mcsolve(H, psi0, tlist, c_ops, e_ops=e_ops, ntraj=ntraj,
                  options=opts, seeds=seeds)

    # Seleccionar trayectoria más activa
    nj = np.zeros(ntraj, dtype=int)
    for k in range(ntraj):
        ct = np.array(res.col_times[k] if res.col_times[k] is not None else [])
        cw = np.array(res.col_which[k] if res.col_which[k] is not None else [])
        nj[k] = np.count_nonzero(cw == idx_cav)

    k_show = int(np.argmax(nj))
    print(f"\nk_show={k_show}, cavity_jumps={int(nj[k_show])}, "
          f"mean_jumps={np.mean(nj):.1f}, max_jumps={np.max(nj)}")

    data = {label: res.expect[i][k_show] for i, label in enumerate(e_labels)}

    ct     = np.array(res.col_times[k_show] if res.col_times[k_show] is not None else [])
    cw     = np.array(res.col_which[k_show] if res.col_which[k_show] is not None else [])
    clicks = omegab * ct[cw == idx_cav]

    export = {'time': x, 'clicks': clicks, 'k_show': k_show,
              'cavity_jumps': int(nj[k_show]), 'mean_jumps': float(np.mean(nj))}
    export.update({label: data[label] for label in e_labels})
    np.savez('results/data/trayectoria_2qd_data.npz', **export)
    print("✓ Datos exportados: trayectoria_2qd_data.npz")

else:
    d      = np.load('results/data/trayectoria_2qd_data.npz', allow_pickle=True)
    clicks = d['clicks']
    data   = {label: d[label] for label in e_labels}
    x      = d['time']
    print(f"✓ Datos cargados: cavity_jumps={d['cavity_jumps']}, "
          f"mean_jumps={d['mean_jumps']:.1f}")

# ————————————————————————————————————————————————————————————————
# RESUMEN NUMÉRICO
# ————————————————————————————————————————————————————————————————
print(f"\n{'='*65}")
print(f"  Clicks fonónicos: {len(clicks)}")
print(f"  ⟨n̂_b⟩ promedio:   {np.mean(data['<n_b>']):.4e}")
print(f"  P(vv,0) promedio:  {np.mean(data['P(vv,0)']):.4f}")
print(f"  P(Ψ+,0) promedio:  {np.mean(data['P(Ψ+,0)']):.4e}")
print(f"  P(Ψ+,2) promedio:  {np.mean(data['P(Ψ+,2)']):.4e}")
print(f"  P(Ψ-)  promedio:   {np.mean(data['P(Ψ-)']):.4e}")

if len(clicks) > 1:
    dt = np.diff(clicks)
    print(f"\n  Inter-click times:")
    print(f"    mean   = {np.mean(dt):.1f} ω_b⁻¹")
    print(f"    min    = {np.min(dt):.1f} ω_b⁻¹")
    print(f"    max    = {np.max(dt):.1f} ω_b⁻¹")
    print(f"    median = {np.median(dt):.1f} ω_b⁻¹")

    bundle_window = 2.0 / kappa
    bundles, cur = [], [clicks[0]]
    for c in clicks[1:]:
        if c - cur[-1] < bundle_window:
            cur.append(c)
        else:
            bundles.append(cur); cur = [c]
    bundles.append(cur)
    sizes = [len(b) for b in bundles]
    print(f"\n  Bundles (ventana {bundle_window:.0f} ω_b⁻¹): {len(bundles)} total")
    for s in sorted(set(sizes)):
        print(f"    {s}-fonón: {sizes.count(s)} ({100*sizes.count(s)/len(bundles):.1f}%)")
print(f"{'='*65}")

# ————————————————————————————————————————————————————————————————
# VISUALIZACIÓN
# ————————————————————————————————————————————————————————————————
t_start = 5500.0
t_end   = 8500.0

mask        = (x >= t_start) & (x <= t_end)
x_plot      = x[mask] - t_start
clicks_plot = clicks[(clicks >= t_start) & (clicks <= t_end)] - t_start

eps = 1e-12

fig, axes = plt.subplots(3, 1, figsize=(5.00, 3.10), sharex=True)

# Panel (a) — n_fock = 0
axes[0].plot(x_plot, data['P(vv,0)'][mask],  lw=0.9, color='black')
axes[0].plot(x_plot, data['P(Ψ+,0)'][mask],  lw=0.9, color='red')
axes[0].set_ylim(0, 1)
axes[0].set_yticks([0, 0.5, 1])
axes[0].set_yticklabels([r'$0$', r'$0.5$', r'$1$'])
axes[0].tick_params(labelsize=8)
axes[0].set_facecolor('white')
axes[0].grid(False)
axes[0].text(400, 0.78, r'$P_{0vv}$', color='black', fontsize=8)
axes[0].text(400, 0.12, r'$P_{0\Psi_+}$', color='red', fontsize=8)

# Panel (b) — n_fock = 1
axes[1].semilogy(x_plot, data['P(Ψ+,1)'][mask] + eps, lw=0.9, color='#c4942a')
axes[1].semilogy(x_plot, data['P(vv,1)'][mask]  + eps, lw=0.9, color='black')
axes[1].set_ylim(1e-8, 1e1)
axes[1].set_yticks([1e-6, 1e-3, 1e0])
axes[1].set_yticklabels([r'$10^{-6}$', r'$10^{-3}$', r'$10^{0}$'])
axes[1].tick_params(labelsize=8)
axes[1].set_facecolor('white')
axes[1].set_ylabel("Poblaciones", fontsize=9, labelpad=8)
axes[1].grid(False)
axes[1].text(400, 2e-1, r'$P_{1\Psi_+}$', color='#c4942a', fontsize=8)
axes[1].text(400, 0.4e-7, r'$P_{1vv}$', color='black', fontsize=8)

# Panel (c) — n_fock = 2 + clicks
axes[2].semilogy(x_plot, data['P(Ψ+,2)'][mask] + eps, lw=0.9, color='#1f77b4')
axes[2].semilogy(x_plot, data['P(vv,2)'][mask]  + eps, lw=0.9, color='black')
axes[2].set_ylim(1e-11, 1e1)
axes[2].set_yticks([1e-10, 1e-5, 1e0])
axes[2].set_yticklabels([r'$10^{-10}$', r'$10^{-5}$', r'$10^{0}$'])
axes[2].set_xlim(0, 3000)
axes[2].set_xticks([0, 1000, 2000, 3000])
axes[2].tick_params(labelsize=8)
axes[2].set_facecolor('white')
axes[-1].set_xlabel(r'$\omega_b\,t$', fontsize=9)
axes[2].grid(False)
axes[2].text(380, 6e-2, r'$P_{2\Psi_+}$', color='#1f77b4', fontsize=8)
axes[2].text(380, 2e-7, r'$P_{2vv}$', color='black', fontsize=8)

# -----------------------------------------------------------------------------
# Etiquetas de panel: (a), (b), (c)
# -----------------------------------------------------------------------------
label_positions = {
    0: (0.004, 0.92),
    1: (0.004, 0.92),
    2: (0.004, 0.95),
}

for idx, ax in enumerate(axes):
    ax.text(
        label_positions[idx][0],
        label_positions[idx][1],
        f'$({chr(96 + idx + 1)})$',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=9,
    )

plt.tight_layout()
fig.subplots_adjust(hspace=0.16)
plt.savefig("results/oficial/trayectoria_2qd_paper.pdf", bbox_inches="tight")
plt.savefig("results/oficial/pgf/trayectoria_2qd_paper.pgf")
plt.close()
print("Imágenes guardadas con éxito.")