#!/usr/bin/env python3
"""
=============================================================================
PUREZA Π_N para molécula excitónica (2 QDs + Förster)

Réplica EXACTA de la lógica del código exitoso de 1QD, adaptada a 2QDs.

HAMILTONIANO (idéntico a forster_2qds.py del proyecto):
   H = ωb b†b + Δ(σ₁†σ₁ + σ₂†σ₂) + λ(σ₁†σ₁ + σ₂†σ₂)(b + b†)
     + Ω(σ₁ + σ₁† + σ₂ + σ₂†) + J(σ₁†σ₂ + σ₂†σ₁)

LINDBLAD (idéntico a forster_2qds.py):
   κ D[b] + γ D[σ₁] + γ D[σ₂] + γ_φ D[σ₁†σ₁] + γ_φ D[σ₂†σ₂]

CORRESPONDENCIA (derivaciones.tex):
   Δ_n(λ) = -nωb - J + λ²/ωb  (resonancia Stokes régimen II)
   Ω̃_eff^(n) = √2·Ω·(λ/ωb)^n/√(n!)  (3rabis.py)

MÉTODO: Para CADA par (λ,κ) busca Δ_opt que maximiza ⟨(b†)^N b^N⟩_ss
        Luego calcula T_N = N·⟨(b†)^N b^N⟩/((N-1)!·⟨b†b⟩)

PARÁMETROS:  Ω=0.08, γ=γ_φ=0.0004, J=0.5, Ncut=8
EJES:        λ ∈ [0.04, 0.20],  κ ∈ [10⁻⁵, 10⁰]
PANELES:     (a) Π_3,  (b) Π_4

RESOLUCIÓN:  TEST=20×20  |  MEDIA=30×30  |  PROD=50×50
=============================================================================
"""
import numpy as np
import qutip as qt
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from math import factorial
import time

print(f"QuTiP v{qt.__version__}  |  NumPy v{np.__version__}")

# =============================================================================
# PARÁMETROS
# =============================================================================
omega_b   = 1.0
Omega     = 0.08
gamma     = 0.0004
gamma_phi = 0.0004
J         = 0.5
Ncut      = 8

n_lam   = 20     # TEST:20 | MEDIA:30 | PROD:50
n_kappa = 20     # TEST:20 | MEDIA:30 | PROD:50
n_delta = 15     # puntos en barrido de Δ

lambda_arr = np.linspace(0.04, 0.20, n_lam)
kappa_arr  = np.logspace(-5, 0, n_kappa)

# =============================================================================
# OPERADORES — QD1 ⊗ QD2 ⊗ Fock(Ncut)
# =============================================================================
b = qt.destroy(Ncut); nb = b.dag()*b; I_b = qt.qeye(Ncut)
I_q = qt.qeye(2); sm = qt.sigmam()

b_sys   = qt.tensor(I_q, I_q, b)
nb_sys  = qt.tensor(I_q, I_q, nb)
sm1     = qt.tensor(sm, I_q, I_b); sp1 = sm1.dag()
sm2     = qt.tensor(I_q, sm, I_b); sp2 = sm2.dag()
proj_e1 = sp1*sm1; proj_e2 = sp2*sm2
proj_sum = proj_e1 + proj_e2

H_phon    = omega_b * nb_sys
H_drive   = Omega * (sm1 + sp1 + sm2 + sp2)
H_Forster = J * (sp1*sm2 + sp2*sm1)

bdagb_ops = {3: b_sys.dag()**3 * b_sys**3,
             4: b_sys.dag()**4 * b_sys**4}

# =============================================================================
# FUNCIONES
# =============================================================================
def validate_rho(rho, tol=1e-6):
    if abs(rho.tr()-1)>1e-4: return False
    if not rho.isherm: return False
    if np.min(np.real(rho.eigenstates()[0]))<-tol: return False
    return True

def solve_ss(H, c_ops):
    for m in ('direct','eigen','svd'):
        try:
            rho = qt.steadystate(H, c_ops, method=m, use_rcm=True)
            if validate_rho(rho): return rho
        except: pass
    return None

def build_H(Delta, lam):
    return H_phon + Delta*proj_sum + lam*proj_sum*(b_sys+b_sys.dag()) + H_drive + H_Forster

def build_c_ops(kappa):
    return [np.sqrt(kappa)*b_sys, np.sqrt(gamma)*sm1, np.sqrt(gamma)*sm2,
            np.sqrt(gamma_phi)*proj_e1, np.sqrt(gamma_phi)*proj_e2]

def T_N(n, rho):
    nbar = np.real(qt.expect(nb_sys, rho))
    if nbar < 1e-18: return 0.0
    exp_n = np.real(qt.expect(bdagb_ops[n], rho))
    return float(np.clip(n*exp_n/(factorial(n-1)*nbar), 0, 1))

def find_optimal_delta(n_bundle, lam, kappa):
    Delta_est = -n_bundle*omega_b - J + lam**2/omega_b
    Delta_scan = np.linspace(Delta_est-0.3, Delta_est+0.3, n_delta)
    c_ops = build_c_ops(kappa)
    op_n = bdagb_ops[n_bundle]
    best = (-1.0, Delta_est)
    for D in Delta_scan:
        rho = solve_ss(build_H(D, lam), c_ops)
        if rho is None: continue
        val = np.real(qt.expect(op_n, rho))
        if val > best[0]: best = (val, D)
    return best[1]

def run_sweep(n_bundle, label):
    purity = np.full((n_kappa, n_lam), np.nan)
    t0 = time.time()
    for j, lam in enumerate(lambda_arr):
        for i, kappa in enumerate(kappa_arr):
            d = find_optimal_delta(n_bundle, lam, kappa)
            rho = solve_ss(build_H(d, lam), build_c_ops(kappa))
            if rho is not None: purity[i,j] = T_N(n_bundle, rho)
        el = time.time()-t0
        print(f"  [{label}] λ={lam:.3f} {j+1}/{n_lam} "
              f"({100*(j+1)/n_lam:.0f}%) {el:.0f}s ETA≈{el/(j+1)*(n_lam-j-1):.0f}s",
              flush=True)
    print(f"  [{label}] ✓ {time.time()-t0:.0f}s\n")
    return purity

# =============================================================================
# EJECUCIÓN
# =============================================================================
print("="*65)
print("Pureza Π_N — 2QD + Förster")
print("="*65)
print(f"  Ω={Omega} γ={gamma} γ_φ={gamma_phi} J={J} Ncut={Ncut}")
print(f"  {n_lam}×{n_kappa} n_delta={n_delta}")
print(f"  λ∈[{lambda_arr[0]:.2f},{lambda_arr[-1]:.2f}] κ∈[{kappa_arr[0]:.0e},{kappa_arr[-1]:.0e}]\n")

purity_3 = run_sweep(3, "Π₃")
purity_4 = run_sweep(4, "Π₄")

np.save('purity_2qd_Pi3.npy', purity_3)
np.save('purity_2qd_Pi4.npy', purity_4)
np.save('purity_2qd_lambda.npy', lambda_arr)
np.save('purity_2qd_kappa.npy', kappa_arr)
print("✓ Datos guardados\n")

# =============================================================================
# FIGURA
# =============================================================================
fig, axes = plt.subplots(1,2, figsize=(9.5,4.8), sharey=True)
fig.subplots_adjust(wspace=0.06, right=0.88)
norm = Normalize(vmin=0, vmax=1)
cmap = plt.cm.YlGnBu_r

for ax, pur, n_b, panel in [(axes[0],purity_3,3,'(a)'), (axes[1],purity_4,4,'(b)')]:
    p = np.clip(np.nan_to_num(pur,nan=0), 0, 1)
    im = ax.pcolormesh(lambda_arr, kappa_arr, p, norm=norm, cmap=cmap, shading='auto')
    try:
        lvls = [0.95,0.97,0.99] if n_b==3 else [0.80,0.90,0.95]
        cs = ax.contour(lambda_arr, kappa_arr, p, levels=lvls,
                        colors='black', linewidths=1, linestyles='dashed')
        ax.clabel(cs, inline=True, fontsize=9, fmt='%.2f')
    except: pass
    for la in [0.08,0.14]:
        ax.annotate('',xy=(la,kappa_arr[-1]*0.85),xytext=(la,kappa_arr[-1]*1.6),
                    arrowprops=dict(arrowstyle='->',color='white',lw=2.5),annotation_clip=False)
    ax.annotate('',xy=(lambda_arr[0]*1.05,kappa_arr[0]),
                xytext=(lambda_arr[0]-0.01,kappa_arr[0]),
                arrowprops=dict(arrowstyle='->',color='black',lw=2),annotation_clip=False)
    ax.set_yscale('log'); ax.set_xlim(lambda_arr[0],lambda_arr[-1])
    ax.set_ylim(kappa_arr[0],kappa_arr[-1])
    ax.set_xlabel(r'$\lambda/\omega_b$',fontsize=13); ax.tick_params(labelsize=11)
    ax.text(0.05,0.95,panel,transform=ax.transAxes,fontsize=14,fontweight='bold',
            va='top',color='white',bbox=dict(boxstyle='round,pad=0.2',facecolor='black',alpha=0.5))
    ax.set_title(rf'$\Pi_{n_b}$',fontsize=13)

axes[0].set_ylabel(r'$\kappa/\omega_b$',fontsize=13)
cb = fig.add_axes([0.90,0.15,0.02,0.70])
cbar = fig.colorbar(im,cax=cb); cbar.set_ticks([0,0.5,1])
cbar.set_label(r'$\Pi_N$',fontsize=13); cbar.ax.tick_params(labelsize=11)
plt.show()