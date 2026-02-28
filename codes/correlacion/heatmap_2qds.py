import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
import matplotlib.colors as mcolors

rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 13,
})

# =============================================================================
# PARÁMETROS FIJOS — 2QD, régimen I, barrido en lambda
# =============================================================================
omega_b  = 1.0
Omega_ob = 0.01
kappa_ob = 0.003
gamma_ob = 0.0004
gphi_ob  = 0.0004
J_ob     = 0.5

Ncut = 8

# =============================================================================
# GRILLA 2D
# =============================================================================
n_Delta  = 30
n_lambda = 10
Delta_arr  = np.linspace(0.0, -6.0, n_Delta)
lambda_arr = np.logspace(-2, 0, n_lambda)

g2_map = np.full((n_lambda, n_Delta), np.nan)

# =============================================================================
# OPERADORES — QD1 ⊗ QD2 ⊗ Fock
# =============================================================================
b    = qt.destroy(Ncut)
numb = b.dag() * b
I_b  = qt.qeye(Ncut)
sm   = qt.sigmam()
sp   = sm.dag()
I_q  = qt.qeye(2)

b_sys   = qt.tensor(I_q, I_q, b)
num_sys = qt.tensor(I_q, I_q, numb)
sm1     = qt.tensor(sm, I_q, I_b)
sp1     = sm1.dag()
sm2     = qt.tensor(I_q, sm, I_b)
sp2     = sm2.dag()
proj_e1 = sp1 * sm1
proj_e2 = sp2 * sm2
I_sys   = qt.tensor(I_q, I_q, I_b)

# =============================================================================
# OPERADOR FACTORIAL
# =============================================================================
def factorial_op(num_op, n, I_op):
    op = I_op
    for k in range(n):
        op = op * (num_op - k * I_op)
    return op

bdagb2 = factorial_op(num_sys, 2, I_sys)

# =============================================================================
# SOLVER ROBUSTO
# =============================================================================
def validate_rho(rho, tol=1e-8):
    if abs(rho.tr() - 1.0) > 1e-6:
        return False
    if not rho.isherm:
        return False
    if np.min(np.real(rho.eigenstates()[0])) < -tol:
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
# HAMILTONIANOS FIJOS
# =============================================================================
H_phonon  = omega_b * num_sys
H_drive   = Omega_ob * (sm1 + sp1 + sm2 + sp2)
H_Forster = J_ob * (sp1 * sm2 + sp2 * sm1)

# =============================================================================
# BARRIDO 2D
# =============================================================================
total = n_lambda * n_Delta
count = 0

for i, lam in enumerate(lambda_arr):
    c_ops = [
        np.sqrt(kappa_ob) * b_sys,
        np.sqrt(gamma_ob) * sm1,
        np.sqrt(gamma_ob) * sm2,
        np.sqrt(gphi_ob)  * proj_e1,
        np.sqrt(gphi_ob)  * proj_e2,
    ]

    H_eph = lam * (proj_e1 + proj_e2) * (b_sys + b_sys.dag())

    for j, Delta in enumerate(Delta_arr):
        H_det = Delta * (proj_e1 + proj_e2)
        H = H_phonon + H_eph + H_drive + H_Forster + H_det

        rho_ss = solve_ss(H, c_ops)
        count += 1

        if rho_ss is None:
            continue

        nbar = qt.expect(num_sys, rho_ss)
        if nbar > 1e-12:
            num = qt.expect(bdagb2, rho_ss)
            g2_map[i, j] = np.real(num) / nbar**2

        if count % 50 == 0:
            print(f"  {count}/{total}  λ={lam:.3f}  Δ={Delta:.2f}  "
                  f"g2={g2_map[i,j]:.2e}" if not np.isnan(g2_map[i,j])
                  else f"  {count}/{total}  λ={lam:.3f}  Δ={Delta:.2f}  nan")

print("\n✓ Barrido completado")

# =============================================================================
# HEATMAP — 2QD régimen I
# =============================================================================
n_res = [1, 2, 3, 4, 5, 6]

colors_list = [
    '#1a6b3c',
    '#6abf7b',
    '#ffffff',
    '#e8d5a3',
    '#c4942a',
]
cmap_bin = mcolors.LinearSegmentedColormap.from_list('bin_cmap', colors_list)

fig, ax = plt.subplots(figsize=(7, 6))

g2_plot = np.where(np.isnan(g2_map), 1e0, g2_map)
g2_plot = np.clip(g2_plot, 1e0, 1e9)

im = ax.pcolormesh(
    Delta_arr, lambda_arr, g2_plot,
    norm=LogNorm(vmin=1e0, vmax=1e9),
    cmap=cmap_bin,
    shading='auto'
)

# ------------------------------------------------------------------
# Colorbar
# ------------------------------------------------------------------
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=1.0, anchor=(0.0, 0.0))

cbar.ax.text(1, 1.08, r'$\log_{10} g^{(2)}$',
             transform=cbar.ax.transAxes,
             ha='center', va='top', fontsize=12)

cbar.set_ticks([1e0, 1e2, 1e4, 1e6, 1e8])
cbar.set_ticklabels(['0', '2', '4', '6', '8'])
cbar.ax.tick_params(labelsize=12)

# ------------------------------------------------------------------
# Líneas de resonancia — régimen I 2QD: Delta_n = -n*omega_b - J
# curva: lambda = sqrt((Delta + n*omega_b + J)*omega_b)
# ------------------------------------------------------------------
Delta_fine = np.linspace(-8.0, 0.0, 1000)

for n in n_res:
    lam_curve = np.sqrt(np.maximum((Delta_fine + n * omega_b + J_ob) * omega_b, 0))

    ax.plot(Delta_fine, lam_curve,
            color='black', ls='--', lw=0.9, alpha=0.85)

    # Etiquetas
    target_lam = 0.45
    mask_label = lam_curve > 0
    idx_label  = np.argmin(np.abs(lam_curve[mask_label] - target_lam))
    x_lab = Delta_fine[mask_label][idx_label]
    y_lab = lam_curve[mask_label][idx_label]
    if lambda_arr.min() < y_lab < lambda_arr.max():
        ax.text(x_lab, y_lab * 1e-1,
                rf'$\Delta_{n}(\lambda)$',
                fontsize=16,
                ha='left', va='bottom',
                rotation=90, color='black')

# ------------------------------------------------------------------
# Ejes
# ------------------------------------------------------------------
ax.set_yscale('log')
ax.set_xlim(-6.0, 0.0)
ax.set_ylim(lambda_arr.min(), lambda_arr.max())

ax.set_xticks([-5, -4, -3, -2, -1, 0])
ax.set_xticklabels([r'$-5$', r'$-4$', r'$-3$', r'$-2$', r'$-1$', r'$0$'])

ax.set_yticks([1e-2, 1e-1, 1e0])
ax.set_yticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])

ax.set_xlabel(r'$\Delta/\omega_b$', fontsize=15)
ax.set_ylabel(r'$\lambda/\omega_b$', fontsize=15)
ax.tick_params(labelsize=13)
ax.set_facecolor('white')

ax.text(0.03, 0.04, r'$(b)$',
        transform=ax.transAxes,
        ha='left', va='bottom', fontsize=16,
        bbox=dict(facecolor='white', edgecolor='none', pad=1))

plt.tight_layout()
plt.show()