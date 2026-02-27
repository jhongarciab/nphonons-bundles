import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# =============================================================================
# PARÁMETROS FIJOS — Bin Fig. 3(c)
# =============================================================================
omega_b   = 1.0
lambda_ob = 0.03    # fijo, régimen III: λ ≪ ωb
kappa_ob  = 0.002
gamma_ob  = 0.0002
gphi_ob   = 0.0004

Ncut = 30

# =============================================================================
# GRILLA 2D: Delta x Omega
# =============================================================================
n_Delta = 30
n_Omega = 10
Delta_arr = np.linspace(0.0, -6.0, n_Delta)
Omega_arr = np.logspace(-2, 0, n_Omega)

g2_map = np.full((n_Omega, n_Delta), np.nan)

# =============================================================================
# OPERADORES BÁSICOS 1QD
# =============================================================================
b    = qt.destroy(Ncut)
numb = b.dag() * b
I_b  = qt.qeye(Ncut)
sm   = qt.sigmam()
sp   = sm.dag()
I_q  = qt.qeye(2)

b_sys   = qt.tensor(I_q, b)
num_sys = qt.tensor(I_q, numb)
sm_sys  = qt.tensor(sm, I_b)
sp_sys  = sm_sys.dag()
proj_e  = sp_sys * sm_sys
I_sys   = qt.tensor(I_q, I_b)

def factorial_op(num_op, n, I_op):
    op = I_op
    for k in range(n):
        op = op * (num_op - k * I_op)
    return op

bdagb2 = factorial_op(num_sys, 2, I_sys)

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
H_phonon = omega_b * num_sys
H_eph    = lambda_ob * proj_e * (b_sys + b_sys.dag())  # lambda fijo

# =============================================================================
# BARRIDO 2D
# =============================================================================
total = n_Omega * n_Delta
count = 0

for i, Omega in enumerate(Omega_arr):
    c_ops = [
        np.sqrt(kappa_ob) * b_sys,
        np.sqrt(gamma_ob) * sm_sys,
        np.sqrt(gphi_ob)  * proj_e,
    ]

    H_drive = Omega * (sm_sys + sp_sys)  # Omega varía

    for j, Delta in enumerate(Delta_arr):
        H_det = Delta * proj_e
        H = H_phonon + H_det + H_eph + H_drive

        rho_ss = solve_ss(H, c_ops)
        count += 1

        if rho_ss is None:
            continue

        nbar = qt.expect(num_sys, rho_ss)
        if nbar > 1e-12:
            num = qt.expect(bdagb2, rho_ss)
            g2_map[i, j] = np.real(num) / nbar**2

        if count % 50 == 0:
            print(f"  {count}/{total}  Ω={Omega:.3f}  Δ={Delta:.2f}  "
                  f"g2={g2_map[i,j]:.2e}" if not np.isnan(g2_map[i,j])
                  else f"  {count}/{total}  Ω={Omega:.3f}  Δ={Delta:.2f}  nan")

print("\n✓ Barrido completado")

# =============================================================================
# PLOT
# =============================================================================
from matplotlib import rcParams
import matplotlib.colors as mcolors

rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 13,
})

n_res = [1, 2, 3, 4, 5]

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
    Delta_arr, Omega_arr, g2_plot,
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
# Líneas de resonancia — régimen III: Delta_n(Omega) = -sqrt(n²ωb² - 4Ω²)
# ------------------------------------------------------------------
Omega_fine = np.logspace(-2, 0, 1000)

for n in n_res:
    arg = (n * omega_b)**2 - 4 * Omega_fine**2
    mask = arg > 0
    Delta_curve = -np.sqrt(arg[mask])

    ax.plot(Delta_curve, Omega_fine[mask],
            color='black', ls='--', lw=0.9, alpha=0.85)

    # Etiquetas
    target_Omega = 0.45
    idx_label = np.argmin(np.abs(Omega_fine[mask] - target_Omega))
    x_lab = Delta_curve[idx_label]
    y_lab = Omega_fine[mask][idx_label]
    if Omega_arr.min() < y_lab < Omega_arr.max():
        ax.text(x_lab, y_lab * 1e-1,
                rf'$\Delta_{n}(\Omega)$',
                fontsize=16,
                ha='left', va='bottom',
                rotation=90, color='black')

# ------------------------------------------------------------------
# Ejes
# ------------------------------------------------------------------
ax.set_yscale('log')
ax.set_xlim(-6.0, 0.0)
ax.set_ylim(Omega_arr.min(), Omega_arr.max())

ax.set_xticks([-5, -4, -3, -2, -1, 0])
ax.set_xticklabels([r'$-5$', r'$-4$', r'$-3$', r'$-2$', r'$-1$', r'$0$'])

ax.set_yticks([1e-2, 1e-1, 1e0])
ax.set_yticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])

ax.set_xlabel(r'$\Delta/\omega_b$', fontsize=15)
ax.set_ylabel(r'$\Omega/\omega_b$', fontsize=15)
ax.tick_params(labelsize=13)
ax.set_facecolor('white')

ax.text(0.03, 0.04, r'$(c)$',
        transform=ax.transAxes,
        ha='left', va='bottom', fontsize=16,
        bbox=dict(facecolor='white', edgecolor='none', pad=1))

plt.tight_layout()
plt.show()