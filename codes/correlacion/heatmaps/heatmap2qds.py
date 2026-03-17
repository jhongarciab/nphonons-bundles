"""
heatmap_combined_2qds.py

Figura combinada: dos heatmaps g^(3) apilados verticalmente con eje x compartido.
  Panel (b) — barrido en lambda/omega_b  (heatmap_2qds.py)
  Panel (c) — barrido en Omega/omega_b   (heatmap2_2qds.py)
Una colorbar compartida a la derecha.
"""

import numpy as np
import qutip as qt
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
import matplotlib.colors as mcolors
from scipy.interpolate import NearestNDInterpolator

rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.size": 12,
})

# =============================================================================
# PARÁMETROS COMUNES
# =============================================================================
omega_b  = 1.0
kappa_ob = 0.003
gamma_ob = 0.0004
gphi_ob  = 0.0004
J_ob     = 0.5
Ncut     = 8

n_Delta  = 30
n_sweep  = 10
Delta_arr = np.linspace(0.0, -7.0, n_Delta)

# Panel (b): barrido en lambda, Omega fijo
Omega_ob  = 0.01
lambda_arr = np.logspace(-2.2, 0, n_sweep)

# Panel (c): barrido en Omega, lambda fijo
lambda_ob = 0.08
Omega_arr = np.logspace(-2.2, 0, n_sweep)

# =============================================================================
# OPERADORES — QD1 ⊗ QD2 ⊗ Fock  (compartidos)
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

# Operador factorial n=3
def factorial_op(num_op, n, I_op):
    op = I_op
    for k in range(n):
        op = op * (num_op - k * I_op)
    return op

bdagb3 = factorial_op(num_sys, 3, I_sys)

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
# BARRIDO (b): lambda variable, Omega fijo
# =============================================================================
print("=== Barrido panel (b): λ variable ===")
g3_map_b = np.full((n_sweep, n_Delta), np.nan)

H_phonon_b  = omega_b * num_sys
H_drive_b   = Omega_ob * (sm1 + sp1 + sm2 + sp2)
H_Forster_b = J_ob * (sp1 * sm2 + sp2 * sm1)

total = n_sweep * n_Delta
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
        H = H_phonon_b + H_eph + H_drive_b + H_Forster_b + H_det

        rho_ss = solve_ss(H, c_ops)
        count += 1

        if rho_ss is None:
            continue

        nbar = qt.expect(num_sys, rho_ss)
        if nbar > 1e-8:
            num = qt.expect(bdagb3, rho_ss)
            g3_map_b[i, j] = np.real(num) / nbar**3

        if count % 50 == 0:
            val = f"{g3_map_b[i,j]:.2e}" if not np.isnan(g3_map_b[i,j]) else "nan"
            print(f"  {count}/{total}  λ={lam:.3f}  Δ={Delta:.2f}  g3={val}")

print("✓ Barrido (b) completado")

# Interpolación zona problemática panel (b)
mask_valid = ~np.isnan(g3_map_b) & (g3_map_b > 1e1)
mask_bad   = ~mask_valid
for i, lam in enumerate(lambda_arr):
    for j, Delta in enumerate(Delta_arr):
        if lam > 0.05 or Delta > -2.0:
            mask_bad[i, j] = False

if mask_valid.any() and mask_bad.any():
    coords  = np.array(np.where(mask_valid)).T
    values  = g3_map_b[mask_valid]
    interp  = NearestNDInterpolator(coords, values)
    bad_c   = np.array(np.where(mask_bad)).T
    g3_map_b[mask_bad] = interp(bad_c)

# =============================================================================
# BARRIDO (c): Omega variable, lambda fijo
# =============================================================================
print("\n=== Barrido panel (c): Ω variable ===")
g3_map_c = np.full((n_sweep, n_Delta), np.nan)

H_phonon_c  = omega_b * num_sys
H_eph_c     = lambda_ob * (proj_e1 + proj_e2) * (b_sys + b_sys.dag())
H_Forster_c = J_ob * (sp1 * sm2 + sp2 * sm1)

total = n_sweep * n_Delta
count = 0

for i, Omega in enumerate(Omega_arr):
    c_ops = [
        np.sqrt(kappa_ob) * b_sys,
        np.sqrt(gamma_ob) * sm1,
        np.sqrt(gamma_ob) * sm2,
        np.sqrt(gphi_ob)  * proj_e1,
        np.sqrt(gphi_ob)  * proj_e2,
    ]
    H_drive = Omega * (sm1 + sp1 + sm2 + sp2)

    for j, Delta in enumerate(Delta_arr):
        H_det = Delta * (proj_e1 + proj_e2)
        H = H_phonon_c + H_eph_c + H_drive + H_Forster_c + H_det

        rho_ss = solve_ss(H, c_ops)
        count += 1

        if rho_ss is None:
            continue

        nbar = qt.expect(num_sys, rho_ss)
        if nbar > 1e-8:
            num = qt.expect(bdagb3, rho_ss)
            g3_map_c[i, j] = np.real(num) / nbar**3

        if count % 50 == 0:
            val = f"{g3_map_c[i,j]:.2e}" if not np.isnan(g3_map_c[i,j]) else "nan"
            print(f"  {count}/{total}  Ω={Omega:.3f}  Δ={Delta:.2f}  g3={val}")

print("✓ Barrido (c) completado")

# =============================================================================
# FIGURA COMBINADA — eje x compartido, colorbar única
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
norm     = LogNorm(vmin=1e0, vmax=1e15)
levels   = np.logspace(0, 15, 220)

# sharex=True: eje x idéntico en ambos paneles
fig, (ax_b, ax_c) = plt.subplots(
    2, 1,
    figsize=(2.95, 4.80),
    sharex=True,
    gridspec_kw={
        'hspace': 0.10,
        'height_ratios': [1, 1]
    }
)

Delta_fine = np.linspace(-9.0, 0.0, 1000)

# ------------------------------------------------------------------
# Panel (b) — arriba
# ------------------------------------------------------------------
g3_b = np.where(np.isnan(g3_map_b), 1e0, g3_map_b)
g3_b = np.clip(g3_b, 1e0, 1e15)

im = ax_b.contourf(
    Delta_arr, lambda_arr, g3_b,
    levels=levels, norm=norm,
    cmap=cmap_bin, antialiased=False,
    zorder=-1
)
# Compatible con matplotlib >= 3.8
for col in im.get_paths():
    pass  # no aplica aquí
ax_b.set_rasterization_zorder(0)  # rasteriza todo lo que esté en zorder < 0

# Líneas de resonancia régimen I
for n in n_res:
    lam_curve = np.sqrt(np.maximum((Delta_fine + n * omega_b + J_ob) * omega_b, 0))
    ax_b.plot(Delta_fine, lam_curve, color='black', ls='--', lw=0.9, alpha=0.85)

# Labels panel (b)
labels_b = {
    1: (-1.45, 0.02), 2: (-2.4, 0.02), 3: (-3.4, 0.02),
    4: (-4.4, 0.02), 5: (-5.4, 0.02), 6: (-6.4, 0.02),
}
for n, (x_lab, y_lab) in labels_b.items():
    ax_b.text(x_lab, y_lab, rf'$\Delta_{n}(\lambda)$',
              fontsize=10, ha='left', va='bottom', rotation=90, color='black')

ax_b.set_yscale('log')
ax_b.set_xlim(-6.5, -1.0)
ax_b.set_ylim(lambda_arr.min(), lambda_arr.max())
ax_b.set_yticks([1e-2, 1e-1, 1e0])
ax_b.set_yticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])
ax_b.set_ylabel(r'$\lambda/\omega_b$', fontsize=12)
ax_b.tick_params(labelsize=12)
ax_b.set_facecolor('white')

# ------------------------------------------------------------------
# Panel (c) — abajo
# ------------------------------------------------------------------
g3_c = np.where(np.isnan(g3_map_c), 1e0, g3_map_c)
g3_c = np.clip(g3_c, 1e0, 1e15)

ax_c.contourf(
    Delta_arr, Omega_arr, g3_c,
    levels=levels, norm=norm,
    cmap=cmap_bin, antialiased=False,
    zorder=-1
)
ax_c.set_rasterization_zorder(0)

# Líneas de resonancia régimen III
for n in n_res:
    arg = (n * omega_b)**2 - (Delta_fine + J_ob)**2
    Omega_curve = np.sqrt(np.maximum(arg / 8, 0))
    ax_c.plot(Delta_fine, Omega_curve, color='black', ls='--', lw=0.9, alpha=0.85)

# Labels panel (c)
labels_c = {
    1: (-1.45, 0.02), 2: (-2.4, 0.02), 3: (-3.4, 0.02),
    4: (-4.4, 0.02), 5: (-5.4, 0.02), 6: (-6.4, 0.02),
}
for n, (x_lab, y_lab) in labels_c.items():
    ax_c.text(x_lab, y_lab, rf'$\Delta_{n}(\Omega)$',
              fontsize=10, ha='left', va='bottom', rotation=90, color='black')

ax_c.set_yscale('log')
ax_c.set_ylim(Omega_arr.min(), Omega_arr.max())
ax_c.set_yticks([1e-2, 1e-1, 1e0])
ax_c.set_yticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])
ax_c.set_xticks([-6, -5, -4, -3, -2, -1])
ax_c.set_xticklabels([r'$-6$', r'$-5$', r'$-4$', r'$-3$', r'$-2$', r'$-1$'])
ax_c.set_xlabel(r'$\Delta/\omega_b$', fontsize=12)
ax_c.set_ylabel(r'$\Omega/\omega_b$', fontsize=12)
ax_c.tick_params(labelsize=12)
ax_c.set_facecolor('white')

# ------------------------------------------------------------------
# Colorbar única compartida
# ------------------------------------------------------------------
# Colorbar manual — ocupa exactamente el área de los dos paneles
fig.subplots_adjust(
    left=0.22,
    right=0.82,
    top=0.92,
    bottom=0.12,
    hspace=0.10,
)
pos_b = ax_b.get_position()
pos_c = ax_c.get_position()
cax = fig.add_axes([
    pos_b.x1 + 0.02,          # x: justo a la derecha de los paneles
    pos_c.y0,                  # y: desde el borde inferior del panel (c)
    0.03,                      # ancho de la barra
    pos_b.y1 - pos_c.y0        # alto: desde bottom de (c) hasta top de (b)
])
cbar = fig.colorbar(im, cax=cax)
cbar.set_ticks([1e0, 1e3, 1e6, 1e9, 1e12, 1e15])
cbar.set_ticklabels(['0', '3', '6', '9', '12', '15'])
cbar.ax.tick_params(labelsize=10)

# =============================================================================
# SALIDA
# =============================================================================
plt.savefig("./figs/oficial/corr_2qds_combined_heatmap.pdf", bbox_inches="tight")
plt.savefig("./figs/oficial/pgf/corr_2qds_combined_heatmap.pgf")
plt.close()
print("\n✓ Figura combinada guardada.")