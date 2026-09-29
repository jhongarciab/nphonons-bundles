"""
Zoom de alta resolución del dip de g^(3)(0) cerca de Δ ≈ -3.5 ω_b
Guarda: results/data/g3_zoom_fine_data.npz
"""

import numpy as np
import qutip as qt
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

CODES_DIR = Path(__file__).resolve().parents[3]
RESULTS_DATA_DIR = CODES_DIR / "results" / "data"
RESULTS_OFICIAL_DIR = CODES_DIR / "results" / "oficial"
RESULTS_PGF_DIR = RESULTS_OFICIAL_DIR / "pgf"
RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_OFICIAL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PGF_DIR.mkdir(parents=True, exist_ok=True)

RERUN = False  # False = calcular y guardar; True = cargar y graficar

# --- Parámetros (idénticos al script principal) ---
omega_b      = 1.0
lam_over_ob  = 0.08
Omega_over_ob= 0.01
kappa_over_ob= 0.003
gamma_over_ob= 0.0002
gphi_over_ob = 0.0004
J_over_ob    = 0.5
Ncut         = 8

Delta_fine = np.linspace(-3.42, -3.62, 1200)  # alta resolución

# --- Operadores ---
b     = qt.destroy(Ncut)
num_b = b.dag() * b
I_b   = qt.qeye(Ncut)
sm    = qt.sigmam()
I_q   = qt.qeye(2)

b_sys   = qt.tensor(I_q, I_q, b)
num_sys = qt.tensor(I_q, I_q, num_b)
sm1     = qt.tensor(sm,  I_q, I_b);  sp1 = sm1.dag()
sm2     = qt.tensor(I_q, sm,  I_b);  sp2 = sm2.dag()
proj_e1 = sp1 * sm1
proj_e2 = sp2 * sm2
I_sys   = qt.tensor(I_q, I_q, I_b)

c_ops = [
    np.sqrt(kappa_over_ob) * b_sys,
    np.sqrt(gamma_over_ob) * sm1,
    np.sqrt(gamma_over_ob) * sm2,
    np.sqrt(gphi_over_ob)  * proj_e1,
    np.sqrt(gphi_over_ob)  * proj_e2,
]

H_phonon     = omega_b      * num_sys
H_interaction= lam_over_ob  * (proj_e1 + proj_e2) * (b_sys + b_sys.dag())
H_drive      = Omega_over_ob* (sm1 + sp1 + sm2 + sp2)
H_Forster    = J_over_ob    * (sp1 * sm2 + sp2 * sm1)

# b†³ b³ en forma factorial
bdagb3 = I_sys
for k in range(3):
    bdagb3 = bdagb3 * (num_sys - k * I_sys)

def solve_ss(H, c_ops):
    for method in ("direct", "eigen", "svd"):
        try:
            rho = qt.steadystate(H, c_ops, method=method, use_rcm=True)
            if abs(rho.tr() - 1.0) < 1e-6 and rho.isherm:
                return rho
        except Exception:
            pass
    return None

# --- Cálculo o carga ---
if not RERUN:
    g3_fine = np.full_like(Delta_fine, np.nan)

    for i, Delta in enumerate(Delta_fine):
        H = Delta * (proj_e1 + proj_e2) + H_phonon + H_interaction + H_drive + H_Forster
        rho_ss = solve_ss(H, c_ops)
        if rho_ss is None:
            continue
        nbar = qt.expect(num_sys, rho_ss)
        if nbar > 1e-12:
            g3_fine[i] = np.real(qt.expect(bdagb3, rho_ss)) / nbar**3
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(Delta_fine)}  Δ={Delta:.5f}  g3={g3_fine[i]:.3e}")

    np.savez(RESULTS_DATA_DIR / "g3_zoom_fine_data.npz",
             Delta_fine=Delta_fine, g3_fine=g3_fine)
    print("✓ Guardado: g3_zoom_fine_data.npz")

else:
    data       = np.load(RESULTS_DATA_DIR / "g3_zoom_fine_data.npz")
    Delta_fine = data["Delta_fine"]
    g3_fine    = data["g3_fine"]

# --- Figura ---
rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts":   False,
    "font.family":   "serif",
    "font.size":     12,
})

fig, ax = plt.subplots(figsize=(3.65, 2.6))
ax.plot(Delta_fine, g3_fine, lw=1.1, color="green")
ax.set_xlim(-3.42, -3.57)
ax.set_ylim(3e10, 9e11)
ax.set_yscale("log")
ax.set_facecolor("white")
ax.grid(False)
ax.tick_params(labelsize=11)
ax.set_xticks([-3.45, -3.50, -3.55])
ax.set_xticklabels([r"$-3.45$", r"$-3.50$", r"$-3.55$"])
ax.set_yticks([1e11])
ax.set_yticklabels([r"$10^{11}$"])

fig.subplots_adjust(left=0.18, right=0.95, top=0.94, bottom=0.20)
plt.savefig(RESULTS_OFICIAL_DIR / "g3_dip_zoom.pdf",  bbox_inches="tight")
plt.savefig(RESULTS_PGF_DIR / "g3_dip_zoom.pgf")
plt.close()
print("Figura guardada")
