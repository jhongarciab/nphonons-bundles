"""
Funciones de Correlación de distintos órdenes

Cálculo de funciones de correlación de igual tiempo g^(n)(0) para un sistema de
2 QDs acoplados por Förster y acoplados a un modo fonónico común.

Figura apilada con n = 2, 3, 4, 5 en función de Δ/ω_b.
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import LogLocator


# -----------------------------------------------------------------------------
# Estilo global de figura
# -----------------------------------------------------------------------------
rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.size": 12,
})


# -----------------------------------------------------------------------------
# Parámetros físicos adimensionales (normalizados por ω_b)
# -----------------------------------------------------------------------------
omega_b = 1.0

lam_over_ob = 0.08
Omega_over_ob = 0.01
kappa_over_ob = 0.003
gamma_over_ob = 0.0002
gphi_over_ob = 0.0004
J_over_ob = 0.5

Delta_list = np.linspace(0.0, -6.0, 501)
n_orders = [2, 3, 4, 5]


# -----------------------------------------------------------------------------
# Truncamiento de Fock
# -----------------------------------------------------------------------------
Ncut = 8
print(f"Usando Ncut = {Ncut}")


# -----------------------------------------------------------------------------
# Operadores básicos y del sistema compuesto (QD1 ⊗ QD2 ⊗ Fock)
# -----------------------------------------------------------------------------
b = qt.destroy(Ncut)
num_b = b.dag() * b
I_b = qt.qeye(Ncut)

sm = qt.sigmam()
I_q = qt.qeye(2)

b_sys = qt.tensor(I_q, I_q, b)
num_sys = qt.tensor(I_q, I_q, num_b)

sm1 = qt.tensor(sm, I_q, I_b)
sp1 = sm1.dag()
sm2 = qt.tensor(I_q, sm, I_b)
sp2 = sm2.dag()

proj_e1 = sp1 * sm1
proj_e2 = sp2 * sm2

I_sys = qt.tensor(I_q, I_q, I_b)


# -----------------------------------------------------------------------------
# Utilidades numéricas
# -----------------------------------------------------------------------------
def factorial_number_operator(num_op, n, I_op):
    """Construye b†^n b^n usando forma factorial del operador número."""
    op = I_op
    for k in range(n):
        op = op * (num_op - k * I_op)
    return op


def validate_steady_state(rho, tol=1e-8):
    if abs(rho.tr() - 1.0) > 1e-6:
        return False
    if not rho.isherm:
        return False
    evals = rho.eigenstates()[0]
    if np.min(np.real(evals)) < -tol:
        return False
    return True


def solve_steady_state_robust(H, c_ops, methods=("direct", "eigen", "svd")):
    for method in methods:
        try:
            rho = qt.steadystate(H, c_ops, method=method, use_rcm=True)
            if validate_steady_state(rho):
                return rho
        except Exception:
            pass
    return None


# -----------------------------------------------------------------------------
# Lindblad (colapsos) y Hamiltonianos fijos
# -----------------------------------------------------------------------------
c_ops = [
    np.sqrt(kappa_over_ob) * b_sys,
    np.sqrt(gamma_over_ob) * sm1,
    np.sqrt(gamma_over_ob) * sm2,
    np.sqrt(gphi_over_ob) * proj_e1,
    np.sqrt(gphi_over_ob) * proj_e2,
]

H_phonon = omega_b * num_sys
H_interaction = lam_over_ob * (proj_e1 + proj_e2) * (b_sys + b_sys.dag())
H_drive = Omega_over_ob * (sm1 + sp1 + sm2 + sp2)
H_Forster = J_over_ob * (sp1 * sm2 + sp2 * sm1)


# -----------------------------------------------------------------------------
# Cálculo de g^(n)
# -----------------------------------------------------------------------------
results = {}

print(f"\n{'=' * 70}")
print(f"Calculando g^(n) para órdenes {n_orders} con J/ωb = {J_over_ob}")
print(f"{'=' * 70}\n")

for n_order in n_orders:
    print(f"\n>>> Calculando g^({n_order}) <<<")

    bdagb_n = factorial_number_operator(num_sys, n_order, I_sys)
    g_vals = np.full_like(Delta_list, np.nan, dtype=float)
    nbar_vals = np.full_like(Delta_list, np.nan, dtype=float)

    for i, Delta in enumerate(Delta_list):
        H_detuning = Delta * (proj_e1 + proj_e2)
        H = H_detuning + H_phonon + H_interaction + H_drive + H_Forster

        rho_ss = solve_steady_state_robust(H, c_ops)
        if rho_ss is None:
            continue

        nbar = qt.expect(num_sys, rho_ss)
        nbar_vals[i] = nbar

        if nbar > 1e-12:
            numerator = qt.expect(bdagb_n, rho_ss)
            g_val = numerator / (nbar ** n_order)
            g_vals[i] = np.real(g_val) if abs(np.imag(g_val)) < 1e-10 else g_val

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(Delta_list)}  Δ={Delta:.2f}  ⟨n⟩={nbar:.2e}")

    results[n_order] = {"g_vals": g_vals.copy(), "nbar_vals": nbar_vals.copy()}
    print(f"  ✓ Completado para g^({n_order})")

print(f"\n{'=' * 70}")
print("Todos los cálculos completados")
print(f"{'=' * 70}\n")


# -----------------------------------------------------------------------------
# Figura apilada g^(n)(0) vs Δ/ω_b
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(3.37, 5.73), sharex=True)
fig.subplots_adjust(hspace=0.08)

color_by_order = {2: "blue", 3: "green", 4: "#c4942a", 5: "red"}

ylims = {
    2: (1e-1, 1e7),
    3: (1e-1, 1e16),
    4: (1e-1, 1e23),
    5: (1e-1, 1e29),
}

resonancias = {
    2: -2 * omega_b - J_over_ob,
    3: -3 * omega_b - J_over_ob,
    4: -4 * omega_b - J_over_ob,
    5: -5 * omega_b - J_over_ob,
}

y_label = {2: 3e5, 3: 1e13, 4: 1e20, 5: 4e25}
x_label = {2: -2.5, 3: -3.5, 4: -4.5, 5: -5.2}


for idx, (ax, n_order) in enumerate(zip(axes, n_orders[::-1])):

    ax.plot(Delta_list, results[n_order]["g_vals"], lw=1.6, color=color_by_order[n_order])

    ax.set_yscale("log")
    ax.set_xlim(0.0, -6.0)
    ax.set_ylim(ylims[n_order])
    ax.set_ylabel(rf"$g^{{({n_order})}}(0)$", fontsize=16)

    ax.grid(False)
    ax.set_facecolor("white")
    ax.set_xticks([0, -1, -2, -3, -4, -5])

    ymin, ymax = ylims[n_order]
    exp_min = int(np.floor(np.log10(ymin)))
    exp_max = int(np.floor(np.log10(ymax)))
    exp_ticks = np.linspace(exp_min, exp_max, 5)
    exp_ticks = np.round(exp_ticks).astype(int)
    y_ticks = [10.0**e for e in exp_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([rf"$10^{{{e}}}$" for e in exp_ticks])
    ax.tick_params(labelsize=15)

    delta_n = resonancias[n_order]
    ymin, ymax = ylims[n_order]

    idx_res = np.argmin(np.abs(Delta_list - delta_n))
    g_at_res = results[n_order]["g_vals"][idx_res]
    y_top = g_at_res if (not np.isnan(g_at_res) and g_at_res > ymin) else ymax

    ax.plot([delta_n, delta_n], [ymin, y_top], color=color_by_order[n_order], lw=0.9, ls="--", alpha=0.8)

    val = delta_n / omega_b
    ax.text(
        x_label[n_order],
        y_label[n_order],
        rf"$\Delta \approx {val}\,\omega_b$",
        ha="center",
        va="bottom",
        fontsize=16,
        color=color_by_order[n_order],
    )


axes[-1].set_xlabel(r"$\Delta/\omega_b$", fontsize=24)
fig.text(0.02, 0.98, r'$(a)$', ha='left', va='top', fontsize=16)


# -----------------------------------------------------------------------------
# Salida
# -----------------------------------------------------------------------------
plt.tight_layout()
#plt.show()
plt.savefig("./figs/oficial/g_n_orders_stacked.pdf", bbox_inches="tight")
plt.savefig("./figs/oficial/pgf/g_n_orders_stacked.pgf")
plt.close()
