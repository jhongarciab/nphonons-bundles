import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import math

# =============================================================================
# PARÁMETROS
# =============================================================================
omega_b = 1.0

lam_over_ob = 0.03
Omega_over_ob = 0.003
kappa_over_ob = 0.002
gamma_over_ob = 0.0002
gphi_over_ob = 0.0004

# VALORES DE J
J_values = [0.3, 0.5, 0.8, 1.0]

Delta_list = np.linspace(0.0, -5.0, 501)
n_order = 3

# =============================================================================
# TRUNCAMIENTO
# =============================================================================
Ncut = 5
print(f"Usando Ncut = {Ncut}")

# =============================================================================
# OPERADORES BÁSICOS
# =============================================================================
# Fonón
b = qt.destroy(Ncut)
num_b = b.dag() * b
I_b = qt.qeye(Ncut)

# QDs
sm = qt.sigmam()
sp = sm.dag()
I_q = qt.qeye(2)

# =============================================================================
# OPERADORES DEL SISTEMA COMPUESTO (QD1 ⊗ QD2 ⊗ b)
# =============================================================================
b_sys = qt.tensor(I_q, I_q, b)
num_sys = qt.tensor(I_q, I_q, num_b)

sm1 = qt.tensor(sm, I_q, I_b)
sp1 = sm1.dag()

sm2 = qt.tensor(I_q, sm, I_b)
sp2 = sm2.dag()

proj_e1 = sp1 * sm1
proj_e2 = sp2 * sm2

I_sys = qt.tensor(I_q, I_q, I_b)

# =============================================================================
# OPERADOR ESTABLE b†^n b^n (FACTORIAL)
# =============================================================================
def factorial_number_operator(num_op, n, I_op):
    op = I_op
    for k in range(n):
        op = op * (num_op - k * I_op)
    return op

bdagb_n = factorial_number_operator(num_sys, n_order, I_sys)

# =============================================================================
# OPERADORES DE COLAPSO (LINDBLAD)
# =============================================================================
c_ops = [
    np.sqrt(kappa_over_ob) * b_sys,
    np.sqrt(gamma_over_ob) * sm1,
    np.sqrt(gamma_over_ob) * sm2,
    np.sqrt(gphi_over_ob) * proj_e1,
    np.sqrt(gphi_over_ob) * proj_e2
]

# =============================================================================
# HAMILTONIANOS INDEPENDIENTES DE J
# =============================================================================
H_phonon = omega_b * num_sys

H_interaction = lam_over_ob * (
        proj_e1 + proj_e2
) * (b_sys + b_sys.dag())

H_drive = Omega_over_ob * (
        sm1 + sp1 +
        sm2 + sp2
)

# Término Förster
H_Forster_base = (sp1 * sm2 + sp2 * sm1)

# =============================================================================
# SOLVER ROBUSTO DE ESTADO ESTACIONARIO
# =============================================================================
def validate_steady_state(rho, tol=1e-8):
    if abs(rho.tr() - 1.0) > 1e-6:
        return False
    if not rho.isherm:
        return False
    evals = rho.eigenstates()[0]
    if np.min(np.real(evals)) < -tol:
        return False
    return True

def solve_steady_state_robust(H, c_ops, methods=('direct', 'eigen', 'svd')):
    for method in methods:
        try:
            rho = qt.steadystate(H, c_ops, method=method, use_rcm=True)
            if validate_steady_state(rho):
                return rho
        except Exception:
            pass
    return None

# =============================================================================
# ALMACENAR RESULTADOS
# =============================================================================
results = {}

print(f"\n{'=' * 70}")
print(f"Calculando g^({n_order}) para diferentes valores de J/ωb")
print(f"{'=' * 70}\n")

# =============================================================================
# BUCLE SOBRE VALORES DE J
# =============================================================================
for J_over_ob in J_values:

    print(f"\n>>> J/ωb = {J_over_ob} <<<")

    g_vals = np.full_like(Delta_list, np.nan, dtype=float)
    nbar_vals = np.full_like(Delta_list, np.nan, dtype=float)

    # Hamiltoniano Förster con J actual
    H_Forster = J_over_ob * H_Forster_base

    for i, Delta in enumerate(Delta_list):

        H_detuning = Delta * (proj_e1 + proj_e2)

        H = (
                H_detuning +
                H_phonon +
                H_interaction +
                H_drive +
                H_Forster
        )

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

    # Guardar resultados
    results[J_over_ob] = {
        'g_vals': g_vals.copy(),
        'nbar_vals': nbar_vals.copy()
    }

    print(f"  ✓ Completado para J/ωb = {J_over_ob}")

print(f"\n{'=' * 70}")
print("Todos los cálculos completados")
print(f"{'=' * 70}\n")

# =============================================================================
# 4 SUBPLOTS
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

colors = ['blue', 'green', 'orange', 'red']

for idx, J_over_ob in enumerate(J_values):
    ax = axes[idx]
    ax.plot(Delta_list, results[J_over_ob]['g_vals'],
            lw=1.8,
            color=colors[idx])

    ax.set_yscale('log')
    ax.set_xlim(0.0, -5.0)
    ax.set_xlabel(r'$\Delta/\omega_b$', fontsize=12)
    ax.set_ylabel(rf'$g^{{({n_order})}}(0)$', fontsize=12)
    ax.set_title(f'J/ωb = {J_over_ob}', fontsize=13)
    ax.grid(True, which='both', ls=':', alpha=0.4)

    if n_order == 2:
        ax.set_ylim(0, 1e10)
    elif n_order == 3:
        ax.set_ylim(0, 1e20)
    elif n_order == 4:
        ax.set_ylim(0, 1e30)
    elif n_order == 5:
        ax.set_ylim(0, 1e40)

plt.tight_layout()
plt.savefig(
    f"./figs/g{n_order}_4subplots.pdf",
    bbox_inches="tight"
)
plt.show()