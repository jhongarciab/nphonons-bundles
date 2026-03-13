import numpy as np
import qutip as qt
import matplotlib.pyplot as plt


def odd(num):
    return num & 0x1


omega_b = 1.0
lam_over_ob = 0.03
Omega_over_ob = 0.003
kappa_over_ob = 0.002
gamma_over_ob = 0.0002
gphi_over_ob = 0.0004

Delta_list = np.linspace(0.0, -6.0, 501)
n_order = 2

Ncut = 5
pdef = -0.5  # Caso normal: 0.0 | Deformado: pdef ~ -0.49
print(f"Usando Ncut = {Ncut}, pdef = {pdef}")

# Operador bosónico deformado
n_max = Ncut - 1
superdiag = np.sqrt(np.add((np.arange(n_max) + 1), 2.0 * pdef * odd(np.arange(n_max) + 1)))
a_np = np.diag(superdiag, 1)
a_dag_np = a_np.T
n_np = np.add(a_dag_np.dot(a_np), np.diag(-2.0 * pdef * odd(np.arange(n_max + 1))))

b = qt.Qobj(a_np)
num_b = qt.Qobj(n_np)
I_b = qt.qeye(Ncut)

sm = qt.sigmam()
sp = qt.sigmap()
I_q = qt.qeye(2)

b_sys = qt.tensor(I_q, b)
num_sys = qt.tensor(I_q, num_b)
sm_sys = qt.tensor(sm, I_b)
sp_sys = qt.tensor(sp, I_b)
proj_e = qt.tensor(sp * sm, I_b)

I_sys = qt.tensor(I_q, I_b)


def factorial_number_operator(num_op, n, I_op):
    op = I_op
    for k in range(n):
        op = op * (num_op - k * I_op)
    return op


bdagb_n = factorial_number_operator(num_sys, n_order, I_sys)

c_ops = [
    np.sqrt(kappa_over_ob) * b_sys,
    np.sqrt(gamma_over_ob) * sm_sys,
    np.sqrt(gphi_over_ob) * proj_e,
]

H_phonon = omega_b * num_sys
H_interaction = lam_over_ob * qt.tensor(sp * sm, b + b.dag())
H_drive = Omega_over_ob * (sm_sys + sp_sys)


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


g_vals = np.full_like(Delta_list, np.nan, dtype=float)
nbar_vals = np.full_like(Delta_list, np.nan, dtype=float)

print(f"\nCalculando g^({n_order}) con pdef={pdef}")

for i, Delta in enumerate(Delta_list):
    H = Delta * proj_e + H_phonon + H_interaction + H_drive
    rho_ss = solve_steady_state_robust(H, c_ops)

    if rho_ss is None:
        continue

    nbar = qt.expect(num_sys, rho_ss)
    nbar_vals[i] = nbar

    if nbar > 1e-12:
        numerator = qt.expect(bdagb_n, rho_ss)
        g_val = numerator / (nbar**n_order)
        g_vals[i] = np.real(g_val) if np.abs(np.imag(g_val)) < 1e-10 else g_val

    if (i + 1) % 100 == 0:
        print(f"{i+1}/{len(Delta_list)}  Δ={Delta:.2f}  ⟨n⟩={nbar:.2e}")

print("Cálculo completado.")

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(Delta_list, g_vals, lw=1.5, label=rf"$p_{{def}}={pdef}$")
ax.set_yscale("log")
ax.set_xlim(0.0, -6.0)
ax.set_xlabel(r"$\Delta/\omega_b$", fontsize=12)
ax.set_ylabel(rf"$g^{{({n_order})}}(0)$", fontsize=12)
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend()

if n_order == 2:
    ax.set_ylim(1e-2, 1e10)
elif n_order == 3:
    ax.set_ylim(1e-2, 1e20)
elif n_order == 4:
    ax.set_ylim(1e-2, 1e30)
elif n_order == 5:
    ax.set_ylim(1e-2, 1e40)

plt.tight_layout()
plt.savefig(f"g_{n_order}_pdef_{pdef:.2f}.png", dpi=300)
