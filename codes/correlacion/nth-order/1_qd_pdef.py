import numpy as np
import qutip as qt
import matplotlib.pyplot as plt


def odd(num):
    return num & 0x1


# ─── Parámetros del sistema ───────────────────────────────────────────────────
omega_b      = 1.0
lam_over_ob  = 0.03
Omega_over_ob= 0.003
kappa_over_ob= 0.002
gamma_over_ob= 0.0002
gphi_over_ob = 0.0004

Delta_list = np.linspace(0.0, -6.0, 501)
n_order    = 2

Ncut = 8
pdef = -0.49   # Caso estándar: 0.0 | Deformado: pdef in (-0.5, 0.5)
               # NOTA: pdef = ±0.5 exacto desconecta estados — no usar

print(f"Usando Ncut = {Ncut}, pdef = {pdef}")

# ─── Operador bosónico DEFORMADO (solo para el Hamiltoniano) ──────────────────
# Relación de conmutación: [a, a†] = 1 + 2p(-1)^n
# Elementos de matriz: a|n> = sqrt(n + 2p·odd(n)) |n-1>
n_max    = Ncut - 1
superdiag = np.sqrt(
    np.add((np.arange(n_max) + 1),
           2.0 * pdef * odd(np.arange(n_max) + 1))
)
a_np     = np.diag(superdiag, 1)   # superdiagonal -> operador aniquilación
a_dag_np = a_np.T
# Operador número deformado: N_p = a†a - 2p·odd(n) (diagonal)
n_np     = np.add(
    a_dag_np.dot(a_np),
    np.diag(-2.0 * pdef * odd(np.arange(n_max + 1)))
)

b_def   = qt.Qobj(a_np)    # operador aniquilación DEFORMADO
num_def = qt.Qobj(n_np)    # operador número DEFORMADO

# ─── Operador bosónico ESTÁNDAR (solo para el baño / Lindblad) ────────────────
b_std = qt.destroy(Ncut)   # operador aniquilación estándar

# ─── Identidades ──────────────────────────────────────────────────────────────
I_b = qt.qeye(Ncut)
I_q = qt.qeye(2)
sm  = qt.sigmam()
sp  = qt.sigmap()

# ─── Operadores en el espacio total QD ⊗ Fonón ───────────────────────────────
# Hamiltoniano: usa operadores DEFORMADOS
b_def_sys = qt.tensor(I_q, b_def)
num_def_sys = qt.tensor(I_q, num_def)

# Lindblad fonónico: usa operador ESTÁNDAR (baño físico no deformado)
b_std_sys = qt.tensor(I_q, b_std)

sm_sys   = qt.tensor(sm, I_b)
sp_sys   = qt.tensor(sp, I_b)
proj_e   = qt.tensor(sp * sm, I_b)   # proyector sobre estado excitado
I_sys    = qt.tensor(I_q, I_b)

# ─── Operadores de colapso ────────────────────────────────────────────────────
# Pérdida fonónica: baño estándar (modelo A)
# Decaimiento y desfase del QD: sin cambios
c_ops = [
    np.sqrt(kappa_over_ob) * b_std_sys,   # pérdida fonónica — baño ESTÁNDAR
    np.sqrt(gamma_over_ob) * sm_sys,       # decaimiento espontáneo del QD
    np.sqrt(gphi_over_ob)  * proj_e,       # desfase puro del QD
]

# ─── Hamiltoniano ─────────────────────────────────────────────────────────────
# Energía fonónica: operador número DEFORMADO
H_phonon      = omega_b * num_def_sys
# Interacción Holstein: usa b DEFORMADO (acoplamiento QD-fonón deformado)
H_interaction = lam_over_ob * qt.tensor(sp * sm, b_def + b_def.dag())
# Bombeo coherente del QD
H_drive       = Omega_over_ob * (sm_sys + sp_sys)

# ─── g^(n) de Glauber — definición correcta con potencias ────────────────────
# g^(n)(0) = <(b†)^n b^n> / <b†b>^n
# Se usan potencias del operador DEFORMADO (consistente con el Hamiltoniano)
# NO se usa el operador factorial del número — ese es numéricamente inestable
bdag_n_b_n = (b_def_sys.dag()**n_order) * (b_def_sys**n_order)


# ─── Funciones auxiliares ─────────────────────────────────────────────────────
def validate_steady_state(rho, tol=1e-8):
    """Verifica traza=1, hermiticidad y positividad."""
    if abs(rho.tr() - 1.0) > 1e-6:
        return False
    if not rho.isherm:
        return False
    evals = rho.eigenstates()[0]
    if np.min(np.real(evals)) < -tol:
        return False
    return True


def solve_steady_state_robust(H, c_ops, methods=("direct", "eigen", "svd")):
    """Intenta resolver el estado estacionario con varios métodos."""
    for method in methods:
        try:
            rho = qt.steadystate(H, c_ops, method=method, use_rcm=True)
            if validate_steady_state(rho):
                return rho
        except Exception:
            pass
    return None


# ─── Barrido en Delta ─────────────────────────────────────────────────────────
g_vals   = np.full_like(Delta_list, np.nan, dtype=float)
nbar_vals= np.full_like(Delta_list, np.nan, dtype=float)

print(f"\nCalculando g^({n_order}) con pdef={pdef}")

for i, Delta in enumerate(Delta_list):
    H = Delta * proj_e + H_phonon + H_interaction + H_drive
    rho_ss = solve_steady_state_robust(H, c_ops)

    if rho_ss is None:
        continue

    # Número medio de fonones con operador DEFORMADO
    nbar = qt.expect(b_def_sys.dag() * b_def_sys, rho_ss)
    nbar_vals[i] = nbar

    if nbar > 1e-12:
        # g^(n) con definición de potencias — consistente con Glauber y Bin
        numerator = qt.expect(bdag_n_b_n, rho_ss)
        g_raw     = numerator / (nbar**n_order)
        g_vals[i] = (np.real(g_raw)
                     if np.abs(np.imag(g_raw)) < 1e-10
                     else np.nan)

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(Delta_list)}  Δ={Delta:.2f}  ⟨n⟩={nbar:.2e}")

print("Cálculo completado.")

# ─── Figura ───────────────────────────────────────────────────────────────────
ylim_map = {2: (1e-2, 1e10), 3: (1e-2, 1e20), 4: (1e-2, 1e30), 5: (1e-2, 1e40)}

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(Delta_list, g_vals, lw=1.5, label=rf"$p_{{\rm def}}={pdef}$")
ax.set_yscale("log")
ax.set_xlim(Delta_list[0], Delta_list[-1])
ax.set_xlabel(r"$\Delta/\omega_b$", fontsize=12)
ax.set_ylabel(rf"$g^{{({n_order})}}(0)$", fontsize=12)
ax.set_ylim(*ylim_map.get(n_order, (1e-2, 1e10)))
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend()
plt.tight_layout()

fname = f"g{n_order}_pdef{pdef:.2f}.png"
plt.savefig(fname, dpi=300)
print(f"Figura guardada: {fname}")
