import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# =====================================================
# TRAYECTORIAS CUÁNTICAS 2QDs: RÉGIMEN n=3 FONONES
# J = 0.5ωb (ACOPLAMIENTO FÖRSTER FUERTE)
# Detuning en resonancia: Δ = -3.5ωb
# =====================================================

print("=" * 70)
print("  EMISIÓN DE BUNDLES DE 3-FONONES: 2 QDs + FÖRSTER FUERTE")
print("=" * 70)

# =====================================================
# PARÁMETROS FÍSICOS
# =====================================================

omegab = 1.0
n_target = 3  # Régimen de 3-fonones

lambda_omegab = 0.3   # = 0.24 (¡más fuerte!)
Omega_omegab = 0.2   # = 0.03
kappa = 0.003  # Mantener igual (propiedad de la cavidad)
gamma = 0.0004  # Mantener igual (propiedad del QD)
gamma_phi = 0.0004

lam = lambda_omegab * omegab
Om = Omega_omegab * omegab

# FÖRSTER FUERTE (el régimen que exploraste originalmente)
J_omegab = 0.5  # ← CORRECCIÓN CLAVE
J = J_omegab * omegab

# DETUNING EN RESONANCIA DE 3-FONONES
# De tu g^(3): dip con antibunching en Δ/ωb = -3.5
# Física: Δ_resonancia = -n·ωb - J = -3.0 - 0.5 = -3.5 ✓
Delta_over_omegab = -3.5  # ← EXACTAMENTE EL DIP DE g^(3)
Delta = Delta_over_omegab * omegab

# Frecuencia de Rabi efectiva de 3-fonones
# Con estado brillante: λ_eff = √2·λ
lambda_eff = np.sqrt(2) * lam
Omega_eff_3_bright = (lambda_eff ** 3 * Om) / (np.abs(Delta) ** 3)

# Sweet spot check
sweet_spot_ratio = kappa / Omega_eff_3_bright

print(f"\n{'PARÁMETROS FÍSICOS':^70}")
print("-" * 70)
print(f"  Régimen:           n = {n_target} fonones")
print(f"  λ/ωb:              {lambda_omegab}")
print(f"  Ω/ωb:              {Omega_omegab}")
print(f"  J/ωb:              {J_omegab}  ← ACOPLAMIENTO FUERTE")
print(f"  Splitting 2J/ωb:   {2 * J_omegab}  (comparable a ωb!)")
print(f"  κ/ωb:              {kappa}  (Q = {1 / kappa:.0f})")
print(f"  γ/ωb:              {gamma}")
print(f"  γφ/ωb:             {gamma_phi}")
print(f"\n  Δ/ωb:              {Delta_over_omegab}  ← RESONANCIA (dip en g³)")
print(f"  Fórmula:           Δ = -n·ωb - J = -3.0 - 0.5 = -3.5 ✓")
print(f"\n  λ_eff (brillante): {lambda_eff / omegab:.3f} ωb  (√2 × λ)")
print(f"  Ω_eff^(3):         {Omega_eff_3_bright:.6f} ωb")
print(f"  κ/Ω_eff^(3):       {sweet_spot_ratio:.1f}  (óptimo: ~10)")
print(f"\n  1/κ:               {1 / kappa:.0f} ωb⁻¹")
print(f"  Bundle 3-fonones:  ~{3 / kappa:.0f} ωb⁻¹  (3 clicks)")
print("=" * 70)

# =====================================================
# NOTA IMPORTANTE SOBRE J = 0.5ωb
# =====================================================
print(f"\n{'RÉGIMEN DE ACOPLAMIENTO FÖRSTER FUERTE':^70}")
print("-" * 70)
print("  Con J = 0.5ωb (mitad de la energía del fonón):")
print("  • Estados |S⟩ y |A⟩ separados por 2J = 1.0 ωb")
print("  • Molécula excitónica genuina (no perturbación)")
print("  • Estado |S⟩ shifted +J = +0.5ωb en energía")
print("  • Estado |A⟩ shifted -J = -0.5ωb en energía")
print("  • Resonancia n=3 desde |S⟩: Δ = -3ωb - J = -3.5ωb")
print("-" * 70)

# =====================================================
# ESPACIO DE HILBERT
# =====================================================

Nph = 12  # Truncamiento fonónico
ntraj = 100
x_min, x_max = 0.0, 25000.0
Nt = 10001

tlist = np.linspace(x_min / omegab, x_max / omegab, Nt)
x = omegab * tlist

print(f"\nDimensión: {Nph} × 2 × 2 = {Nph * 4}")
print(f"Trayectorias: {ntraj}")
print(f"Ventana: {x_max:.0f} ωb⁻¹")

# Estados TLS
ket_e = basis(2, 0)
ket_g = basis(2, 1)

Pe_1qd = ket2dm(ket_e)
Pg_1qd = ket2dm(ket_g)

sm_1qd = ket_g * ket_e.dag()
sp_1qd = sm_1qd.dag()

# Operadores base
b = destroy(Nph)
nb = b.dag() * b
Ib = qeye(Nph)
Is = qeye(2)

# =====================================================
# OPERADORES EN ESPACIO TOTAL
# =====================================================

B = tensor(b, Is, Is)
NB = tensor(nb, Is, Is)

SM1 = tensor(Ib, sm_1qd, Is)
SP1 = tensor(Ib, sp_1qd, Is)
PE1 = tensor(Ib, Pe_1qd, Is)

SM2 = tensor(Ib, Is, sm_1qd)
SP2 = tensor(Ib, Is, sp_1qd)
PE2 = tensor(Ib, Is, Pe_1qd)

# =====================================================
# HAMILTONIANO
# =====================================================

H = (
        omegab * NB
        + Delta * (PE1 + PE2)
        + lam * (PE1 + PE2) * (B + B.dag())
        + Om * (SM1 + SP1 + SM2 + SP2)
        + J * (SP1 * SM2 + SM1 * SP2)  # J = 0.5ωb (FUERTE)
)

print("\nHamiltoniano construido con J = 0.5ωb ✓")

# =====================================================
# OPERADORES DE COLAPSO COLECTIVOS
# =====================================================

SM_bright = (SM1 + SM2).unit()
PE_total = PE1 + PE2

c_ops = [
    np.sqrt(kappa) * B,
    np.sqrt(gamma) * SM_bright,
    np.sqrt(gamma_phi / 2) * PE_total,
]

IDX_CAV = 0

print("Operadores colectivos (preservan simetría) ✓")

psi0 = tensor(basis(Nph, 0), ket_g, ket_g)

# =====================================================
# PROYECTORES EN BASE COLECTIVA
# =====================================================

projectors = {}
obs_keys = []

for nph in range(6):  # 0,1,2,3,4,5 fonones
    ket_n = basis(Nph, nph)

    # |n,gg⟩
    key_gg = f"P{nph}_gg"
    projectors[key_gg] = ket2dm(tensor(ket_n, ket_g, ket_g))
    obs_keys.append(key_gg)

    # |n,S⟩ (brillante)
    key_S = f"P{nph}_S"
    ket_S = (tensor(ket_n, ket_e, ket_g) + tensor(ket_n, ket_g, ket_e)).unit()
    projectors[key_S] = ket2dm(ket_S)
    obs_keys.append(key_S)

    # |n,A⟩ (oscuro)
    key_A = f"P{nph}_A"
    ket_A = (tensor(ket_n, ket_e, ket_g) - tensor(ket_n, ket_g, ket_e)).unit()
    projectors[key_A] = ket2dm(ket_A)
    obs_keys.append(key_A)

    # |n,ee⟩
    key_ee = f"P{nph}_ee"
    projectors[key_ee] = ket2dm(tensor(ket_n, ket_e, ket_e))
    obs_keys.append(key_ee)

e_ops = [projectors[k] for k in obs_keys]

print(f"Proyectores: {len(e_ops)} observables (n=0-5)")

# =====================================================
# SIMULACIÓN
# =====================================================

opts = {
    "keep_runs_results": True,
    "store_states": False,
    "progress_bar": "text",
    "nsteps": 200000,
    "improved_sampling": True,
}

print(f"\n{'CORRIENDO SIMULACIÓN':^70}")
print("-" * 70)
res = mcsolve(H, psi0, tlist, c_ops, e_ops=e_ops, ntraj=ntraj, options=opts)

# =====================================================
# ANÁLISIS
# =====================================================

nj_cav = np.zeros(ntraj, dtype=int)
for k in range(ntraj):
    ct = np.array(res.col_times[k] if res.col_times[k] is not None else [])
    cw = np.array(res.col_which[k] if res.col_which[k] is not None else [])
    nj_cav[k] = np.count_nonzero(cw == IDX_CAV)

k_show = int(np.argmax(nj_cav))

print("\n" + "=" * 70)
print(f"{'RESULTADOS':^70}")
print("=" * 70)
print(f"  Trayectoria:     k = {k_show}")
print(f"  Clicks:          {int(nj_cav[k_show])}")
print(f"  Promedio:        {np.mean(nj_cav):.2f} clicks/trayectoria")
print(f"  Máximo:          {int(np.max(nj_cav))}")
print(f"  Tasa:            {np.mean(nj_cav) / x_max:.6f} clicks/ωb⁻¹")
print("=" * 70)

obs = {}
for i, key in enumerate(obs_keys):
    obs[key] = res.expect[i][k_show]

ct_show = np.array(res.col_times[k_show] if res.col_times[k_show] is not None else [])
cw_show = np.array(res.col_which[k_show] if res.col_which[k_show] is not None else [])
clicks = omegab * ct_show[cw_show == IDX_CAV]

# Verificación simetría
P_A_total = sum(obs[f"P{n}_A"] for n in range(6))
P_S_total = sum(obs[f"P{n}_S"] for n in range(6))

print(f"\nVerificación simetría:")
print(f"  P_oscuro_max:     {np.max(P_A_total):.8f}  (debe ser ~0)")
print(f"  P_brillante_max:  {np.max(P_S_total):.4f}")

# =====================================================
# GRÁFICAS
# =====================================================

eps = 1e-12

fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

# Panel (a): n=0
axes[0].plot(x, obs["P0_gg"], lw=1.8, label=r"$P_{0,gg}$", color="C0")
axes[0].plot(x, obs["P0_S"], lw=1.8, label=r"$P_{0,S}$ (brillante)", color="C1")
axes[0].plot(x, obs["P0_A"], lw=0.8, label=r"$P_{0,A}$ (oscuro)",
             color="red", ls="--", alpha=0.4)
axes[0].set_ylim(-0.02, 1.02)
axes[0].set_ylabel("Probabilidad", fontsize=11)
axes[0].legend(loc="upper right", fontsize=9)
axes[0].grid(True, alpha=0.25)
axes[0].set_title(
    f"n=3 FONONES: J/ωb={J_omegab}, Δ/ωb={Delta_over_omegab} (resonancia) "
    f"— Traj k={k_show} ({int(nj_cav[k_show])} clicks)",
    fontsize=11, fontweight='bold', pad=10
)

# Panel (b): n=1
axes[1].semilogy(x, obs["P1_S"] + eps, lw=1.8,
                 label=r"$P_{1,S}$ (brillante)", color="C1")
axes[1].semilogy(x, obs["P1_gg"] + eps, lw=1.2,
                 label=r"$P_{1,gg}$", color="C0", alpha=0.7)
axes[1].set_ylim(1e-10, 1e0)
axes[1].set_ylabel("Prob. (log)", fontsize=11)
axes[1].legend(loc="upper right", fontsize=9)
axes[1].grid(True, which="both", alpha=0.25)

# Panel (c): n=2
axes[2].semilogy(x, obs["P2_S"] + eps, lw=1.8,
                 label=r"$P_{2,S}$ (brillante)", color="C1")
axes[2].semilogy(x, obs["P2_gg"] + eps, lw=1.2,
                 label=r"$P_{2,gg}$", color="C0", alpha=0.7)
axes[2].set_ylim(1e-10, 1e0)
axes[2].set_ylabel("Prob. (log)", fontsize=11)
axes[2].legend(loc="upper right", fontsize=9)
axes[2].grid(True, which="both", alpha=0.25)

# Panel (d): n=3 ← CLAVE
axes[3].semilogy(x, obs["P3_S"] + eps, lw=2.2,
                 label=r"$\mathbf{P_{3,S}}$ (brillante) ← ESTADO CLAVE",
                 color="C1")
axes[3].semilogy(x, obs["P3_gg"] + eps, lw=1.2,
                 label=r"$P_{3,gg}$", color="C0", alpha=0.7)
for xc in clicks:
    axes[3].axvline(xc, lw=0.6, alpha=0.25, color="red")
axes[3].set_ylim(1e-10, 1e0)
axes[3].set_xlabel(r"$\omega_b t$", fontsize=12)
axes[3].set_ylabel("Prob. (log)", fontsize=11)
axes[3].legend(loc="upper right", fontsize=9)
axes[3].grid(True, which="both", alpha=0.25)

fig.tight_layout()
#plt.savefig("traj_2QD_n3_J05_resonancia.png", dpi=200)
#print(f"\n✓ Guardado: traj_2QD_n3_J05_resonancia.png")

plt.show()

print("\n" + "=" * 70)
print("  SIMULACIÓN COMPLETADA")
print(f"  Régimen: n=3 fonones, J=0.5ωb, Δ=-3.5ωb (resonancia exacta)")
print("=" * 70)