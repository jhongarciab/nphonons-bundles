"""
Trayectorias cuánticas — Molécula excitónica (2QDs + Förster)
=============================================================
Adaptación del script de trayectorias (1QD, análogo Bin2020 Fig.4)
al sistema de 2QDs acoplados vía Förster con baño fonónico común.

Hamiltoniano (marco rotante, 2QD):
  H = ω_b b†b
      + Δ·(n̂_e1 + n̂_e2)
      + λ·(n̂_e1 + n̂_e2)·(b + b†)
      + Ω·(σ₁⁺ + σ₁⁻ + σ₂⁺ + σ₂⁻)
      + J·(σ₁⁺σ₂⁻ + σ₂⁺σ₁⁻)

  Espacio: QD₁ ⊗ QD₂ ⊗ Fock(Nph)  →  dim = 4·Nph

Operadores de Lindblad:
  L₁ = √κ  · b        (pérdida fonónica — click de cavidad)
  L₂ = √γ  · σ₁⁻      (emisión espontánea QD₁)
  L₃ = √γ  · σ₂⁻      (emisión espontánea QD₂)
  L₄ = √γφ · n̂_e1     (dephasing puro QD₁)
  L₅ = √γφ · n̂_e2     (dephasing puro QD₂)

Estados electrónicos relevantes (base producto QD₁ ⊗ QD₂):
  |vv⟩ = |g⟩₁|g⟩₂           ambos en base
  |Ψ₊⟩ = (|cv⟩ + |vc⟩)/√2   estado brillante (Dicke)
  |Ψ₋⟩ = (|cv⟩ - |vc⟩)/√2   estado oscuro

Resonancia Stokes 2QD:
  Régimen I:   Δ_n = −n·ω_b − J
  Régimen III: Δ_n = −√(n²ω_b² − 8Ω²) − J
    (el factor 8Ω² = 2·(2Ω²)·2 viene del factor √2 superradiante)

Sweet spot (máxima pureza):
  κ ≈ 10·Ω_eff^(n),   con  Ω_eff^(n) = √2·Ω·(λ/ω_b)^n/√(n!)

Parámetros de producción (n=3, énfasis Régimen III):
  Ω/ω_b = 0.8,  λ/ω_b = 0.1,  κ/ω_b = 0.0002

Autor: Jhon S. García B. — Tesis UQ 2025
Referencia: Bin et al. PRL 124, 053601 (2020), Fig. 4
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, qeye, destroy, tensor, ket2dm, mcsolve

# =============================================================================
# PARÁMETROS FÍSICOS
# =============================================================================
omegab    = 1.0
n_bundle  = 3        # orden del paquete a visualizar en la trayectoria

# Parámetros 2QD — barrido de trayectorias para n=3 (Régimen III)
# Referencia sweet spot (estimador): κ ≈ 10·Ω_eff^(n)
# con Ω_eff^(n)=√2·Ω·(λ/ω_b)^n/√(n!)
lambda_omegab = 0.1
Omega_omegab  = 0.2
kappa         = 0.002
gamma         = 0.0002
gamma_phi     = 0.0004
J             = 0.5    # acoplamiento Förster

lam = lambda_omegab * omegab
Om  = Omega_omegab  * omegab

# -----------------------------------------------------------------------------
# Resonancia Stokes 2QD
# -----------------------------------------------------------------------------
# Régimen I:   Δ_n = −n·ω_b − J
# Régimen III: Δ_n = −√(n²ω_b² − 8Ω²) − J
# Con Ω=0.8 y n=3: arg = 9 − 8·0.64 = 3.88 > 0  → Régimen III aplicable
arg_III = (n_bundle * omegab)**2 - 8.0 * Om**2

if arg_III > 0:
    Delta_III = -np.sqrt(arg_III) - J
else:
    Delta_III = -n_bundle * omegab - J   # fallback Régimen I

Delta_I = -n_bundle * omegab - J         # Régimen I siempre disponible

# Usamos Régimen III (más preciso para Ω=0.2, no despreciable)
Delta = Delta_III

# Ω_eff^(n) — Régimen I (estimado del sweet spot)
import math
Omega_eff_n = np.sqrt(2) * Om * (lam / omegab)**n_bundle / math.sqrt(math.factorial(n_bundle))
kappa_sweet = 10.0 * Omega_eff_n

print("=== PARÁMETROS 2QD TRAYECTORIA ===")
print(f"  n_bundle     = {n_bundle}")
print(f"  λ/ω_b        = {lambda_omegab}")
print(f"  Ω/ω_b        = {Omega_omegab}")
print(f"  κ/ω_b        = {kappa}")
print(f"  γ/ω_b        = {gamma}   (γ/κ = {gamma/kappa:.4f}  ← debe ser ≪ 1)")
print(f"  γφ/ω_b       = {gamma_phi}")
print(f"  J/ω_b        = {J}")
print(f"  Δ Régimen I  = {Delta_I:.6f}")
print(f"  Δ Régimen III= {Delta_III:.6f}  ← usado")
print(f"  Ω_eff^({n_bundle})   = {Omega_eff_n:.4e}")
print(f"  κ_sweet      = {kappa_sweet:.4e}  (κ/κ_sweet = {kappa/kappa_sweet:.2f})")

# =============================================================================
# CONSTRUCCIÓN DE OPERADORES — QD₁ ⊗ QD₂ ⊗ Fock(Nph)
# =============================================================================
Nph = 14   # truncamiento Fock; Nph > n_bundle + margen holgado

# --- TLS: |e⟩ = |0⟩,  |g⟩ = |1⟩  (consistente con forster_2qds.py) ---
ket_e = basis(2, 0)   # |e⟩ excitado
ket_g = basis(2, 1)   # |g⟩ base

Pe = ket2dm(ket_e)    # |e⟩⟨e|
Pg = ket2dm(ket_g)    # |g⟩⟨g|

sm_tls = ket_g * ket_e.dag()   # σ⁻ = |g⟩⟨e|
sp_tls = sm_tls.dag()           # σ⁺ = |e⟩⟨g|

b  = destroy(Nph)
nb = b.dag() * b
Ib = qeye(Nph)
Iq = qeye(2)

# Operadores en el espacio compuesto QD₁ ⊗ QD₂ ⊗ Fock
b_sys  = tensor(Iq, Iq, b)
nb_sys = tensor(Iq, Iq, nb)

sm1 = tensor(sm_tls, Iq, Ib)   # σ₁⁻
sp1 = sm1.dag()                  # σ₁⁺
sm2 = tensor(Iq, sm_tls, Ib)   # σ₂⁻
sp2 = sm2.dag()                  # σ₂⁺

pe1 = sp1 * sm1    # n̂_e1 = |e⟩⟨e|₁  (proyector excitado QD₁)
pe2 = sp2 * sm2    # n̂_e2 = |e⟩⟨e|₂  (proyector excitado QD₂)
ne_total = pe1 + pe2

# =============================================================================
# HAMILTONIANO 2QD
# =============================================================================
H = (
    omegab * nb_sys                            # energía fonónica
    + Delta  * ne_total                        # detuning QD₁ + QD₂
    + lam    * ne_total * (b_sys + b_sys.dag())# acoplamiento e-fonón
    + Om     * (sm1 + sp1 + sm2 + sp2)         # driving láser
    + J      * (sp1 * sm2 + sp2 * sm1)         # acoplamiento Förster
)

# =============================================================================
# OPERADORES DE COLAPSO
# =============================================================================
c_ops = [
    np.sqrt(kappa)     * b_sys,   # L₁: pérdida fonónica (clicks de cavidad)
    np.sqrt(gamma)     * sm1,     # L₂: emisión espontánea QD₁
    np.sqrt(gamma)     * sm2,     # L₃: emisión espontánea QD₂
    np.sqrt(gamma_phi) * pe1,     # L₄: dephasing puro QD₁
    np.sqrt(gamma_phi) * pe2,     # L₅: dephasing puro QD₂
]
idx_cav = 0   # índice del operador de pérdida fonónica

# =============================================================================
# PROYECTORES POBLACIONALES
#
# Notación:
#   P_{m, vv}  = Prob( m fonones, ambos QDs en base )
#   P_{m, Ψ+}  = Prob( m fonones, estado brillante Dicke )
#
# Se construyen como tensores  ρ_elec ⊗ |m⟩⟨m|  para ser usados como e_ops
# en mcsolve (devuelven ⟨P⟩(t) a lo largo de cada trayectoria).
# =============================================================================
vv_dm   = tensor(Pg, Pg)   # |vv⟩⟨vv| en espacio QD₁⊗QD₂

# Estado brillante |Ψ₊⟩ = (|ev⟩ + |ve⟩)/√2
Psi_plus_ket = (tensor(ket_e, ket_g) + tensor(ket_g, ket_e)) / np.sqrt(2)
bright_dm    = ket2dm(Psi_plus_ket)   # |Ψ₊⟩⟨Ψ₊|

def make_proj(m, elec_dm):
    """Proyector |m⟩⟨m|_fon ⊗ elec_dm en el espacio completo."""
    return tensor(elec_dm, ket2dm(basis(Nph, m)))

# Proyectores para m = 0, 1, 2  (+ m=3 si n_bundle=3)
P0_vv  = make_proj(0, vv_dm)
P0_br  = make_proj(0, bright_dm)
P1_vv  = make_proj(1, vv_dm)
P1_br  = make_proj(1, bright_dm)
P2_vv  = make_proj(2, vv_dm)
P2_br  = make_proj(2, bright_dm)
P3_vv  = make_proj(3, vv_dm)
P3_br  = make_proj(3, bright_dm)

e_ops = [P0_vv, P0_br, P1_vv, P1_br, P2_vv, P2_br, P3_vv, P3_br]
# índices:   0      1      2      3      4      5      6      7

# =============================================================================
# ESTADO INICIAL: vacío fonónico, ambos QDs en base
# psi0 = |g⟩₁ ⊗ |g⟩₂ ⊗ |0⟩_fon
# =============================================================================
psi0 = tensor(ket_g, ket_g, basis(Nph, 0))

# =============================================================================
# TIEMPO DE SIMULACIÓN
# =============================================================================
# Escala temporal natural: τ_cav = 1/κ = 500 ω_b⁻¹
# Mostramos ~40 ciclos de κ⁻¹ para ver varios eventos de emisión como Bin Fig.4
x_min, x_max = 0.0, 60000.0
Nt = 20001

tlist = np.linspace(x_min / omegab, x_max / omegab, Nt)
x     = omegab * tlist

ntraj = 25
opts  = {
    "keep_runs_results": True,
    "store_states":      False,
    "progress_bar":      "text",
    "nsteps":            200000,
    "improved_sampling": True,
}

print(f"\n  Iniciando mcsolve: {ntraj} trayectorias, {Nt} puntos, "
      f"dim(H)={4*Nph}×{4*Nph}")
res = mcsolve(H, psi0, tlist, c_ops, e_ops=e_ops, ntraj=ntraj, options=opts)

# =============================================================================
# CONTEO Y CLASIFICACIÓN DE CLICKS DE CAVIDAD
# =============================================================================
nj = np.zeros(ntraj, dtype=int)
all_click_times = []

for k in range(ntraj):
    ct = np.array(res.col_times[k] if res.col_times[k] is not None else [])
    cw = np.array(res.col_which[k] if res.col_which[k] is not None else [])
    cav_times = ct[cw == idx_cav]
    nj[k] = len(cav_times)
    all_click_times.append(cav_times)

# Trayectoria con más clicks (más representativa de la cascada)
k_show = int(np.argmax(nj))
print(f"\n  k_show={k_show},  clicks_cavidad={int(nj[k_show])},  "
      f"media={np.mean(nj):.2f}")

# -----------------------------------------------------------------------------
# Clasificación por ventana temporal (análogo inset Fig.4 Bin)
# Agrupa clicks separados por menos de τ_bundle = 3/κ
# -----------------------------------------------------------------------------
tau_bundle = 3.0 / kappa

def classify_click_train(times, tau):
    """
    Clasifica clicks en grupos temporales.
    Devuelve (singles, n_bundle-plets, 2n-plets).
    """
    if len(times) == 0:
        return 0, 0, 0
    groups  = []
    current = [times[0]]
    for t in times[1:]:
        if (t - current[-1]) <= tau:
            current.append(t)
        else:
            groups.append(current)
            current = [t]
    groups.append(current)

    singles  = sum(1 for g in groups if len(g) == 1)
    nbundles = sum(1 for g in groups if len(g) == n_bundle)
    twoN     = sum(1 for g in groups if len(g) >= 2 * n_bundle)
    return singles, nbundles, twoN

single_total = nbundle_total = twoN_total = 0
for times in all_click_times:
    s, nb_, tn = classify_click_train(times, tau_bundle)
    single_total  += s
    nbundle_total += nb_
    twoN_total    += tn

# =============================================================================
# SEÑALES DE LA TRAYECTORIA DESTACADA
# =============================================================================
P0vv_t = res.expect[0][k_show]
P0br_t = res.expect[1][k_show]
P1vv_t = res.expect[2][k_show]
P1br_t = res.expect[3][k_show]
P2vv_t = res.expect[4][k_show]
P2br_t = res.expect[5][k_show]
P3vv_t = res.expect[6][k_show]
P3br_t = res.expect[7][k_show]

ct_k    = np.array(res.col_times[k_show] if res.col_times[k_show] is not None else [])
cw_k    = np.array(res.col_which[k_show] if res.col_which[k_show] is not None else [])
clicks  = omegab * ct_k[cw_k == idx_cav]   # en unidades de ω_b·t

# =============================================================================
# FIGURA — 3 paneles (a)(b)(c), análogo Fig.4 para n=3 (2QD)
# =============================================================================
eps = 1e-12

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
fig.subplots_adjust(hspace=0.06)

# --- Panel (a): estado vacío m=0 ---
axes[0].plot(x, P0vv_t, lw=1.2, color="black",   label=r"$P_{0,vv}$")
axes[0].plot(x, P0br_t, lw=1.2, color="#1a6b3c", label=r"$P_{0,\Psi_+}$")
axes[0].set_ylim(0, 1)
axes[0].set_yticks([0, 1])
axes[0].set_ylabel("Probabilidad", fontsize=13)
axes[0].legend(loc="upper right", fontsize=11)
axes[0].text(0.01, 0.92, r"$(a)$", transform=axes[0].transAxes, fontsize=14)

# Inset de barras — total de 25 trayectorias (análogo inset Fig.4 Bin)
ax_in = axes[0].inset_axes([0.62, 0.18, 0.35, 0.68])
labels_bar = ["single", f"{n_bundle}-fonón", f"2n (n={n_bundle})"]
vals_bar   = [single_total, nbundle_total, twoN_total]
colors_bar = ["#4c78a8", "#1a6b3c", "#c4942a"]
ax_in.bar(np.arange(3), vals_bar, color=colors_bar, alpha=0.9)
ax_in.set_xticks(np.arange(3))
ax_in.set_xticklabels(labels_bar, rotation=20, fontsize=8)
ax_in.set_title(f"clicks (25 traj.)", fontsize=9)
ax_in.tick_params(axis="y", labelsize=8)

# --- Panel (b): estado m=1 (escala log) ---
axes[1].semilogy(x, P1br_t + eps, lw=1.2, color="#1a6b3c", label=r"$P_{1,\Psi_+}$")
axes[1].semilogy(x, P1vv_t + eps, lw=1.2, color="black",   label=r"$P_{1,vv}$")
axes[1].set_ylim(1e-10, 1e0)
axes[1].set_ylabel("Prob. (log)", fontsize=13)
axes[1].legend(loc="upper right", fontsize=11)
axes[1].text(0.01, 0.90, r"$(b)$", transform=axes[1].transAxes, fontsize=14)

# --- Panel (c): estado m=3 + líneas de clicks (cascada triple) ---
axes[2].semilogy(x, P3br_t + eps, lw=1.2, color="#1a6b3c", label=r"$P_{3,\Psi_+}$")
axes[2].semilogy(x, P3vv_t + eps, lw=1.2, color="black",   label=r"$P_{3,vv}$")
for xc in clicks:
    axes[2].axvline(xc, lw=0.6, alpha=0.25, color="red")
axes[2].set_ylim(1e-10, 1e0)
axes[2].set_xlabel(r"$\omega_b\,t$", fontsize=14)
axes[2].set_ylabel("Prob. (log)", fontsize=13)
axes[2].legend(loc="upper right", fontsize=11)
axes[2].text(0.01, 0.90, r"$(c)$", transform=axes[2].transAxes, fontsize=14)

for ax in axes:
    ax.tick_params(labelsize=12)

fig.suptitle(
    (rf"Trayectorias 2QD  ($n={n_bundle}$, Reg. III) — "
     rf"$\lambda/\omega_b={lambda_omegab}$, $\Omega/\omega_b={Omega_omegab}$, "
     rf"$\kappa/\omega_b={kappa}$, $\Delta/\omega_b={Delta:.3f}$, $J/\omega_b={J}$"),
    fontsize=10, y=0.998
)

plt.tight_layout()
plt.savefig("trayectorias_2qds.pdf", bbox_inches="tight")
# Para PGF (LaTeX): descomentar las dos líneas siguientes y comentar plt.show()
# import matplotlib
# matplotlib.use("pgf")  # debe llamarse ANTES de importar pyplot
# plt.savefig("trayectorias_2qds.pgf")
plt.show()

# =============================================================================
# RESUMEN EN CONSOLA
# =============================================================================
print(f"\n{'='*60}")
print(f"  RESUMEN TRAYECTORIA — n={n_bundle}, 2QD con Förster")
print(f"{'='*60}")
print(f"  λ/ω_b   = {lambda_omegab}     Ω/ω_b = {Omega_omegab}")
print(f"  κ/ω_b   = {kappa}    γ/ω_b = {gamma}   J/ω_b = {J}")
print(f"  Δ/ω_b   = {Delta:.6f}  (Régimen III + J)")
print(f"  Ω_eff^({n_bundle}) = {Omega_eff_n:.4e},  κ/Ω_eff = {kappa/Omega_eff_n:.1f}x")
print(f"  Clasificación de clicks (25 trayectorias):")
print(f"    singles      = {single_total}")
print(f"    {n_bundle}-fonón      = {nbundle_total}   ← proceso dominante esperado")
print(f"    2n (n={n_bundle})    = {twoN_total}   ← debe ser ≪ {n_bundle}-fonón")
print(f"  Trayectoria mostrada: k={k_show}, "
      f"clicks cavidad={int(nj[k_show])}")