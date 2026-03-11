import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, qeye, destroy, tensor, ket2dm, mcsolve

# -----------------------------------------------------------------------------
# Parámetros (Bin2020 Fig. 4)
# -----------------------------------------------------------------------------
omegab = 1.0
n_bundle = 2

lambda_omegab = 0.1
Omega_omegab = 0.2
kappa = 0.002
gamma = 0.0002
gamma_phi = 0.0004

lam = lambda_omegab * omegab
Om = Omega_omegab * omegab

# -----------------------------------------------------------------------------
# Resonancia / sweet-point (n=2)
# -----------------------------------------------------------------------------
# Opción A (dressed): Delta_n = -sqrt((n*wb)^2 - 4*Omega^2)
Delta_dressed = -np.sqrt((n_bundle * omegab) ** 2 - 4 * Om ** 2)

# Opción B (reg. II simplificada 1QD): Delta_n = (lambda^2/wb) - n*wb
Delta_regII = (lam ** 2 / omegab) - n_bundle * omegab

# Se deja por defecto la dressed (alineada con el bloque de trayectorias previo)
Delta = Delta_dressed

# Condición cualitativa de sweet point reportada: kappa ~ 10 * Omega_eff^(2)
Omega_eff2_est = Om * np.exp(-0.5 * (lam / omegab) ** 2) * (lam / omegab) ** 2 / np.sqrt(2)
print(f"Delta_dressed = {Delta_dressed:.6f}")
print(f"Delta_regII   = {Delta_regII:.6f}")
print(f"Omega_eff^(2) est = {Omega_eff2_est:.6e}")
print(f"kappa / (10*Omega_eff2) = {kappa/(10*Omega_eff2_est):.3f}")

# -----------------------------------------------------------------------------
# Simulación MC (25 trayectorias como en Fig. 4)
# -----------------------------------------------------------------------------
Nph = 14
ntraj = 25
x_min, x_max = 0.0, 20000.0
Nt = 20001

tlist = np.linspace(x_min / omegab, x_max / omegab, Nt)
x = omegab * tlist

# -----------------------------------------------------------------------------
# Base TLS consistente con script previo:
# |e> = |0>, |g> = |1>
# -----------------------------------------------------------------------------
ket_e = basis(2, 0)
ket_g = basis(2, 1)

Pe = ket2dm(ket_e)
Pg = ket2dm(ket_g)

sm_tls = ket_g * ket_e.dag()  # |g><e|
sp_tls = sm_tls.dag()         # |e><g|

b = destroy(Nph)
Ib = qeye(Nph)
Is = qeye(2)
nb = b.dag() * b

# Proyectores poblacionales (hasta 2 fonones para visualizar cascada de 2)
P0g = tensor(ket2dm(basis(Nph, 0)), Pg)
P0e = tensor(ket2dm(basis(Nph, 0)), Pe)
P1g = tensor(ket2dm(basis(Nph, 1)), Pg)
P1e = tensor(ket2dm(basis(Nph, 1)), Pe)
P2g = tensor(ket2dm(basis(Nph, 2)), Pg)
P2e = tensor(ket2dm(basis(Nph, 2)), Pe)

H = (
    omegab * tensor(nb, Is)
    + Delta * tensor(Ib, Pe)
    + lam * tensor(b + b.dag(), Pe)
    + Om * tensor(Ib, sp_tls + sm_tls)
)

c_ops = [
    np.sqrt(kappa) * tensor(b, Is),
    np.sqrt(gamma) * tensor(Ib, sm_tls),
    np.sqrt(gamma_phi) * tensor(Ib, Pe),
]
idx_cav = 0

psi0 = tensor(basis(Nph, 0), ket_g)
e_ops = [P0g, P0e, P1g, P1e, P2g, P2e]

opts = {
    "keep_runs_results": True,
    "store_states": False,
    "progress_bar": "text",
    "nsteps": 200000,
    "improved_sampling": True,
}

res = mcsolve(H, psi0, tlist, c_ops, e_ops=e_ops, ntraj=ntraj, options=opts)

# -----------------------------------------------------------------------------
# Conteo de clicks por trayectoria
# -----------------------------------------------------------------------------
nj = np.zeros(ntraj, dtype=int)
all_click_times = []
for k in range(ntraj):
    ct = np.array(res.col_times[k] if res.col_times[k] is not None else [])
    cw = np.array(res.col_which[k] if res.col_which[k] is not None else [])
    cav_times = ct[cw == idx_cav]
    nj[k] = len(cav_times)
    all_click_times.append(cav_times)

k_show = int(np.argmax(nj))
print(f"k_show={k_show}, cavity_jumps={int(nj[k_show])}, mean_jumps={np.mean(nj):.3f}")

# -----------------------------------------------------------------------------
# Clasificación simple de eventos (single / 2-phonon / 2n-phonon)
# -----------------------------------------------------------------------------
# Heurística por ventana temporal de agrupamiento de clicks
# (aproximación operativa para comparar cualitativamente con Fig. 4)
tau_bundle = 3.0 / kappa

def classify_click_train(times, tau):
    if len(times) == 0:
        return 0, 0, 0
    groups = []
    current = [times[0]]
    for t in times[1:]:
        if (t - current[-1]) <= tau:
            current.append(t)
        else:
            groups.append(current)
            current = [t]
    groups.append(current)

    singles = sum(1 for g in groups if len(g) == 1)
    pairs = sum(1 for g in groups if len(g) == 2)
    n2 = sum(1 for g in groups if len(g) >= 4)  # proxy para 2n con n=2
    return singles, pairs, n2

single_total = 0
pair_total = 0
n2_total = 0
for times in all_click_times:
    s, p, n2 = classify_click_train(times, tau_bundle)
    single_total += s
    pair_total += p
    n2_total += n2

# -----------------------------------------------------------------------------
# Señales de la trayectoria destacada
# -----------------------------------------------------------------------------
P0g_t = res.expect[0][k_show]
P0e_t = res.expect[1][k_show]
P1g_t = res.expect[2][k_show]
P1e_t = res.expect[3][k_show]
P2g_t = res.expect[4][k_show]
P2e_t = res.expect[5][k_show]

ct = np.array(res.col_times[k_show] if res.col_times[k_show] is not None else [])
cw = np.array(res.col_which[k_show] if res.col_which[k_show] is not None else [])
clicks = omegab * ct[cw == idx_cav]

# -----------------------------------------------------------------------------
# Figura (a)-(c): fracción de trayectoria + barras tipo inset Fig.4
# -----------------------------------------------------------------------------
eps = 1e-12
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

axes[0].plot(x, P0g_t, lw=1.2, label=r"$P_{0g}$")
axes[0].plot(x, P0e_t, lw=1.2, label=r"$P_{0e}$")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Probabilidad")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.25)
axes[0].text(0.01, 0.92, r"$(a)$", transform=axes[0].transAxes, fontsize=14)

axes[1].semilogy(x, P1e_t + eps, lw=1.2, label=r"$P_{1e}$")
axes[1].semilogy(x, P1g_t + eps, lw=1.2, label=r"$P_{1g}$")
axes[1].set_ylim(1e-10, 1e0)
axes[1].set_ylabel("Prob. (log)")
axes[1].legend(loc="upper right")
axes[1].grid(True, which="both", alpha=0.25)
axes[1].text(0.01, 0.90, r"$(b)$", transform=axes[1].transAxes, fontsize=14)

axes[2].semilogy(x, P2e_t + eps, lw=1.2, label=r"$P_{2e}$")
axes[2].semilogy(x, P2g_t + eps, lw=1.2, label=r"$P_{2g}$")
for xc in clicks:
    axes[2].axvline(xc, lw=0.6, alpha=0.25)
axes[2].set_ylim(1e-10, 1e0)
axes[2].set_xlabel(r"$\omega_b t$")
axes[2].set_ylabel("Prob. (log)")
axes[2].legend(loc="upper right")
axes[2].grid(True, which="both", alpha=0.25)
axes[2].text(0.01, 0.90, r"$(c)$", transform=axes[2].transAxes, fontsize=14)

# Inset de barras (acumulado 25 trayectorias)
ax_in = axes[0].inset_axes([0.62, 0.18, 0.35, 0.68])
labels = ["single", "2-phonon", "2n (n=2)"]
vals = [single_total, pair_total, n2_total]
colors = ["#4c78a8", "#1a6b3c", "#c4942a"]
xpos = np.arange(len(labels))
ax_in.bar(xpos, vals, color=colors, alpha=0.9)
ax_in.set_xticks(xpos)
ax_in.set_xticklabels(labels, rotation=20, fontsize=8)
ax_in.set_title("clicks (25 traj)", fontsize=9)
ax_in.tick_params(axis="y", labelsize=8)

fig.suptitle(
    f"Trayectorias noJ ajustadas a Fig.4 (k={k_show}, jumps={int(nj[k_show])})",
    y=0.995,
)
plt.tight_layout()
plt.show()
# plt.savefig("./figs/oficial/trayectorias_jorge.pdf", bbox_inches="tight")
# plt.savefig("./figs/oficial/pgf/trayectorias_jorge.pgf")
# plt.close()
