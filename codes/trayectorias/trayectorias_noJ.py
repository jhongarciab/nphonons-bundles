import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# =====================================================
# SOLO TRAYECTORIAS (corregido: base TLS consistente)
# =====================================================
omegab = 1.0
n = 2

lambda_omegab = 0.1
Omega_omegab = 0.2
kappa = 0.002
gamma = 0.0002
gamma_phi = 0.0004

lam = lambda_omegab * omegab
Om = Omega_omegab * omegab

# prueba: dressed resonance (también prueba Delta=-2.0)
Delta = -np.sqrt((n * omegab) ** 2 - 4 * Om ** 2)

Nph = 12
ntraj = 250
x_min, x_max = 0.0, 20000.0
Nt = 20001

tlist = np.linspace(x_min / omegab, x_max / omegab, Nt)
x = omegab * tlist

# -----------------------------
# TLS basis (importante)
# QuTiP: sigmam = |1><0|, así que tomamos:
# |e> = |0>, |g> = |1>
# -----------------------------
ket_e = basis(2, 0)
ket_g = basis(2, 1)

Pe = ket2dm(ket_e)
Pg = ket2dm(ket_g)

# operadores TLS explícitos (sin ambigüedad)
sm_tls = ket_g * ket_e.dag()   # |g><e|
sp_tls = sm_tls.dag()          # |e><g|

b = destroy(Nph)
Ib = qeye(Nph)
Is = qeye(2)
nb = b.dag() * b

# proyectores para poblaciones a mostrar
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
    np.sqrt(kappa) * tensor(b, Is),         # canal 0: salida fonón
    np.sqrt(gamma) * tensor(Ib, sm_tls),    # decaimiento e->g
    np.sqrt(gamma_phi) * tensor(Ib, Pe),    # dephasing paper-like
]
idx_cav = 0

# estado inicial en ground |g>
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

nj = np.zeros(ntraj, dtype=int)
for k in range(ntraj):
    ct = np.array(res.col_times[k] if res.col_times[k] is not None else [])
    cw = np.array(res.col_which[k] if res.col_which[k] is not None else [])
    nj[k] = np.count_nonzero(cw == idx_cav)

# mostramos la trayectoria más activa
k_show = int(np.argmax(nj))
print(f"k_show={k_show}, cavity_jumps={int(nj[k_show])}, mean_jumps={np.mean(nj):.3f}")

P0g_t = res.expect[0][k_show]
P0e_t = res.expect[1][k_show]
P1g_t = res.expect[2][k_show]
P1e_t = res.expect[3][k_show]
P2g_t = res.expect[4][k_show]
P2e_t = res.expect[5][k_show]

ct = np.array(res.col_times[k_show] if res.col_times[k_show] is not None else [])
cw = np.array(res.col_which[k_show] if res.col_which[k_show] is not None else [])
clicks = omegab * ct[cw == idx_cav]

# --------------------------------
# plots
# --------------------------------
eps = 1e-12
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

axes[0].plot(x, P0g_t, lw=1.2, label=r"$P_{0g}$")
axes[0].plot(x, P0e_t, lw=1.2, label=r"$P_{0e}$")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Probabilidad")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.25)

axes[1].semilogy(x, P1e_t + eps, lw=1.2, label=r"$P_{1e}$")
axes[1].semilogy(x, P1g_t + eps, lw=1.2, label=r"$P_{1g}$")
axes[1].set_ylim(1e-10, 1e0)
axes[1].set_ylabel("Prob. (log)")
axes[1].legend(loc="upper right")
axes[1].grid(True, which="both", alpha=0.25)

axes[2].semilogy(x, P2e_t + eps, lw=1.2, label=r"$P_{2e}$")
axes[2].semilogy(x, P2g_t + eps, lw=1.2, label=r"$P_{2g}$")
for xc in clicks:
    axes[2].axvline(xc, lw=0.6, alpha=0.25)
axes[2].set_ylim(1e-10, 1e0)
axes[2].set_xlabel(r"$\omega_b t$")
axes[2].set_ylabel("Prob. (log)")
axes[2].legend(loc="upper right")
axes[2].grid(True, which="both", alpha=0.25)

fig.suptitle(f"Trayectoria corregida (k={k_show}, jumps={int(nj[k_show])})", y=0.995)
plt.tight_layout()
#plt.savefig("trayectorias_solo_fix.png", dpi=180)
plt.show()


