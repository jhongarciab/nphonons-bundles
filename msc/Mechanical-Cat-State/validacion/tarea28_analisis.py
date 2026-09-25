"""Seguimiento de ramas espectrales (Tarea 28) + coalescencia."""
import glob, numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

fs = sorted(glob.glob("tarea28_cache/G*.npz"), key=lambda f: float(f.split("/G")[1][:-4]))
D = [np.load(f) for f in fs]
G = np.array([float(d['Gamma2']) for d in D]); n = len(G)
LAM = np.array([d['lam'] for d in D]); OV = np.array([d['ov'] for d in D]); keys = list(D[0]['ov_keys'])
K = 10
# rama j en punto 0 = j-esimo mas lento; se sigue por asignacion minima en el plano complejo
idx = np.zeros((n, K), int); idx[0] = np.arange(K)
for i in range(1, n):
    C = np.abs(LAM[i-1][idx[i-1]][:, None] - LAM[i][None, :])
    r, c = linear_sum_assignment(C); idx[i][r] = c
BL = np.array([LAM[i][idx[i]] for i in range(n)])          # (n,K) rama x punto
BO = np.array([OV[i][idx[i]] for i in range(n)])           # (n,K,7)
L = ["# Tarea 28 — ramas espectrales (α²=2, (δ_m,Δ_q)=(0.048,0.144))\n",
     "Ramas (por continuidad); Re λ, Im λ y overlaps dominantes en cada Γ₂/κ. Rama 0 = estacionario.\n",
     "| Γ₂/κ | " + " | ".join(f"r{j} Re / Im" for j in range(1, K)) + " |", "|---|" + "---|" * (K - 1)]
for i in range(n):
    L.append(f"| {G[i]:.4f} | " + " | ".join(f"{BL[i,j].real:.3g} / {BL[i,j].imag:+.2g}" for j in range(1, K)) + " |")
# coalescencia: rama real (|Im|<tol) en i -> par complejo conjugado en i+1
tol = 1e-6
L += ["", "## Eventos real→complejo (coalescencia)\n"]
ev = []
for i in range(n - 1):
    for j in range(K):
        for k in range(j + 1, K):
            a0, b0, a1, b1 = BL[i, j], BL[i, k], BL[i+1, j], BL[i+1, k]
            if abs(a0.imag) < tol and abs(b0.imag) < tol and abs(a1.imag) > tol and abs(a1.imag + b1.imag) < 1e-3 * abs(a1.imag):
                ev.append((i, j, k))
                lab = lambda j_: ", ".join(f"{keys[m][3:]}={BO[i, j_, m]:.2f}" for m in np.argsort(-BO[i, j_])[:3])
                L.append(f"- Γ₂/κ ∈ ({G[i]:.4f}, {G[i+1]:.4f}): ramas {j},{k}. Antes Re={a0.real:.4g}, {b0.real:.4g}; "
                         f"después Re={a1.real:.4g}, Im=±{abs(a1.imag):.3g}. Overlaps antes: r{j}[{lab(j)}] r{k}[{lab(k)}]")
if not ev: L.append("- ninguno detectado con esta rejilla")
# transiciones inversas / cambios de Im en general
L += ["", "## |Im λ| por rama (K=1..9) — cambios de real a complejo y viceversa\n"]
for j in range(1, K):
    im = np.abs(BL[:, j].imag); ch = [i for i in range(n-1) if (im[i] < tol) != (im[i+1] < tol)]
    L.append(f"- rama {j}: transiciones real↔complejo entre " + (", ".join(f"({G[i]:.4f},{G[i+1]:.4f})" for i in ch) or "ninguna"))
open("tarea28_resultados.md", "w").write("\n".join(L))
print("\n".join(L))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for j in range(1, K):
    ax[0].semilogx(G, BL[:, j].real, '.-', label=f"r{j}"); ax[1].semilogx(G, BL[:, j].imag, '.-')
ax[0].set_yscale('symlog', linthresh=1e-3); ax[0].set_xlabel("Γ₂/κ"); ax[0].set_ylabel("Re λ"); ax[0].legend(ncol=3, fontsize=7)
ax[1].set_xlabel("Γ₂/κ"); ax[1].set_ylabel("Im λ"); fig.tight_layout(); fig.savefig("tarea28_ramas.png", dpi=130)
np.savez("tarea28_resultados.npz", G=G, BL=BL, BO=BO)
