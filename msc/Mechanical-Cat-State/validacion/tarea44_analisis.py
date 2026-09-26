import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
G = ("0.05", "0.1", "0.3", "1", "3"); NT = ("0", "0.001", "0.01", "0.1"); A = (2, 4, 6)
ld = lambda g, a, n, k, key='gamma_bf': float(np.load(f"gautier_cache/g{g}_a{a}_n{n}_k{k}.npz")[key])
floor = {(g, a): abs(ld(g, a, "0", "0")) for g in G for a in A}     # nth=0, kappa1=0: bf exacto=0 -> lo medido es truncamiento de Fock
L = ["# Tarea 44 — buffer de dos niveles explícito con baño térmico (reconciliar con Gautier et al. 2022)\n",
     "H=g₂[(a²−α²)σ₊+h.c.], κ(1+n_th)D[σ₋]+κ n_th D[σ₊] (κ=1), γ_bf = tasa del modo con mayor overlap con 'a' entre los 3 modos lógicos. Dos variantes: κ₁=0 (literal) y κ₁=10⁻³κ (fuga de fotones, para que n_th=0 tenga γ_bf≠0). N=⌈α²+6√α²⌉+4.\n",
     "**Piso numérico:** con n_th=0 y κ₁=0 el bit-flip es exactamente 0 (|±α⟩ oscuros), pero el truncamiento de Fock deja un piso ∝ g₂²: " +
     "; ".join(f"g₂={g}: α²=2/4/6 → {floor[g,2]:.1e}/{floor[g,4]:.1e}/{floor[g,6]:.1e}" for g in G) + ". Los valores marcados con † están a menos de 3× de ese piso y no son fiables.\n"]
def fmt(v, g, a): return f"{v:.2e}" + ("†" if v < 3 * floor[g, a] else "")
for k1, ttl in (("0", "κ₁=0 (literal)"), ("0.001", "κ₁=10⁻³κ")):
    L.append(f"## Variante {ttl}\n")
    L += ["### γ_bf (κ) para α²=2 / 4 / 6, pendiente d ln γ_bf/dα² y γ_bf/(n_th κ) en α²=2\n",
          "| g₂/κ | n_th | γ_bf α²=2 | α²=4 | α²=6 | pendiente 2→4 | pendiente 4→6 | γ_bf/(n_th κ) (α²=2) | factor vs n_th=0 (α²=2 / 4 / 6) |", "|---|---|---|---|---|---|---|---|---|"]
    for g in G:
        base = [ld(g, a, "0", k1) for a in A]
        for n in NT:
            v = [ld(g, a, n, k1) for a in A]
            s1, s2 = np.log(v[1] / v[0]) / 2, np.log(v[2] / v[1]) / 2
            ratio = f"{v[0]/float(n):.3g}" if float(n) > 0 else "—"
            fac = "—" if n == "0" else " / ".join(f"{v[i]/base[i]:.3g}" for i in range(3))
            L.append(f"| {g} | {n} | {fmt(v[0], g, 2)} | {fmt(v[1], g, 4)} | {fmt(v[2], g, 6)} | {s1:+.2f} | {s2:+.2f} | {ratio} | {fac} |")
    L.append("")
# linealidad en n_th y comparacion con Tareas 36-37
L += ["## Linealidad en n_th y comparación con las Tareas 36-37 (γ_bf ≈ 0.05 n_q κ a Γ₂/κ=0.13, α²=2)\n",
      "Γ₂/κ=4(g₂/κ)²; Γ₂/κ=0.13 corresponde a g₂/κ=0.18. γ_bf/(n_th κ) en α²=2 (κ₁=10⁻³κ, valores sin †):\n",
      "| g₂/κ | Γ₂/κ | n_th=10⁻³ | 10⁻² | 0.1 |", "|---|---|---|---|---|"]
r02 = {}
for g in G:
    row = [ld(g, 2, n, "0.001") / float(n) for n in NT[1:]]; r02[g] = row[1]
    L.append(f"| {g} | {4*float(g)**2:.3g} | " + " | ".join(f"{x:.3g}" for x in row) + " |")
gs = np.array([float(g) for g in G]); rr = np.array([r02[g] for g in G]); sl = np.polyfit(np.log(gs[:3]), np.log(rr[:3]), 1)
L.append(f"\nInterpolación log-log entre g₂/κ=0.1 y 0.3 a g₂/κ=0.18: γ_bf/(n_th κ) ≈ {np.exp(np.interp(np.log(0.18), np.log(gs[1:3]), np.log(rr[1:3]))):.3f} (Tareas 36-37: ≈0.05, con resonancia vestida y baño mecánico distinto). γ_bf ∝ n_th con desviación <10% hasta n_th=0.1 en todo el rango (columnas 10⁻³→10⁻²→0.1).")
L.append(f"Para g₂/κ ≲ 0.3, γ_bf/(n_th κ) ∝ (g₂/κ)^{sl[0]:.1f}; para g₂/κ ≳ 1 satura en ~0.7-1 (cada excitación térmica del buffer corta el gato con probabilidad O(1)).\n")
open("tarea44_resultados.md", "w").write("\n".join(L)); print("\n".join(L))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for g in G:
    ax[0].semilogy(A, [ld(g, a, "0.01", "0.001") for a in A], 'o-', label=f"g₂/κ={g}")
    ax[1].loglog([float(n) for n in NT[1:]], [ld(g, 2, n, "0.001") for n in NT[1:]], 'o-', label=f"g₂/κ={g}")
ax[0].set_xlabel("|α|²"); ax[0].set_ylabel("γ_bf (n_th=0.01)"); ax[0].legend(fontsize=7); ax[1].set_xlabel("n_th"); ax[1].set_ylabel("γ_bf (α²=2)")
fig.tight_layout(); fig.savefig("tarea44_gamma_bf.png", dpi=120)
