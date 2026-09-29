import numpy as np, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
KS = np.logspace(np.log10(0.05), np.log10(50), 12)
R = {}
for m in "AB":
    for a2 in (2, 4, 6):
        rows = []
        for k in KS:
            d = np.load(f"tarea29bis_cache/{m}_a2{a2}_k{k:.6f}.npz")
            rows.append([float(d[x]) for x in ("gap", "gap_im", "gamma_pf", "gamma_bf")] + [bool(d['ok'])])
        R[m, a2] = np.array(rows, float)
L = ["# Tarea 29-bis (original): buffer de dos niveles (A) vs armónico Nb=6 (B); g=1, κ₁=1e-3 g\n"]
L.append("Brecha = tasa del 5º modo (Re λ asc); γ_pf, γ_bf de los 3 modos lógicos. Columna κ/4 para comparar la saturación.\n")
for a2 in (2, 4, 6):
    L += [f"### |α|²={a2}\n", "| κ/g | κ/4 | brecha A | Im A | brecha B | Im B | γ_pf A | γ_pf B | γ_bf A | γ_bf B | lógicos OK (A/B) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, k in enumerate(KS):
        A, B = R['A', a2][i], R['B', a2][i]
        L.append(f"| {k:.3f} | {k/4:.3f} | {A[0]:.4f} | {A[1]:+.2f} | {B[0]:.4f} | {B[1]:+.2f} | {A[2]:.3e} | {B[2]:.3e} | {A[3]:.3e} | {B[3]:.3e} | {bool(A[4])}/{bool(B[4])} |")
    L.append("")
L += ["## (1) Saturación de la brecha de A\n"]
for a2 in (2, 4, 6):
    A = R['A', a2]; i = np.argmax(A[:, 0])
    L.append(f"- |α|²={a2}: máx brecha A = {A[i,0]:.4f} en κ/g={KS[i]:.3f}; para κ/g=50: {A[-1,0]:.4f} (κ/4=12.5). Razón brecha(50)/máx = {A[-1,0]/A[i,0]:.2f}")
L += ["", "## (2) Escalamiento de γ_bf con |α|² (pendiente de ln γ_bf vs α², ajuste con α²=2,4,6)\n", "| κ/g | pendiente A | pendiente B |", "|---|---|---|"]
for i, k in enumerate(KS):
    sA = np.polyfit([2, 4, 6], np.log([R['A', a][i, 3] for a in (2, 4, 6)]), 1)[0]
    sB = np.polyfit([2, 4, 6], np.log([R['B', a][i, 3] for a in (2, 4, 6)]), 1)[0]
    L.append(f"| {k:.3f} | {sA:+.3f} | {sB:+.3f} |")
L += ["", "## (3) Posición del máximo de la brecha\n"]
for a2 in (2, 4, 6):
    for m in "AB":
        X = R[m, a2]; i = np.argmax(X[:, 0]); L.append(f"- {m}, |α|²={a2}: máx {X[i,0]:.4f} en κ/g={KS[i]:.3f} (κ/4={KS[i]/4:.3f})")
open("tarea29bis_resultados.md", "w").write("\n".join(L)); print("\n".join(L))
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a2, c in zip((2, 4, 6), "rgb"):
    for m, ls in zip("AB", ("-", "--")):
        for j, nm in enumerate(("gap", "gm", "pf", "bf")):
            if j in (0, 3): ax[[0, 1][j == 3]].loglog(KS, R[m, a2][:, j if j == 0 else 3], ls, color=c, label=f"{m} α²={a2}")
    ax[2].loglog(KS, R['A', a2][:, 2], color=c)
ax[0].set_title("brecha"); ax[1].set_title("γ_bf"); ax[2].set_title("γ_pf (A)"); ax[0].legend(fontsize=6)
for x in ax: x.set_xlabel("κ/g")
fig.tight_layout(); fig.savefig("tarea29bis_curvas.png", dpi=120)
