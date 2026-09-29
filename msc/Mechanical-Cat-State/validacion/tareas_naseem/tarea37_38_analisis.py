import numpy as np, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = ["# Tareas 37-38\n"]
# ============ Tarea 37 ============
FS = (0.1, 0.2, 0.5, 0.75, 1, 1.5, 2); TS = (10, 20, 50)
L += ["## Tarea 37 — baños térmicos consistentes (Γ₂/κ=0.13, (δ_m,Δ_q)=(0.048,0.144); qubit κ[(n_q+1)D[σ₋]+n_qD[σ₊]], mecánico γ_m[(n_m+1)D[a]+n_mD[a†]])\n",
      "n_q=1/(e^{h·2f/kT}−1), n_m=1/(e^{h f/kT}−1). Propagador (1e-15,1e-13).\n"]
bad = []
for a2 in (2, 3):
    L += [f"### |α|²={a2}\n", "| f_m (GHz) | T (mK) | n_q | n_m | γ_pf | γ_bf | η | P(código) | \\|⟨a²⟩\\| | herm |", "|---|---|---|---|---|---|---|---|---|---|"]
    for f in FS:
        for T in TS:
            d = np.load(f"tarea37_cache/a2{a2}_f{f}_T{T}.npz")
            L.append(f"| {f} | {T} | {float(d['nq']):.2e} | {float(d['nm']):.2e} | {float(d['gamma_pf']):.3e} | {float(d['gamma_bf']):.3e} | {float(d['eta']):.3g} | {float(d['p_code']):.4f} | {float(d['a2']):.4f} | {float(d['herm']):.0e} |")
            if not (float(d['trace_err']) < 1e-10 and float(d['herm']) < 1e-10 and float(d['min_eig']) > -1e-10): bad.append((a2, f, T, float(d['herm']), float(d['min_eig'])))
    L.append("")
L += ["### Verificación del modo bit-flip (|α⟩|g⟩ propagado por U^n, ajuste de ⟨sgn x⟩(t))\n", "| α² | f (GHz) | T (mK) | n_q | n_m | γ_bf espectral | tasa ajustada | dif. rel. | puntos del ajuste |", "|---|---|---|---|---|---|---|---|---|"]
for f in sorted(glob.glob("tarea37_cache/chk_*.npz")):
    d = np.load(f); gs, rf = float(d['gamma_bf_spectral']), float(d['rate_fit'])
    L.append(f"| {float(d['alpha2']):.0f} | {float(d['f'])} | {float(d['T']):.0f} | {float(d['nq']):.2e} | {float(d['nm']):.2e} | {gs:.4e} | {rf:.4e} | {abs(rf/gs-1):.1e} | {int(d['n_fit'])} |")
L.append("")
for a2, Np in ((2, 26), (3, 30)):
    r0 = np.load(f"tarea37_cache/a2{a2}_f0.1_T50.npz"); r1 = np.load(f"tarea37_cache/a2_{a2}_hot_Nplus.npz")
    ch = lambda k: abs(float(r1[k]) / float(r0[k]) - 1)
    L.append(f"- Convergencia N {int(r0['N'])}→{int(r1['N'])} en el punto más caliente (0.1 GHz, 50 mK, α²={a2}): γ_pf {ch('gamma_pf'):.1e}, γ_bf {ch('gamma_bf'):.1e}, η {ch('eta'):.1e}, P(código) {ch('p_code'):.1e}, |⟨a²⟩| {ch('a2'):.1e}.")
L.append(f"\nValidaciones de ρ (traza/herm/autovalores, tol 1e-10): {'TODAS OK' if not bad else str(len(bad)) + ' puntos FALLAN en hermiticidad (antes de hermitizar) o autovalores: ' + '; '.join(f'(α²={a},f={f},T={t}: herm={h:.0e}, mineig={m:.0e})' for a,f,t,h,m in bad[:12])}\n")
# mapa eta
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, a2 in enumerate((2, 3)):
    M = np.array([[float(np.load(f"tarea37_cache/a2{a2}_f{f}_T{T}.npz")['eta']) for f in FS] for T in TS])
    im = ax[k].imshow(np.log10(np.clip(M, 1e-3, None)), origin='lower', aspect='auto'); ax[k].set_xticks(range(len(FS))); ax[k].set_xticklabels(FS)
    ax[k].set_yticks(range(3)); ax[k].set_yticklabels(TS); ax[k].set_xlabel("f_m (GHz)"); ax[k].set_ylabel("T (mK)"); ax[k].set_title(f"log10 η, α²={a2}"); plt.colorbar(im, ax=ax[k])
fig.tight_layout(); fig.savefig("tarea37_eta_mapa.png", dpi=120)
# ============ Tarea 38 ============
def load(tag, G, N): return np.load(f"audit_cache/{tag}_G{G}_N{N}.npz")
def flags(d20, d26):
    w = d20['w_edge'] > 1e-3
    lam20, lam26 = d20['lam'], d26['lam']; unst = np.zeros(len(lam20), bool)
    for k, x in enumerate(lam20):
        dist = np.abs(lam26 - x); j = int(np.argmin(dist))
        unst[k] = not (abs(lam26[j].real - x.real) < 0.02 * abs(x.real) + 1e-4 and abs(lam26[j].imag - x.imag) < 0.02 * abs(x.imag) + 0.05)
    return w, unst
def classes(d): return np.argmax(d['ov'], axis=1)      # 0=P,1=n,2=a
L += ["\n## Tarea 38 — auditoría de la brecha (criterio: peso de borde >1e-3 en n>N−6, o modo sin pareja estable en N→N+6)\n"]
# ---- T23 / T21 : "confinamiento" = menor Re entre modos de clase n
gl = sorted(glob.glob("audit_cache/base_G*_N20.npz"), key=lambda f: float(f.split("_G")[1].split("_N")[0]))
L += ["### Tareas 23 y 21 (marco base δ_m=Δ_q=0, α²=2): brecha de confinamiento = menor Re entre modos clasificados 'n'\n",
      "| Γ₂/κ | brecha original (N=20) | brecha sin modos de borde/inestables | ¿modo original de borde? | Im original | brecha (N=26, original) |", "|---|---|---|---|---|---|"]
T23 = []
for f in gl:
    G = f.split("_G")[1].split("_N")[0]; d20, d26 = np.load(f), load('base', G, 26)
    w, un = flags(d20, d26); cl = classes(d20); lam = d20['lam']; ok = np.arange(len(lam)) >= 1
    cand = [k for k in range(1, len(lam)) if cl[k] == 1]; k0 = min(cand, key=lambda k: lam[k].real)
    cand2 = [k for k in cand if not w[k] and not un[k]]; k1 = min(cand2, key=lambda k: lam[k].real) if cand2 else None
    cl26 = classes(d26); c26 = [k for k in range(1, len(d26['lam'])) if cl26[k] == 1]; g26 = min(d26['lam'][c26].real) if c26 else np.nan
    T23.append((float(G), lam[k0].real, lam[k1].real if k1 is not None else np.nan))
    L.append(f"| {float(G):.4f} | {lam[k0].real:.5f} | {lam[k1].real if k1 is not None else float('nan'):.5f} | {'SÍ' if (w[k0] or un[k0]) else 'no'} | {lam[k0].imag:+.2f} | {g26:.5f} |")
T23 = np.array(T23); ii = np.argmax(T23[:, 1]); jj = int(np.nanargmax(T23[:, 2]))
L.append(f"\nMáx original: {T23[ii,1]:.4f} en Γ₂/κ={T23[ii,0]:.3f}; máx tras excluir borde/inestables: {T23[jj,2]:.4f} en Γ₂/κ={T23[jj,0]:.3f}; valor en Γ₂/κ=3: original {T23[-1,1]:.4f}, filtrado {T23[-1,2]:.4f}.\n")
# ---- T27 / T28
def gaps(tag, Gs):
    out = []
    for G in Gs:
        d20, d26 = load(tag, G, 20), load(tag, G, 26); w, un = flags(d20, d26); lam = d20['lam']
        good = [k for k in range(len(lam)) if not w[k] and not un[k]]
        # 5o modo por Re entre los no espurios (los 4 primeros son el codigo)
        g_old = lam[4].real; g_ph = lam[good[4]].real if len(good) > 4 else np.nan
        out.append((float(G), g_old, lam[4].imag, g_ph, lam[good[4]].imag if len(good) > 4 else np.nan, bool(w[4]), bool(un[4]), d26['lam'][4].real))
    return np.array(out)
G27 = ["0.01", "0.03", "0.07", "0.13", "0.24", "0.45", "0.84", "1.6", "3.0"]; A = gaps('res', G27)
L += ["### Tarea 27 (resonancia vestida, α²=2): brecha robusta (5º modo por Re)\n", "| Γ₂/κ | original N=20 | Im | ¿5º modo es de borde? | brecha física (excluyendo borde) | Im | original N=26 |", "|---|---|---|---|---|---|---|"]
for r in A: L.append(f"| {r[0]} | {r[1]:.4f} | {r[2]:+.2f} | {'SÍ' if (r[5] or r[6]) else 'no'} | {r[3]:.4f} | {r[4]:+.2f} | {r[7]:.4f} |")
i0 = int(np.argmax(A[:, 1])); i1 = int(np.nanargmax(A[:, 3]))
L.append(f"\nMáx original {A[i0,1]:.4f} en Γ₂/κ={A[i0,0]:.2f}; máx física {A[i1,3]:.4f} en Γ₂/κ={A[i1,0]:.2f}; valor en Γ₂/κ=3.0: original {A[-1,1]:.4f}, física {A[-1,3]:.4f}; monótona (física): {bool(np.all(np.diff(A[:,3])>-1e-3))}.\n")
G28 = sorted({f.split('_G')[1].split('_N')[0] for f in glob.glob("audit_cache/res_G*_N20.npz") if f.split('_G')[1].split('_N')[0] not in G27 or True}, key=float)
G28 = [g for g in G28 if len(g) == 8]                       # los 30 puntos de la Tarea 28 (formato .6f)
B = gaps('res', G28)
L += ["### Tarea 28 (30 puntos Γ₂/κ∈[0.02,1], α²=2)\n", "| Γ₂/κ | 5º modo original | Im | ¿borde/inestable? | brecha física | Im |", "|---|---|---|---|---|---|"]
for r in B: L.append(f"| {r[0]:.4f} | {r[1]:.4f} | {r[2]:+.2f} | {'SÍ' if (r[5] or r[6]) else 'no'} | {r[3]:.4f} | {r[4]:+.2f} |")
nb = int(sum(1 for r in B if (r[5] or r[6]))); L.append(f"\nPuntos donde el 5º modo original es de borde/inestable: {nb} de {len(B)}; primer punto: Γ₂/κ={next((r[0] for r in B if (r[5] or r[6])), float('nan')):.4f}.")
L.append(f"Máx original {B[:,1].max():.4f} en Γ₂/κ={B[np.argmax(B[:,1]),0]:.3f}; física: máx {np.nanmax(B[:,3]):.4f} en Γ₂/κ={B[np.nanargmax(B[:,3]),0]:.3f}; física en Γ₂/κ=1: {B[-1,3]:.4f}; monótona (física): {bool(np.all(np.diff(B[:,3])>-2e-3))}.")
# cuantos modos de borde en la ventana de 12 de la Tarea 28
ne = []
for G in G28:
    d = load('res', G, 20); w = d['w_edge'][:12] > 1e-3; ne.append(int(w.sum()))
L.append(f"Modos de borde dentro de los 12 más lentos por punto: mín {min(ne)}, máx {max(ne)}. (La 'pareja Im≈7-18' de la Tarea 27 y los 'r4/r5' de la Tarea 28 con Im creciente son esos modos.)")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogx(T23[:, 0], T23[:, 1], 'o-', label='original'); ax[0].semilogx(T23[:, 0], T23[:, 2], 's--', label='sin borde'); ax[0].set_title("Tareas 21/23: conf. (base)"); ax[0].legend()
ax[1].semilogx(B[:, 0], B[:, 1], 'o-', label='T28 original'); ax[1].semilogx(B[:, 0], B[:, 3], 's--', label='física'); ax[1].set_title("Tareas 27/28: 5º modo"); ax[1].legend()
for a in ax: a.set_xlabel("Γ₂/κ"); a.set_ylabel("brecha")
fig.tight_layout(); fig.savefig("tarea38_auditoria.png", dpi=120)
open("tarea37_38_resultados.md", "w").write("\n".join(L)); print("\n".join(L))
