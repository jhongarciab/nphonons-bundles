import numpy as np, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
THR = 0.1     # peso de borde: fisicos <~2e-2 (alpha^2=4 puebla n hasta ~16), espurios ~1
def gap_phys(d, thr=THR):
    good = [k for k in range(len(d['lam'])) if d['edge'][k] <= thr]; return d['lam'][good[4]].real if len(good) > 4 else np.nan
rows = []; L = ["# Tarea 42 — figura de mérito κ₁/κ₂ (modelo completo de Ma re-sintonizado)\n",
    "w_p=2(w−4g_x²/(3w)), d=w_p, κ=0.03, |α|²=4, Ω=|α|²G, G=2g_xg_z/w, N=22. κ₁=(tasa de paridad en la ventana P_c>0.99)/(2|α|²); κ₂=4G²/κ; brecha física = 5º modo de Floquet con peso de borde ≤0.1 (los espurios de la Tarea 38 tenían ~1; con |α|²=4 los modos físicos llegan a ~2e-2, por eso no se usa 1e-3). Predicción κ₁/κ₂=(5/72)(κ/g_z)².\n"]
def load(f): 
    d = np.load(f); return dict(tag=f.split('/')[-1][:-4], w=float(d['p_w']), gx=abs(float(d['p_gx'])), gzk=float(d['p_gz']) / 0.03, k2=float(d['p_kappa2']), rate=float(d['rate_fit']),
        rate_spec=float(d['rate_spec']), pred_rate=float(d['rate_pred']), gap=gap_phys(d), gap_old=float(d['gap_old']), Pcmax=float(d['Pc_max']), nwin=int(d['nwin']),
        te=float(d['trace_err']), he=float(d['herm']), me=float(d['mineig']), N=int(d['N']))
R = [load(f) for f in sorted(glob.glob("cache42/*.npz")) if 'N28' not in f]
def block(title, sel):
    L.extend([f"### {title}\n", "| g_x | w | g_z/κ | κ₂/κ | adiabático (κ₂/κ<0.1) | P_c máx | tasa paridad (ajuste) | tasa (espectral) | κ₁/κ₂ medido | (5/72)(κ/g_z)² | razón | brecha física | κ₁/brecha |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"])
    for r in sel:
        k1 = r['rate'] / 8; k2 = r['k2']; pred = (5 / 72) / r['gzk']**2
        L.append(f"| {r['gx']} | {r['w']:.0f} | {r['gzk']:.0f} | {k2/0.03:.3f} | {'sí' if k2/0.03 < 0.1 else 'NO'} | {r['Pcmax']:.4f} | {r['rate']:.3e} | {r['rate_spec']:.3e} | {k1/k2:.3e} | {pred:.3e} | {k1/k2/pred:.3f} | {r['gap']:.3e} | {k1/r['gap']:.3e} |")
    L.append("")
A = sorted([r for r in R if r['tag'].startswith('a_')], key=lambda r: r['gzk']); B = sorted([r for r in R if r['tag'].startswith('b_')], key=lambda r: r['gzk'])
C = sorted([r for r in R if r['tag'].startswith('c_')], key=lambda r: (r['w'], r['gx']))
block("(a) g_x=0.05 (adiabático)", A); block("(b) g_x=0.2 (κ₂/κ hasta ≳1)", B); block("(c) independencia: g_z/κ=5 fijo, g_x y w variables", C)
ratios = [r['rate'] / 8 / r['k2'] / ((5 / 72) / r['gzk']**2) for r in R if np.isfinite(r['rate'])]
L.append(f"Razón medido/predicho en los {len(ratios)} puntos con ventana: mín {min(ratios):.3f}, máx {max(ratios):.3f}, media {np.mean(ratios):.3f}.")
kg = [r['rate'] / 8 / r['gap'] for r in A + B if np.isfinite(r['gap']) and np.isfinite(r['rate'])]
L.append(f"κ₁/brecha en (a)+(b): rango {min(kg):.2e}–{max(kg):.2e}.")
# convergencia N
d22 = load("cache42/a_gx0.05_gz12.npz"); d28 = load("cache42/a_gx0.05_gz12_N28.npz")
L += ["", "### Convergencia N=22→28 (g_x=0.05, g_z/κ=12)\n", "| cantidad | N=22 | N=28 | Δrel |", "|---|---|---|---|"]
for k, nm in (("rate", "tasa de paridad"), ("rate_spec", "tasa espectral"), ("gap", "brecha física"), ("Pcmax", "P_c máx")):
    x, y = d22[k], d28[k]; L.append(f"| {nm} | {x:.5e} | {y:.5e} | {abs(y/x-1):.1e} |")
L.append(f"\nValidaciones (máximos sobre las {len(R)+1} celdas y todos los tiempos): |Tr ρ−1| ≤ {max(r['te'] for r in R):.1e}, ‖ρ−ρ†‖ ≤ {max(r['he'] for r in R):.1e}, mín autovalor ≥ {min(r['me'] for r in R):.1e} (tol 1e-10 / 1e-10 / −1e-9).")
open("tarea42_resultados.md", "w").write("\n".join(L)); print("\n".join(L))
fig, ax = plt.subplots(figsize=(6.2, 4.8)); x = np.logspace(-1.3, -0.2, 50)
for nm, S, mk in (("gx=0.05", A, 'o'), ("gx=0.2", B, 's'), ("independencia", C, '^')):
    S = [r for r in S if np.isfinite(r['rate'])]; ax.loglog([1 / r['gzk'] for r in S], [r['rate'] / 8 / r['k2'] for r in S], mk, label=nm)
ax.loglog(x, (5 / 72) * x**2, 'k-', label="(5/72)(κ/g_z)²"); ax.set_xlabel("κ/g_z"); ax.set_ylabel("κ₁/κ₂"); ax.legend(); fig.tight_layout(); fig.savefig("tarea42_kappa1_kappa2.png", dpi=120)
