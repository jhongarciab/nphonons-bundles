import numpy as np, glob, os
from scipy.interpolate import PchipInterpolator as CubicSpline
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = ["\n# Tarea 45(b)–(e)\n"]
def gp(d, thr=0.5):
    good = [k for k in range(len(d['lam'])) if d['edge'][k] <= thr]; return d['lam'][good[4]].real if len(good) > 4 else np.nan
# ================= (d) =================
L += ["## (d) Origen de la caída de P_c (Tarea 42(a), g_x=0.05, w=6, N=22)\n",
      "'full' = Tarea 42; 'nocounter' = sin los términos contrarrotantes (3ω); 'nogz_pair' = sin g_z σ_z a pero con el intercambio de pares efectivo G(σ₊a²+h.c.), G=2g_xg_z/w; 'nocounter_nogz_pair' = ambos.\n",
      "| g_z/κ | variante | P_c máx | tasa de paridad (ajuste) | κ₁/κ₂ (medido) | previsto (5/72)(κ/g_z)² | brecha física (peso≤0.5) | P_e final |", "|---|---|---|---|---|---|---|---|"]
rows = []
for gzk in (5, 12):
    items = [("full", np.load(f"cache42/a_gx0.05_gz{gzk}.npz"))]
    for v in ("nocounter", "nogz_pair", "nocounter_nogz_pair"):
        f = f"cache45d/{v}_gz{gzk}.npz"
        if os.path.exists(f): items.append((v, np.load(f)))
    for v, d in items:
        k2 = float(d['p_kappa2']); rate = float(d['rate_fit']); pred = (5 / 72) / gzk**2
        L.append(f"| {gzk} | {v} | {float(d['Pc_max']):.4f} | {rate:.3e} | {rate/8/k2:.3e} | {pred:.3e} | {gp(d):.3e} | — |")
        rows.append((gzk, v, float(d['Pc_max'])))
base = {g: [r[2] for r in rows if r[0] == g and r[1] == 'full'][0] for g in (5, 12)}
L.append(f"\nBase (full): P_c máx = {base[5]:.4f} (g_z/κ=5), {base[12]:.4f} (g_z/κ=12). Un valor cercano a 1 tras apagar un término indica que ese término causa la caída.\n")
# ================= (e) =================
L += ["## (e) Ancho de la resonancia vestida frente a κ₂/κ (g_z=0.2121 fijo, g_x variable, d=12 fijo como en la Tarea 40, κ₂t=120)\n",
      "w_p=w_p*(g_x)+sκ₂ con w_p*=2(w−4g_x²/(3w)). P_c a t=120/κ₂ (misma escala κ₂t que la Tarea 40: Γt=60 en κ₂/κ=1).\n"]
fig, ax = plt.subplots(figsize=(6, 4.2)); res = []
for r2 in ("0.03", "0.1", "0.3", "1"):
    fs = sorted(glob.glob(f"cache45e/r{r2}_s*.npz"), key=lambda f: float(np.load(f)['s'])); 
    if len(fs) < 8: continue
    R = [np.load(f) for f in fs]; s = np.array([float(r['s']) for r in R]); Pc = np.array([float(r['Pc']) for r in R]); Pe = np.array([float(r['Pe']) for r in R]); k2 = float(R[0]['kappa2']); gx = float(R[0]['gx'])
    cs = CubicSpline(s, Pc); xx = np.linspace(s[0], s[-1], 20001); yy = cs(xx); im = int(np.argmax(yy)); half = yy[im] / 2
    def cross(sign):
        rng = range(im, len(xx)) if sign > 0 else range(im, -1, -1)
        for i in rng:
            if yy[i] < half: return xx[i]
        return np.nan
    xl, xr = cross(-1), cross(1); fw = (xr - xl) if np.isfinite(xl) and np.isfinite(xr) else np.nan
    res.append((float(r2), k2, gx, xx[im], yy[im], xl, xr, fw))
    L += [f"### κ₂/κ={r2} (κ₂={k2:.3e}, g_x={gx:.4f})\n", "| s=(w_p−w_p*)/κ₂ | w_p−w_p* | P_c | P_e |", "|---|---|---|---|"]
    for i in range(len(s)): L.append(f"| {s[i]:+.2f} | {s[i]*k2:+.3e} | {Pc[i]:.4f} | {Pe[i]:.4f} |")
    L.append(f"\nPico (interpolación PCHIP monótona): P_c={yy[im]:.4f} en s={xx[im]:+.3f}; medias alturas s∈[{xl:+.3f}, {xr:+.3f}] ⇒ FWHM={fw:.3f} κ₂={fw*k2:.3e} (={fw*k2/0.03:.3e} κ).\n")
    ax.plot(s, Pc, 'o-', label=f"κ₂/κ={r2}")
ax.set_xlabel("(w_p−w_p*)/κ₂"); ax.set_ylabel("P_c"); ax.legend(); fig.tight_layout(); fig.savefig("tarea45e_resonancias.png", dpi=120)
if res:
    L += ["### FWHM y ley de escala\n", "| κ₂/κ | κ₂ | FWHM (unid. κ₂) | FWHM (unid. κ) | pico s* (unid. κ₂) | P_c pico |", "|---|---|---|---|---|---|"]
    for r in res: L.append(f"| {r[0]} | {r[1]:.3e} | {r[7]:.3f} | {r[7]*r[1]/0.03:.3e} | {r[3]:+.3f} | {r[4]:.4f} |")
    ok = [r for r in res if np.isfinite(r[7])]
    if len(ok) >= 3:
        x = np.log([r[1] for r in ok]); y = np.log([r[7] * r[1] for r in ok]); p = np.polyfit(x, y, 1)
        L.append(f"\nAjuste FWHM = A·κ₂^p (en unidades absolutas): p={p[0]:.3f}, A={np.exp(p[1]):.3f} (FWHM/κ₂ ≈ {np.exp(p[1])*np.mean([r[1] for r in ok])**(p[0]-1):.2f} en el centro). FWHM/κ₂ va de {min(r[7] for r in ok):.2f} a {max(r[7] for r in ok):.2f}.")
# ================= (b),(c) =================
def load(f): return np.load(f) if os.path.exists(f) else None
L += ["\n## (b) Filtro armónico: N_f=2, 3, 4 (N=10, α²=2, sin evolución temporal)\n", "κ₁ = tasa espectral del modo de paridad/(2|α|²). 'brecha' = 5º modo (todos) y con peso de borde ≤0.5 (N=10 es muy pequeño: los pesos de borde no son fiables; se comparan entre N_f a igual N).\n",
      "| κ_f | N_f | κ₁ (espectral) | κ₁ respecto a N_f=2 | 5º modo Re | brecha (peso≤0.5) | brecha respecto a N_f=2 |", "|---|---|---|---|---|---|---|"]
for kf in ("0.3", "1"):
    b = load(f"cache45b/kf{kf}_N10_Nf2.npz")
    for nf in (2, 3, 4):
        d = load(f"cache45b/kf{kf}_N10_Nf{nf}.npz")
        if d is None or b is None: continue
        k1 = float(d['rate_spec']) / 4; k1b = float(b['rate_spec']) / 4
        L.append(f"| {kf} | {nf} | {k1:.4e} | {k1/k1b:.5f} | {float(d['gap_old']):.4e} | {gp(d):.4e} | {gp(d)/gp(b):.4f} |")
L += ["\n## (c) Convergencia de la Tarea 43: κ_f=0.3, N=16 frente a N=20 (N_f=2)\n", "| N | κ₁ (espectral) | 5º modo Re | brecha (peso≤0.5) | brecha (criterio adoptado: peso>0.5 o inestable 0.3%) |", "|---|---|---|---|---|"]
d16, d20 = load("cache43/kf0.3_N16.npz"), load("cache45b/kf0.3_N20_Nf2.npz")
def adopted(d, dn):
    lam = d['lam']; lamn = dn['lam']; bad = d['edge'] > 0.5
    for k, x in enumerate(lam):
        j = int(np.argmin(np.abs(lamn - x))); y = lamn[j]; bad[k] |= not (abs(y.real - x.real) < 3e-3 * abs(x.real) + 1e-6 and abs(y.imag - x.imag) < 1e-2 * abs(x.imag) + 0.01)
    good = [k for k in range(len(lam)) if not bad[k]]; return lam[good[4]].real if len(good) > 4 else np.nan
if d16 is not None and d20 is not None:
    for N, d in ((16, d16), (20, d20)):
        L.append(f"| {N} | {float(d['rate_spec'])/4:.4e} | {float(d['gap_old']):.4e} | {gp(d):.4e} | {adopted(d, d20 if N == 16 else d16):.4e} |")
    L.append(f"\nΔrel κ₁ (N=16→20): {abs(float(d20['rate_spec'])/float(d16['rate_spec'])-1):.1e}; Δrel brecha (peso≤0.5): {abs(gp(d20)/gp(d16)-1):.1e}.")
open("tarea45_resultados_bcde.md", "w").write("\n".join(L)); print("\n".join(L))
