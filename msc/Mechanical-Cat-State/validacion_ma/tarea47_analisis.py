import numpy as np, glob, os
from scipy.interpolate import PchipInterpolator
L = ["# Tarea 47 — cierres finales\n"]
# ================= (a) =================
L += ["## (a) Peso de los estados iniciales sobre la rama interior (autovectores izquierdos, α²=2, resonancia vestida)\n",
      "Fracción del estado inicial (relativa al peso total sobre modos no-código, como en la Tarea 35) sobre la rama real interior (modo real, 0.03<Re<0.2, overlap con n>1.5). Irrelevante si <1e-4.\n",
      "| Γ₂/κ | N | Re de la rama | |1.3α⟩|g⟩ | |4⟩|g⟩ | |α⟩|e⟩ | ¿irrelevante (<1e-4)? |", "|---|---|---|---|---|---|---|"]
for G in ("0.5", "1.0", "3.0"):
    for N in (26, 32):
        f = f"validacion/tarea47a_cache/G{G}_N{N}.npz"
        if not os.path.exists(f): continue
        d = np.load(f); lam, ov = d['lam'], d['ov']
        ks = [k for k in range(4, 40) if abs(lam[k].imag) < 1e-3 and 0.03 < lam[k].real < 0.2 and ov[k][1] > 1.5]
        if not ks: L.append(f"| {G} | {N} | (no hallada entre los 40 más lentos) | | | | |"); continue
        k = ks[0]; w = [float(d[n + '_w'][k]) for n in ("coh", "fock", "exc")]
        L.append(f"| {G} | {N} | {lam[k].real:.4f} | {w[0]:.1e} | {w[1]:.1e} | {w[2]:.1e} | {'SÍ' if max(w) < 1e-4 else 'NO (máx %.1e)' % max(w)} |")
L.append("")
# ================= (c) =================
L += ["## (c) 45(d) 'nocounter' re-sintonizado con δ_osc=−g_x²/w (w_p=2(w−g_x²/w), d=w_p)\n",
      "Predicción: quitar los términos de 3ω elimina el factor (1+1/9) ⇒ κ₁ baja a 1/(10/9)=0.900 de 'full'.\n",
      "| g_z/κ | variante | P_c máx | tasa de paridad (ajuste) | tasa (espectral) | κ₁ respecto a full (ajuste / espectral) | κ₁/κ₂ |", "|---|---|---|---|---|---|---|"]
for gzk in (5, 12):
    full = np.load(f"validacion_ma/cache42/a_gx0.05_gz{gzk}.npz"); items = [("full", full)]
    for v, f in (("nocounter (sin re-sintonizar, 45d)", f"validacion_ma/cache45d/nocounter_gz{gzk}.npz"), ("nocounter re-sintonizado", f"validacion_ma/cache47c/nocounter_retuned_gz{gzk}.npz")):
        if os.path.exists(f): items.append((v, np.load(f)))
    for v, d in items:
        k2 = float(d['p_kappa2']); rf, rs = float(d['rate_fit']), float(d['rate_spec'])
        L.append(f"| {gzk} | {v} | {float(d['Pc_max']):.5f} | {rf:.3e} | {rs:.3e} | {rf/float(full['rate_fit']):.3f} / {rs/float(full['rate_spec']):.3f} | {rf/8/k2:.3e} |")
L.append("")
# ================= (d) =================
L += ["## (d) Ancho de la resonancia: FWHM = c·G (G=2g_xg_z/w=√(κκ₂)/2), 7 valores de κ₂/κ\n",
      "FWHM medio-altura de P_c(w_p) a κ₂t=120 (PCHIP monótona; d=12 fijo). G en unidades absolutas (κ=0.03).\n",
      "| κ₂/κ | κ₂ | G | FWHM | FWHM/κ₂ | c=FWHM/G | pico P_c |", "|---|---|---|---|---|---|---|"]
rows = []
for r2 in ("0.03", "0.05", "0.1", "0.2", "0.3", "0.5", "1"):
    fs = sorted(glob.glob(f"validacion_ma/cache45e/r{r2}_s*.npz"), key=lambda f: float(np.load(f)['s']))
    if len(fs) < 10: continue
    R = [np.load(f) for f in fs]; s = np.array([float(r['s']) for r in R]); Pc = np.array([float(r['Pc']) for r in R]); k2 = float(R[0]['kappa2'])
    cs = PchipInterpolator(s, Pc); xx = np.linspace(s[0], s[-1], 40001); yy = cs(xx); im = int(np.argmax(yy)); half = yy[im] / 2
    def cross(sign):
        rng = range(im, len(xx)) if sign > 0 else range(im, -1, -1)
        for i in rng:
            if yy[i] < half: return xx[i]
        return np.nan
    xl, xr = cross(-1), cross(1); fw = (xr - xl) * k2; G = np.sqrt(0.03 * k2) / 2
    if np.isfinite(fw): rows.append((float(r2), k2, G, fw, fw / G))
    L.append(f"| {r2} | {k2:.3e} | {G:.3e} | {fw:.3e} | {fw/k2:.3f} | {fw/G:.3f} | {yy[im]:.4f} |")
if rows:
    c = np.array([r[4] for r in rows]); x = np.log([r[1] for r in rows]); y = np.log([r[3] for r in rows]); p = np.polyfit(x, y, 1)
    A = np.array([[np.log(r[2]), 1] for r in rows]); pG = np.polyfit(np.log([r[2] for r in rows]), np.log([r[3] for r in rows]), 1)
    L.append(f"\n**c = FWHM/G: media {c.mean():.3f}, desviación estándar {c.std(ddof=1):.3f} ({c.std(ddof=1)/c.mean()*100:.1f}%), rango [{c.min():.3f}, {c.max():.3f}] (n={len(c)}).** Ajuste libre FWHM∝κ₂^{p[0]:.3f} (=G^{pG[0]:.3f}); ajuste con pendiente fija 1 en G (FWHM=cG por mínimos cuadrados en log): c={np.exp(np.mean(np.log([r[3] for r in rows]) - np.log([r[2] for r in rows]))):.3f}.")
    L.append("Tendencia de c con κ₂/κ: " + ", ".join(f"{r[0]}→{r[4]:.2f}" for r in rows) + ".")
open("validacion_ma/tarea47_resultados.md", "w").write("\n".join(L)); print("\n".join(L))
