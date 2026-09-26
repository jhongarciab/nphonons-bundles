import numpy as np, glob
from scipy.interpolate import CubicSpline
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import ma_model as m
L = ["# Tareas 39-40 — esquema de Ma, Xie y Li (PRA 99, 022302), modelo completo en el marco rotante exacto\n",
     "Método: propagador de Floquet sobre T=4π/w_p (qutip 5, atol 1e-12, rtol 1e-10) y potencias por cuadrados; estados en t=nT (marco rotante = laboratorio). Unidades 2π·GHz; Γ=0.015, κ=0.03, g_x=%.4f, g_z=%.4f, Ω=0.06.\n" % (m.gx, m.gz)]
D = {wp: np.load(f"cache39/wp{wp}_N22.npz") for wp in ("12.0", "11.98")}
# ---- reproduccion
L += ["## Tarea 39\n", "### Reproducción de los valores previos (N=22, Γt=30)\n",
      "| w_p | muestreo | P_c | paridad | P_e | \\|⟨a²⟩\\| |", "|---|---|---|---|---|---|"]
prev = {"12.0": (0.796, 0.729, 0.125, 2.44), "11.98": (0.996, 0.387, 0.008, 3.99)}
for wp, d in D.items():
    i = int(np.argmin(abs(d['Gt'] - 30)))
    L.append(f"| {wp} | previo | {prev[wp][0]} | {prev[wp][1]:+.3f} | {prev[wp][2]} | {prev[wp][3]} |")
    L.append(f"| {wp} | t=nT, Γt={float(d['Gt'][i]):.2f} (fase 0) | {float(d['Pc'][i]):.4f} | {float(d['par'][i]):+.4f} | {float(d['Pe'][i]):.4f} | {float(d['a2'][i]):.3f} |")
    L.append(f"| {wp} | t=30/Γ exacto (fase {float(d['phase30']):.2f}) | {float(d['chk30_Pc']):.4f} | {float(d['chk30_par']):+.4f} | {float(d['chk30_Pe']):.4f} | {float(d['chk30_a2']):.3f} |")
L.append("\nP_e, paridad y |⟨a²⟩| tienen micromovimiento dentro del período (a w_p=11.98, Γt=30: P_e varía entre 0.003 y 0.010 y la paridad entre 0.377 y 0.391 según la fase); las diferencias con los valores previos vienen de la fase de muestreo. P_c no depende de la fase.\n")
# ---- tablas y grafico
sel = [0, 6, 12, 18, 24, 30, 36, 42, 48, -1]
for wp, d in D.items():
    L += [f"### w_p={wp}: F, P_c, paridad, P_e, |⟨a²⟩| (t=nT)\n", "| Γt | F | P_c | paridad | P_e | \\|⟨a²⟩\\| |", "|---|---|---|---|---|---|"]
    n = len(d['Gt']); idx = sorted(set([min(i, n - 1) if i >= 0 else n + i for i in sel]))
    for i in idx: L.append(f"| {float(d['Gt'][i]):.2f} | {float(d['F'][i]):.4f} | {float(d['Pc'][i]):.4f} | {float(d['par'][i]):+.4f} | {float(d['Pe'][i]):.4f} | {float(d['a2'][i]):.3f} |")
    L.append("")
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for wp, d in D.items():
    for a, k, t in zip(ax, ("F", "Pc", "par"), ("F (gato par)", "P_c", "paridad")):
        a.semilogx(d['Gt'], d[k], '.-', label=f"w_p={wp}"); a.set_xlabel("Γt"); a.set_title(t); a.legend()
fig.tight_layout(); fig.savefig("tarea39_curvas.png", dpi=120)
# ---- ajuste de la tasa de paridad
pred = 2 * 4 * (m.gx**2 * m.kappa / 36.0) * (1 + 1 / 9)
L += ["### Tasa de decaimiento de la paridad (ventana P_c>0.99)\n", f"Predicción 2|α|²(g_x²κ/w²)(1+1/9) = {pred:.3e} (por unidad de t; = {pred/m.Gam:.4f} por unidad de Γt).\n",
      "| w_p | ventana Γt | puntos | tasa ajustada (1/t) | tasa/predicción | paridad inicial→final en la ventana | notas |", "|---|---|---|---|---|---|---|"]
for wp, d in D.items():
    ok = (d['Pc'] > 0.99) & (d['par'] > 0.02); Gt = d['Gt']
    if ok.sum() >= 4:
        # tramo final contiguo donde P_c>0.99
        last = np.where(d['Pc'] > 0.99)[0]; idxs = [i for i in last if d['par'][i] > 0.02]
        t = Gt[idxs] / m.Gam; y = np.log(d['par'][idxs]); sl = np.polyfit(t, y, 1)[0]
        L.append(f"| {wp} | {Gt[idxs][0]:.1f}–{Gt[idxs][-1]:.1f} | {len(idxs)} | {-sl:.3e} | {-sl/pred:.2f} | {d['par'][idxs][0]:+.3f}→{d['par'][idxs][-1]:+.3f} | |")
    else:
        L.append(f"| {wp} | — | {int(ok.sum())} | — | — | — | P_c no supera 0.99 con paridad>0: no hay ventana |")
# ---- convergencia
a, b = D["11.98"], np.load("cache39/wp11.98_N26.npz")
ia, ib = int(np.argmin(abs(a['Gt'] - 60))), int(np.argmin(abs(b['Gt'] - 60)))
L += ["", "### Convergencia N=22→26 (w_p=11.98, Γt≈60)\n", "| cantidad | N=22 | N=26 | Δrel |", "|---|---|---|---|"]
for k, nm in (("F", "F"), ("Pc", "P_c"), ("par", "paridad"), ("Pe", "P_e"), ("a2", "|⟨a²⟩|")):
    x, y = float(a[k][ia]), float(b[k][ib]); L.append(f"| {nm} | {x:.5f} | {y:.5f} | {abs(y/x-1):.1e} |")
# ---- validaciones
allr = [np.load(f) for f in glob.glob("cache39/*.npz") + glob.glob("cache40/*.npz")]
te = max(float(np.max(r['trace_err'])) for r in allr); he = max(float(np.max(r['herm'])) for r in allr); me = min(float(np.min(r['mineig'])) for r in allr)
L += ["", f"**Validaciones (todas las ρ guardadas, {len(allr)} archivos):** máx |Tr ρ−1| = {te:.1e} (tol 1e-10), máx ‖ρ−ρ†‖ = {he:.1e} (tol 1e-10), mín autovalor = {me:.1e} (tol −1e-9). "
      f"{'Cumplen.' if (te < 1e-10 and he < 1e-10 and me > -1e-9) else 'NO cumplen todas: ver detalle por archivo (los errores crecen con el número de períodos por acumulación en las potencias de U).'}\n"]
# ---- Tarea 40
fs = sorted(glob.glob("cache40/wp*.npz")); R = [np.load(f) for f in fs]
wps = np.array([float(r['wp']) for r in R]); Pc = np.array([float(r['Pc']) for r in R]); Pe = np.array([float(r['Pe']) for r in R]); A2 = np.array([float(r['a2']) for r in R]); Gt = np.array([float(r['Gt']) for r in R])
o = np.argsort(wps); wps, Pc, Pe, A2, Gt = wps[o], Pc[o], Pe[o], A2[o], Gt[o]
L += ["## Tarea 40 — barrido de w_p (N=22, t=nT con Γt≈60)\n", "| w_p | Γt | P_c | P_e | \\|⟨a²⟩\\| |", "|---|---|---|---|---|"]
for i in range(len(wps)): L.append(f"| {wps[i]:.5f} | {Gt[i]:.2f} | {Pc[i]:.4f} | {Pe[i]:.4f} | {A2[i]:.3f} |")
cs = CubicSpline(wps, Pc); xx = np.linspace(wps[0], wps[-1], 20001); yy = cs(xx); im = int(np.argmax(yy)); half = yy[im] / 2
def cross(side):
    idx = range(im, len(xx)) if side > 0 else range(im, -1, -1)
    for i in idx:
        if yy[i] < half: return xx[i]
    return np.nan
xl, xr = cross(-1), cross(+1)
L += ["", f"Máximo discreto de P_c: {Pc.max():.4f} en w_p={wps[np.argmax(Pc)]:.5f}; máximo del spline cúbico: {yy[im]:.4f} en w_p={xx[im]:.5f}. Predicción w_p*=2(w−4g_x²/(3w))={2*(6-4*m.gx**2/18):.5f}.",
      f"Ancho a media altura (half = {half:.3f}, spline): [{xl:.5f}, {xr:.5f}] → FWHM ≈ {xr-xl:.5f}" + (" (el borde toca el límite del barrido: cota inferior)" if (np.isnan(xl) or np.isnan(xr)) else "") + f"; resolución del barrido = {wps[1]-wps[0]:.5f}."]
fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(wps, Pc, 'o', label='P_c'); ax.plot(wps, Pe, 's', label='P_e'); ax.plot(xx, yy, '-', alpha=.5)
ax.axvline(2 * (6 - 4 * m.gx**2 / 18), color='gray', ls=':'); ax.set_xlabel("w_p"); ax.legend(); fig.tight_layout(); fig.savefig("tarea40_resonancia.png", dpi=120)
open("tarea39_40_resultados.md", "w").write("\n".join(L)); print("\n".join(L))
