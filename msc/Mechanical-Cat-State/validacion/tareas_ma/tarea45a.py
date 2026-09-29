"""Tarea 45(a): sensibilidad del umbral de peso de borde en la brecha fisica (Tareas 38 y 42)."""
import numpy as np, glob
THR = (1e-3, 1e-2, 3e-2, 0.1)
def gap_from(lam, edge, thr, idx=4, extra_bad=None):
    good = [k for k in range(len(lam)) if edge[k] <= thr and not (extra_bad is not None and extra_bad[k])]
    return lam[good[idx]].real if len(good) > idx else np.nan
L = ["# Tarea 45(a) — sensibilidad del umbral de borde\n"]
# ---- Tarea 38: 5o modo (T27/T28), N=20 (audit_cache/res_*), con y sin el criterio de estabilidad N->N+6
def unstable(l20, l26):
    u = np.zeros(len(l20), bool)
    for k, x in enumerate(l20):
        j = int(np.argmin(np.abs(l26 - x))); u[k] = not (abs(l26[j].real - x.real) < 0.02 * abs(x.real) + 1e-4 and abs(l26[j].imag - x.imag) < 0.02 * abs(x.imag) + 0.05)
    return u
Gs = sorted({f.split('_G')[1].split('_N')[0] for f in glob.glob("validacion/audit_cache/res_G*_N20.npz")}, key=float)
L += ["## Tarea 38 (α²=2, N=20; 5º modo por Re con resonancia vestida; 39 puntos: los 9 de la T27 y los 30 de la T28)\n", "| Γ₂/κ | sin filtro | 1e-3 | 1e-2 | 3e-2 | 0.1 | 1e-3 + estabilidad N | ¿cambia con el umbral? |", "|---|---|---|---|---|---|---|---|"]
chg = 0; rows = []
for G in Gs:
    d20 = np.load(f"validacion/audit_cache/res_G{G}_N20.npz"); d26 = np.load(f"validacion/audit_cache/res_G{G}_N26.npz")
    lam, e = d20['lam'], d20['w_edge']; u = unstable(lam, d26['lam'])
    g = [gap_from(lam, e, t) for t in THR]; gs = gap_from(lam, e, 1e-3, extra_bad=u); g0 = lam[4].real
    diff = max(g) - min(g) > 1e-6 * abs(g[-1]); chg += diff; rows.append((float(G), g0, g, gs, diff))
    if diff or float(G) in (0.01, 0.13, 3.0): L.append(f"| {float(G):.4f} | {g0:.4f} | " + " | ".join(f"{x:.4f}" for x in g) + f" | {gs:.4f} | {'SÍ' if diff else 'no'} |")
L.append(f"\nPuntos donde cambia con el umbral: {chg} de {len(Gs)} (se muestran los que cambian y tres de referencia).\n")
# ---- Tarea 38 (T21/T23): confinamiento = menor Re entre modos clase n
L += ["### T21/T23 (marco base, confinamiento = menor Re entre modos 'n')\n", "| Γ₂/κ | sin filtro | 1e-3 | 1e-2 | 3e-2 | 0.1 | ¿cambia? |", "|---|---|---|---|---|---|---|"]
chg2 = 0; nb = 0
for f in sorted(glob.glob("validacion/audit_cache/base_G*_N20.npz"), key=lambda f: float(f.split("_G")[1].split("_N")[0])):
    d = np.load(f); lam, e, cl = d['lam'], d['w_edge'], np.argmax(d['ov'], axis=1); nb += 1
    def conf(thr):
        c = [k for k in range(1, len(lam)) if cl[k] == 1 and e[k] <= thr]; return min(lam[k].real for k in c) if c else np.nan
    c0 = min(lam[k].real for k in range(1, len(lam)) if cl[k] == 1); g = [conf(t) for t in THR]
    diff = max(g) - min(g) > 1e-6 * abs(g[-1]); chg2 += diff
    L.append(f"| {float(f.split('_G')[1].split('_N')[0]):.4f} | {c0:.4f} | " + " | ".join(f"{x:.4f}" for x in g) + f" | {'SÍ' if diff else 'no'} |")
L.append(f"\nCambian {chg2} de {nb}.\n")
# ---- Tarea 42
L += ["## Tarea 42 (α²=4, N=22; 5º modo con peso de borde ≤ umbral)\n", "| celda | sin filtro | 1e-3 | 1e-2 | 3e-2 | 0.1 | ¿cambia? | pesos de borde k=4..8 |", "|---|---|---|---|---|---|---|---|"]
chg3 = 0; n3 = 0
for f in sorted(glob.glob("validacion_ma/cache42/*.npz")):
    if 'N28' in f: continue
    d = np.load(f); lam, e = d['lam'], d['edge']; g = [gap_from(lam, e, t) for t in THR]; n3 += 1
    diff = (np.any(np.isnan(g)) and not np.all(np.isnan(g))) or (np.nanmax(g) - np.nanmin(g) > 1e-6 * abs(np.nanmax(g)) if not np.all(np.isnan(g)) else False); chg3 += bool(diff)
    L.append(f"| {f.split('/')[-1][:-4]} | {lam[4].real:.3e} | " + " | ".join(f"{x:.3e}" for x in g) + f" | {'SÍ' if diff else 'no'} | {', '.join(f'{x:.3f}' for x in e[4:9])} |")
L.append(f"\nCambian {chg3} de {n3} celdas.\n")
open("validacion_ma/tarea45a_resultados.md", "w").write("\n".join(L)); print("\n".join(L))

# ================= criterio unico adoptado =================
# espurio si (i) peso de borde > 0.5 (separacion limpia: fisicos <=0.05, espurios ~1.0), o
# (ii) su autovalor se mueve mas de 0.3% en Re (o 1%+0.01 en Im) al pasar de N a N+6 (cuando hay dos truncamientos).
def unstable_tight(l20, l26):
    u = np.zeros(len(l20), bool)
    for k, x in enumerate(l20):
        j = int(np.argmin(np.abs(l26 - x))); y = l26[j]
        u[k] = not (abs(y.real - x.real) < 3e-3 * abs(x.real) + 1e-6 and abs(y.imag - x.imag) < 1e-2 * abs(x.imag) + 0.01)
    return u
L2 = ["## Criterio único adoptado\n",
      "**Espurio si (i) peso de borde > 0.5, o (ii) su autovalor cambia >0.3% en Re (>1%+0.01 en Im) al pasar de N a N+6.** Razones: el peso de borde separa limpiamente los modos de borde (≈1.0) de los físicos (≤0.05) cuando α²=4 (Tarea 42); pero en α²=2 hay una rama real (Re≈0.09 en Γ₂/κ≥0.5) con peso intermedio (0.04 a N=20, 0.014 a N=26) que un umbral de peso clasifica de forma inconsistente entre N y N+6; su Re deriva con N (0.0901→0.0896→0.0878 para N=20/26/32) y su overlap con n crece (3.16→3.51→3.78), mientras la rama física de 0.2224 no cambia en 4 cifras. Por eso el criterio (ii) es necesario cuando hay dos truncamientos; con un solo N (Tarea 42, peso bimodal) el (i) basta.\n",
      "| Γ₂/κ | 5º modo original | brecha con el criterio adoptado | (peso ≤ 0.1 solo) | ¿la rama 0.09 se excluye? |", "|---|---|---|---|---|"]
adopt = []
for G in Gs:
    d20 = np.load(f"validacion/audit_cache/res_G{G}_N20.npz"); d26 = np.load(f"validacion/audit_cache/res_G{G}_N26.npz"); lam, e = d20['lam'], d20['w_edge']
    bad = (e > 0.5) | unstable_tight(lam, d26['lam']); good = [k for k in range(60) if not bad[k]]; ga = lam[good[4]].real; adopt.append((float(G), ga))
    g01 = gap_from(lam, e, 0.1)
    if abs(ga - g01) > 1e-6 * abs(ga) or float(G) in (0.01, 0.13, 1.0, 3.0): L2.append(f"| {float(G):.4f} | {lam[4].real:.4f} | {ga:.4f} | {g01:.4f} | {'sí' if abs(ga - g01) > 1e-6 * abs(ga) else '—'} |")
ad = np.array(adopt); L2.append(f"\nCon el criterio adoptado la brecha física (39 puntos) va de {ad[0,1]:.4f} a {ad[-1,1]:.4f}; monótona: {bool(np.all(np.diff(ad[:,1]) > -2e-3))}; valor en Γ₂/κ=1: {ad[np.argmin(abs(ad[:,0]-1)),1]:.4f} (modelo estático s5: 0.2338).\n")
L2.append("**Efecto en las conclusiones de la Tarea 38:** las correcciones ya publicadas (brecha física monótona ≈0.22–0.23) se mantienen porque la Tarea 38 usaba (peso>1e-3) o (inestable a 2%): el criterio adoptado da los mismos valores (tabla arriba, columna 3 vs 'sin filtro'). Lo que la sensibilidad muestra es que el umbral de peso *solo* no es suficiente en α²=2: con 0.1 reaparece la rama de 0.09 (no convergida con N). **Queda como incertidumbre**: esa rama no es identificable como física ni como artefacto con N≤32 (deriva −2% entre N=26 y 32); si fuera física, la brecha en Γ₂/κ≳0.5 sería ≈0.09 en vez de ≈0.22.\n")
L2.append("**Efecto en la Tarea 42:** los 7 cambios con umbrales bajos (1e-3, 1e-2) son celdas con g_x=0.2 (κ₂/κ≥0.07) donde los modos físicos llegan a peso 0.006–0.016; con el criterio adoptado (y con 3e-2, 0.1) la brecha es la publicada. La brecha física de la Tarea 42 tiene ~3% de incertidumbre en N (g_z/κ=12: 2.7% entre N=22 y 28).\n")
open("validacion_ma/tarea45a_resultados.md", "a").write("\n" + "\n".join(L2)); print("\n".join(L2))
