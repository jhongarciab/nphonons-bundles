import numpy as np, glob, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
GS = np.logspace(np.log10(0.005), 0, 16)
def ld(a2, G, v, N=20): return np.load(f"ladder_cache/a2{int(a2)}_G{G:.6f}_{v}_N{N}.npz")
L = ["# Tareas 31-34 — origen del máximo de la brecha (α²=2, resonancia vestida (0.048, 0.144), κ=1)\n"]
# ---- validaciones
fails = []; nval = 0
for f in glob.glob("ladder_cache/*.npz"):
    d = np.load(f)
    if bool(d['v_unico']):
        nval += 1
        if not (d['v_trace_err'] < 1e-10 and d['v_herm'] < 1e-10 and d['v_min_eig'] > -1e-10): fails.append(f)
L.append(f"**Validaciones (estacionario único, {nval} celdas de escalera):** traza, hermiticidad, autovalores de ρ ≥ −1e-10 → "
         f"{'TODAS OK' if not fails else 'FALLAN: ' + str(fails[:5])}. Las celdas sin estacionario único (modelo mínimo: 4 modos con tasa 0) se marcan N/A.\n")
# ---- T31
L += ["## Tarea 31 — modelo mínimo con buffer de dos niveles (N=20×2)\n",
      "Fórmula exacta α=0: Δ=(κ/4)[1−√(1−8Γ₂/κ)] (Γ₂<κ/8), κ/4 si Γ₂≥κ/8.\n",
      "| Γ₂/κ | α²=0 robusta | α²=0 simple | exacta α=0 | dif rel | α²=2 robusta | α²=2 simple | dif. defs α²=2 |", "|---|---|---|---|---|---|---|---|"]
maxd0 = 0; ddef = []
for G in GS:
    d0, d2 = ld(0, G, 'min'), ld(2, G, 'min')
    ex = 0.25 * (1 - np.sqrt(1 - 8 * G)) if G < 1/8 else 0.25
    rd = abs(float(d0['gap_rob']) - ex) / ex; maxd0 = max(maxd0, rd)
    dd = abs(float(d2['gap_rob']) - float(d2['gap_simple'])) / float(d2['gap_rob']); ddef.append(dd)
    L.append(f"| {G:.4f} | {float(d0['gap_rob']):.5f} | {float(d0['gap_simple']):.5f} | {ex:.5f} | {rd:.1e} | {float(d2['gap_rob']):.5f} | {float(d2['gap_simple']):.5f} | {dd:.1e} |")
g2 = np.array([float(ld(2, G, 'min')['gap_rob']) for G in GS]); i = int(np.argmax(g2))
L.append(f"\nα²=0 vs fórmula: máx diferencia relativa {maxd0:.1e}. Definiciones robusta/simple difieren en {sum(x>1e-3 for x in ddef)} de 16 puntos (α²=2, umbral 1e-3).")
L.append(f"Mínimo α²=2: máx brecha {g2.max():.4f} en Γ₂/κ={GS[i]:.3f}; valor en Γ₂/κ=1: {g2[-1]:.4f} (κ/4=0.25) → {'monótono' if np.all(np.diff(g2) > -1e-4*g2[:-1]) else 'NO monótono'}.\n")
# ---- T32
names = [('min', '0 mínimo'), ('s1', '1 +Δ_q,δ_m'), ('s2', '2 +Lamb δ₁'), ('s3', '3 +1 fonón,γ_m'), ('s4', '4 +Kerr (resid.)'),
         ('s5', '5 +no resonantes'), ('s4L', '4L +Kerr completo'), ('s5L', '5L +no res.(Kerr compl.)')]
L += ["## Tarea 32 — escalera de ingredientes (α²=2), brecha robusta\n",
      "| Γ₂/κ | " + " | ".join(n for _, n in names) + " | efectivo(modelo_comun) |", "|---|" + "---|" * (len(names) + 1)]
import modelo_comun as mc
curves = {}
for v, _ in names: curves[v] = np.array([float(ld(2, G, v)['gap_rob']) for G in GS])
eff = []
for G in GS:
    Lq, _ = mc.effective_liouvillian(mc.gz_scale_from_Gamma2(G), 2.0, 0.144); ee, eve = Lq.eigenstates(sparse=False)
    me = mc.modos_ordenados(ee, eve, 20, False, None); eff.append(me[4]['lam'].real)
eff = np.array(eff); curves['eff'] = eff
for k, G in enumerate(GS): L.append(f"| {G:.4f} | " + " | ".join(f"{curves[v][k]:.4f}" for v, _ in names) + f" | {eff[k]:.4f} |")
def resumen(c):
    i = int(np.argmax(c)); interior = 0 < i < len(c) - 1 and c[-1] < 0.97 * c[i]
    return c.max(), GS[i], c[-1], interior
L += ["", "| paso | máx brecha | Γ₂/κ del máx | valor en Γ₂/κ=1 | ¿máx interior seguido de descenso (>3%)? |", "|---|---|---|---|---|"]
for v, n in names + [('eff', 'efectivo modelo_comun')]:
    m, gm, gl, it = resumen(curves[v]); L.append(f"| {n} | {m:.4f} | {gm:.3f} | {gl:.4f} | {'SÍ' if it else 'no'} |")
L += ["", "### Prueba inversa (a s5 se le quita UN ingrediente)\n", "| Γ₂/κ | s5 | sin Δ_q | sin (δ_m+δ₁) | sin 1-fonón | sin no-res | sin Kerr resid. |", "|---|---|---|---|---|---|---|"]
inv = {v: np.array([float(ld(2, G, v)['gap_rob']) for G in GS]) for v in ['no_dq', 'no_dmlamb', 'no_oneph', 'no_nonres', 'no_kerrp']}
for k, G in enumerate(GS): L.append(f"| {G:.4f} | {curves['s5'][k]:.4f} | " + " | ".join(f"{inv[v][k]:.4f}" for v in inv) + " |")
L += ["", "| variante | máx | Γ₂/κ del máx | valor Γ₂/κ=1 | ¿máx interior? |", "|---|---|---|---|---|"]
for v, c in [('s5', curves['s5'])] + list(inv.items()):
    m, gm, gl, it = resumen(c); L.append(f"| {v} | {m:.4f} | {gm:.3f} | {gl:.4f} | {'SÍ' if it else 'no'} |")
c0 = np.load("ladder_cache/a2%d_G%.6f_%s_N%d.npz" % (2, GS[9], 'min', 20)); c1 = np.load("ladder_cache/a2%d_G%.6f_%s_N%d.npz" % (2, GS[9], 'min', 26))
s0 = np.load("ladder_cache/a2%d_G%.6f_%s_N%d.npz" % (2, GS[9], 's5', 20)); s1 = np.load("ladder_cache/a2%d_G%.6f_%s_N%d.npz" % (2, GS[9], 's5', 26))
L.append(f"\n**Convergencia N→N+6 (Γ₂/κ={GS[9]:.4f})**: mínimo {float(c0['gap_rob']):.6f}→{float(c1['gap_rob']):.6f} (Δrel {abs(float(c1['gap_rob'])/float(c0['gap_rob'])-1):.1e}); "
         f"paso 5 {float(s0['gap_rob']):.6f}→{float(s1['gap_rob']):.6f} (Δrel {abs(float(s1['gap_rob'])/float(s0['gap_rob'])-1):.1e}).\n")
# ---- T33
L += ["## Tarea 33 — Floquet vs paso (5) vs efectivo\n", "| Γ₂/κ | Floquet (atol 1e-12, rtol 1e-10) | Im | paso 5 | paso 5 simple | efectivo modelo_comun |", "|---|---|---|---|---|---|"]
for G in ("0.03", "0.08", "0.13", "0.3", "1.0"):
    d = np.load(f"tarea33_cache/G{G}_N20.npz")
    L.append(f"| {G} | {float(d['gap_floquet']):.5f} | {float(d['im_floquet']):+.2f} | {float(d['gap_s5']):.5f} | {float(d['gap_s5_simple']):.5f} | {float(d['gap_eff']):.5f} |")
a, b = np.load("tarea33_cache/G0.13_N20.npz"), np.load("tarea33_cache/G0.13_N26.npz")
L.append(f"\nConvergencia N→N+6 (Γ₂/κ=0.13): Floquet {float(a['gap_floquet']):.6f}→{float(b['gap_floquet']):.6f} (Δrel {abs(float(b['gap_floquet'])/float(a['gap_floquet'])-1):.1e}); paso 5 {float(a['gap_s5']):.6f}→{float(b['gap_s5']):.6f}.\n")
# ---- T34
L += ["## Tarea 34 — qubit térmico (Γ₂/κ=0.13, α²=2, (atol,rtol)=(1e-14,1e-12))\n",
      "| n_q | pureza | pureza cavidad | \\|⟨a²⟩\\| (ideal 2) | P(qubit exc.) | P(código) | γ_pf | γ_bf | brecha | Im brecha | tr / herm / min λ(ρ) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for nq in ("0", "0.1", "0.3", "0.6"):
    d = np.load(f"tarea34_cache/nq{nq}_N20.npz")
    L.append(f"| {nq} | {float(d['purity']):.4f} | {float(d['purity_cav']):.4f} | {float(d['a2']):.4f} | {float(d['pexc']):.4f} | {float(d['p_code']):.4f} | {float(d['gamma_pf']):.3e} | {float(d['gamma_bf']):.3e} | {float(d['gap']):.4e} | {float(d['gap_im']):+.2f} | {float(d['trace_err']):.0e}/{float(d['herm']):.0e}/{float(d['min_eig']):.0e} |")
a, b = np.load("tarea34_cache/nq0.6_N20.npz"), np.load("tarea34_cache/nq0.6_N26.npz")
L.append(f"\nConvergencia N→N+6 (n_q=0.6): brecha {float(a['gap']):.5f}→{float(b['gap']):.5f} (Δrel {abs(float(b['gap'])/float(a['gap'])-1):.1e}); |⟨a²⟩| {float(a['a2']):.4f}→{float(b['a2']):.4f}.\n")
open("tarea31_34_resultados.md", "w").write("\n".join(L)); print("\n".join(L))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for v, n in names[:6]: ax[0].semilogx(GS, curves[v], '.-', label=n)
ax[0].semilogx(GS, eff, 'k--', label='efectivo (modelo_comun)'); ax[0].axhline(0.25, color='gray', ls=':'); ax[0].set_xlabel("Γ₂/κ"); ax[0].set_ylabel("brecha robusta"); ax[0].legend(fontsize=7)
ax[0].set_title("Escalera de ingredientes (α²=2)")
for v, c in [('s5', curves['s5'])] + list(inv.items()): ax[1].semilogx(GS, c, '.-', label=v)
ax[1].set_xlabel("Γ₂/κ"); ax[1].legend(fontsize=7); ax[1].set_title("Prueba inversa: s5 sin un ingrediente")
fig.tight_layout(); fig.savefig("tarea32_escalera.png", dpi=130)
