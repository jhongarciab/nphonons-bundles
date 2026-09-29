"""Tablas (a) y (b) de la Tarea 25 a partir de tarea25_cache/."""
import numpy as np
GZ = [("0.03", "0.12028"), ("0.13", "0.25039"), ("0.52", "0.5"), ("2.07", "0.99913")]
A2 = [1, 2, 3, 4, 5]
L = ["# Tarea 25 — resonancia vestida (δ_m, Δ_q) = (0.048, 0.144)\n"]
res = {}
def slope(x, y):
    return np.polyfit(x, np.log(y), 1)[0]
L.append("## (a) γ_bf, γ_pf vs |α|²  (completo | efectivo resonante)\n")
for lab, gz in GZ:
    D = {a: np.load(f"tarea25_cache/gz{gz}_a2{a}.npz") for a in A2}
    L.append(f"### Γ₂/κ = {lab}\n\n| α² | γ_bf compl | γ_bf efec | γ_pf compl | γ_pf efec |\n|---|---|---|---|---|")
    for a in A2:
        d = D[a]
        L.append(f"| {a} | {float(d['bf_full']):.3e} | {float(d['bf_eff']):.3e} | {float(d['pf_full']):.3e} | {float(d['pf_eff']):.3e} |")
    s = {k: slope(A2, [float(D[a][k]) for a in A2]) for k in ("bf_full", "bf_eff", "pf_full", "pf_eff")}
    res[lab] = s
    L.append(f"\nPendiente d ln(γ)/dα²: bf completo **{s['bf_full']:+.3f}**, bf efectivo {s['bf_eff']:+.3f}, "
             f"pf completo {s['pf_full']:+.3f}, pf efectivo {s['pf_eff']:+.3f}\n")
L.append("## (b) Modo de confinamiento: γ_conf, Im(λ), overlaps (P, n, a)\n")
for lab, gz in GZ:
    L.append(f"### Γ₂/κ = {lab}\n\n| α² | modelo | γ_conf | Im(λ) | P | n | a |\n|---|---|---|---|---|---|---|")
    for a in A2:
        d = np.load(f"tarea25_cache/gz{gz}_a2{a}.npz")
        for m, t in (("full", "completo"), ("eff", "efectivo")):
            o = d[f"ov_conf_{m}"]
            L.append(f"| {a} | {t} | {float(d['conf_'+m]):.4e} | {float(d['im_conf_'+m]):+.1e} | {o[0]:.3f} | {o[1]:.3f} | {o[2]:.3f} |")
    L.append("")
open("tarea25_resultados.md", "w").write("\n".join(L))
np.savez("tarea25_resultados.npz", **{f"slope_{k}_{l}": v for l, s in res.items() for k, v in s.items()})
print("\n".join(L))
