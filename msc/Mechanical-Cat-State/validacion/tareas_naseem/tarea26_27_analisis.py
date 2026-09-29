import numpy as np
T_r = 2*np.pi/1000.0
L = ["# Tareas 26-27 — resultados (δ_m,Δ_q)=(0.048,0.144)\n", "## Tarea 26: convergencia de γ_bf (modelo completo)\n",
     "Resolución: |μ|−1 resoluble ~ rtol → γ_min ≈ rtol/T_r = "
     + ", ".join(f"{r:g}→{r/T_r:.1e}" for r in (1e-10, 1e-12, 1e-13)) + "\n",
     "| Γ₂/κ | α² | (atol,rtol) | γ_bf (1 período) | γ_bf (U^10) | 1−\\|μ\\| (1T) | γ_min≈rtol/T_r | γ_bf/γ_min |", "|---|---|---|---|---|---|---|---|"]
tols = [("1e-12","1e-10"),("1e-14","1e-12"),("1e-15","1e-13")]
for G in ("0.13","2.07"):
    for a2 in (3,4,5):
        for at,rt in tols:
            d = np.load(f"tarea26_cache/G{G}_a2{a2}_r{rt}.npz"); gm = float(rt)/T_r
            L.append(f"| {G} | {a2} | ({at},{rt}) | {float(d['gamma_bf_1']):.4e} | {float(d['gamma_bf_M']):.4e} | {float(d['one_minus_absmu_1']):.2e} | {gm:.1e} | {float(d['gamma_bf_1'])/gm:.1f} |")
L += ["", "## Tarea 27: brecha robusta (5º autovalor por Re λ ascendente), α²=2\n",
      "| Γ₂/κ | brecha completo | Im λ compl. | γ_pf compl. | brecha efectivo | Im λ efec. | γ_pf efec. | lógicos OK (c/e) |", "|---|---|---|---|---|---|---|---|"]
gs = ["0.01","0.03","0.07","0.13","0.24","0.45","0.84","1.6","3.0"]
gap_f, gap_e = [], []
for G in gs:
    d = np.load(f"tarea27_cache/G{G}_a22.npz")
    gap_f.append(float(d['gap_full'])); gap_e.append(float(d['gap_eff']))
    L.append(f"| {G} | {float(d['gap_full']):.4e} | {float(d['im_gap_full']):+.3e} | {float(d['gamma_pf_full']):.4e} | {float(d['gap_eff']):.4e} | {float(d['im_gap_eff']):+.3e} | {float(d['gamma_pf_eff']):.4e} | {bool(d['logicos_ok_full'])}/{bool(d['logicos_ok_eff'])} |")
open("tarea26_27_resultados.md","w").write("\n".join(L))
print("\n".join(L))
