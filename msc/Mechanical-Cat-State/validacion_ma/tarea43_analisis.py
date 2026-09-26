import numpy as np, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
KF = ("0", "3", "1", "0.3", "0.1"); D = {k: np.load(f"cache43/kf{k}_N16.npz") for k in KF if glob.glob(f"cache43/kf{k}_N16.npz")}
kappa, w, g = 0.03, 6.0, 0.3; gx = g * np.sin(np.pi / 4); gz = g * np.cos(np.pi / 4); a2 = 2.0
def gap_phys(d, thr=0.1):
    good = [k for k in range(len(d['lam'])) if d['edge'][k] <= thr]; return d['lam'][good[4]].real if len(good) > 4 else np.nan
k2 = float(D["0"]['kappa2']); pred_flat = (5 / 72) * (kappa / gz)**2
L = ["# Tarea 43 — baño filtrado (Ma re-sintonizado, |α|²=2, N=16, filtro de 2 niveles)\n",
     f"Parámetros: w=6, g_x=g_z={gx:.4f}, κ=0.03, Ω=|α|²|G|, G=2g_xg_z/w={2*gx*gz/w:.4f}. κ₂=4G²/κ={k2:.3e}; predicción de baño plano κ₁/κ₂=(5/72)(κ/g_z)²={pred_flat:.3e}, κ₁^plano=(10/9)g_x²κ/w²={(10/9)*gx**2*kappa/w**2:.3e}.\n",
     "**Expresión filtrada usada.** Filtro en ω_f=2w, ancho κ_f, acoplamiento J (4J²/κ_f=κ). La autoenergía del qubit a frecuencia ω es Σ(ω)=J²/(ω−ω_f+iκ_f/2); su parte imaginaria da la tasa efectiva κ_eff(ω)=4J²κ_f/(4(ω−ω_f)²+κ_f²) (=κ en ω=ω_f). "
     "Los canales de un fotón del efectivo (Γ₁∓=2g_x²Re S(D), S=1/(κ_eff/2+iD)) se evalúan a desintonías D=w y D=3w, es decir a δ=w y 3w del filtro: "
     "**κ₁^filt = g_x²[κ_eff(w)/w² + κ_eff(3w)/(9w²)]**, con κ_eff(δ)=κκ_f²/(4δ²+κ_f²) (usando 4J²=κκ_f). Para κ_f≪w: κ₁^filt/κ₁^plano ≈ [κ_f²/(4w²)]·(1+1/81)/(1+1/9)·... (≈0.91 κ_f²/4w²·(1+1/9)⁻¹·(1+1/81)). La expresión del enunciado κ_eff(w)=4J²κ_f/(κ_f²+4w²) es el primer término; aquí se añade el término contrarrotante D=3w.\n",
     "κ₁ se mide (i) del modo de Floquet con mayor overlap con la paridad (tasa/(2|α|²)) y (ii) por evolución temporal de |0⟩|g⟩ (ajuste de la paridad en la ventana P_c>0.99, si la hay). κ₂ = brecha física (5º modo con peso de borde ≤0.1) y 4G²/κ.\n",
     "| κ_f | J | κ₁ previsto | κ₁ medido (espectral) | κ₁ medido (temporal) | medido/previsto | κ₁/κ₂ (κ₂=4G²/κ) | κ₁/brecha | brecha física | mejora vs plano (medida) | mejora prevista | P_c máx |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
k1flat = float(D["0"]['rate_spec']) / (2 * a2)
for k in KF:
    if k not in D: continue
    d = D[k]; kf = float(d['kf']); k1p = float(d['kappa1_pred']); k1s = float(d['rate_spec']) / (2 * a2); k1t = float(d['rate_fit']) / (2 * a2)
    J = np.sqrt(kappa * kf / 4) if kf > 0 else np.nan; gp = gap_phys(d)
    L.append(f"| {'plano' if kf == 0 else kf} | {J:.3e} | {k1p:.3e} | {k1s:.3e} | {k1t:.3e} | {k1s/k1p:.3f} | {k1s/k2:.3e} | {k1s/gp:.3e} | {gp:.3e} | {k1flat/k1s:.3g} | {float(D['0']['kappa1_pred'])/k1p:.3g} | {float(d['Pc_max']):.4f} |")
L.append("\n(el control plano usa el mismo N=16, α²=2; su κ₁ previsto es (10/9)g_x²κ/w².)")
L.append(f"Validaciones (máximos): |Tr ρ−1| ≤ {max(float(d['trace_err']) for d in D.values()):.1e}, ‖ρ−ρ†‖ ≤ {max(float(d['herm']) for d in D.values()):.1e}, mín autovalor ≥ {min(float(d['mineig']) for d in D.values()):.1e}.")
open("tarea43_resultados.md", "w").write("\n".join(L)); print("\n".join(L))
