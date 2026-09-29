# Tarea 43 — baño filtrado (Ma re-sintonizado, |α|²=2, N=16, filtro de 2 niveles)

Parámetros: w=6, g_x=g_z=0.2121, κ=0.03, Ω=|α|²|G|, G=2g_xg_z/w=0.0150. κ₂=4G²/κ=3.000e-02; predicción de baño plano κ₁/κ₂=(5/72)(κ/g_z)²=1.389e-03, κ₁^plano=(10/9)g_x²κ/w²=4.167e-05.

**Expresión filtrada usada.** Filtro en ω_f=2w, ancho κ_f, acoplamiento J (4J²/κ_f=κ). La autoenergía del qubit a frecuencia ω es Σ(ω)=J²/(ω−ω_f+iκ_f/2); su parte imaginaria da la tasa efectiva κ_eff(ω)=4J²κ_f/(4(ω−ω_f)²+κ_f²) (=κ en ω=ω_f). Los canales de un fotón del efectivo (Γ₁∓=2g_x²Re S(D), S=1/(κ_eff/2+iD)) se evalúan a desintonías D=w y D=3w, es decir a δ=w y 3w del filtro: **κ₁^filt = g_x²[κ_eff(w)/w² + κ_eff(3w)/(9w²)]**, con κ_eff(δ)=κκ_f²/(4δ²+κ_f²) (usando 4J²=κκ_f). Para κ_f≪w: κ₁^filt/κ₁^plano ≈ [κ_f²/(4w²)]·(1+1/81)/(1+1/9)·... (≈0.91 κ_f²/4w²·(1+1/9)⁻¹·(1+1/81)). La expresión del enunciado κ_eff(w)=4J²κ_f/(κ_f²+4w²) es el primer término; aquí se añade el término contrarrotante D=3w.

κ₁ se mide (i) del modo de Floquet con mayor overlap con la paridad (tasa/(2|α|²)) y (ii) por evolución temporal de |0⟩|g⟩ (ajuste de la paridad en la ventana P_c>0.99, si la hay). κ₂ = brecha física (5º modo con peso de borde ≤0.1) y 4G²/κ.

| κ_f | J | κ₁ previsto | κ₁ medido (espectral) | κ₁ medido (temporal) | medido/previsto | κ₁/κ₂ (κ₂=4G²/κ) | κ₁/brecha | brecha física | mejora vs plano (medida) | mejora prevista | P_c máx |
|---|---|---|---|---|---|---|---|---|---|---|---|
| plano | nan | 4.167e-05 | 4.319e-05 | 3.440e-05 | 1.037 | 1.440e-03 | 1.035e-02 | 4.173e-03 | 1 | 1 | 0.9984 |
| 3.0 | 1.500e-01 | 2.235e-06 | 2.215e-06 | 1.796e-06 | 0.991 | 7.382e-05 | 6.141e-04 | 3.606e-03 | 19.5 | 18.6 | 0.9988 |
| 1.0 | 8.660e-02 | 2.618e-07 | 2.594e-07 | 2.087e-07 | 0.991 | 8.647e-06 | 8.354e-05 | 3.105e-03 | 166 | 159 | 0.9988 |
| 0.3 | 4.743e-02 | 2.371e-08 | 2.350e-08 | 1.888e-08 | 0.991 | 7.832e-07 | 1.059e-05 | 2.220e-03 | 1.84e+03 | 1.76e+03 | 0.9988 |
| 0.1 | 2.739e-02 | 2.636e-09 | 2.618e-09 | 2.242e-09 | 0.993 | 8.727e-08 | 2.682e-06 | 9.759e-04 | 1.65e+04 | 1.58e+04 | 0.9988 |

(el control plano usa el mismo N=16, α²=2; su κ₁ previsto es (10/9)g_x²κ/w².)
Validaciones (máximos): |Tr ρ−1| ≤ 6.9e-08, ‖ρ−ρ†‖ ≤ 1.9e-09, mín autovalor ≥ -1.3e-15.