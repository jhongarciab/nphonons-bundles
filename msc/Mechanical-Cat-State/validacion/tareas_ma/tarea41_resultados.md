# Tarea 41 — δ₁ frente a κ/ω_m (modelo_comun: g_x=6.0, ω_m=1000, κ variable)

Comprobación: δ₁(κ/ω_m=1e-03) = -0.04799999; `modelo_ladder.valores()['lamb']` = -0.04799999 (κ=1). Límite −4g_x²/(3ω_m) = -0.04800000.

| κ/ω_m | δ₁ | −4g_x²/(3ω_m) | razón δ₁/(−4g_x²/(3ω_m)) | 1 − razón | corrección analítica (7/9)(κ/2ω_m)² |
|---|---|---|---|---|---|
| 1.000e-04 | -0.04800000 | -0.04800000 | 0.999999998 | 1.944e-09 | 1.944e-09 |
| 2.154e-04 | -0.04800000 | -0.04800000 | 0.999999991 | 9.025e-09 | 9.025e-09 |
| 4.642e-04 | -0.04800000 | -0.04800000 | 0.999999958 | 4.189e-08 | 4.189e-08 |
| 1.000e-03 | -0.04799999 | -0.04800000 | 0.999999806 | 1.944e-07 | 1.944e-07 |
| 2.154e-03 | -0.04799996 | -0.04800000 | 0.999999097 | 9.025e-07 | 9.025e-07 |
| 4.642e-03 | -0.04799980 | -0.04800000 | 0.999995811 | 4.189e-06 | 4.189e-06 |
| 1.000e-02 | -0.04799907 | -0.04800000 | 0.999980556 | 1.944e-05 | 1.944e-05 |
| 2.154e-02 | -0.04799567 | -0.04800000 | 0.999909757 | 9.024e-05 | 9.025e-05 |
| 4.642e-02 | -0.04797990 | -0.04800000 | 0.999581300 | 4.187e-04 | 4.189e-04 |
| 1.000e-01 | -0.04790689 | -0.04800000 | 0.998060251 | 1.940e-03 | 1.944e-03 |

Parámetros de Ma (g_x=-0.2121, w=6.0, κ=0.03, κ/w=0.005): δ₁=-0.00999995 vs −4g_x²/(3w)=-0.01000000; w_p*=2(w+δ₁)=11.980000.

**Convenciones.** Naseem: oscilador ω_m, qubit a 2ω_m, referencia ω_r=ω_d/2, δ_m=ω_m−ω_r y δ₁ es el corrimiento (Lamb) de la frecuencia del oscilador; la resonancia vestida es δ_m=−δ₁ (>0 pues δ₁<0). Ma: oscilador w, qubit d=2w, marco a w_p/2 y w_p; resonancia w_p/2 = w + δ₁ ⇒ w_p*=2(w−4g_x²/(3w)) (mismo δ₁, con el signo tal cual). En ambos g_x entra al cuadrado (signo irrelevante) y los términos contrarrotantes dan el factor (1+1/3): D₁₋=ω, D₁₊=3ω. La diferencia de unidades: Naseem mide en κ (ω_m/κ=1000) y Ma en 2π·GHz (κ/w=0.005).