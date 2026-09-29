# Tarea 47 — cierres finales

## (a) Peso de los estados iniciales sobre la rama interior (autovectores izquierdos, α²=2, resonancia vestida)

Fracción del estado inicial (relativa al peso total sobre modos no-código, como en la Tarea 35) sobre la rama real interior (modo real, 0.03<Re<0.2, overlap con n>1.5). Irrelevante si <1e-4.

| Γ₂/κ | N | Re de la rama | |1.3α⟩|g⟩ | |4⟩|g⟩ | |α⟩|e⟩ | ¿irrelevante (<1e-4)? |
|---|---|---|---|---|---|---|
| 0.5 | 26 | 0.1244 | 1.4e-15 | 5.8e-16 | 9.9e-15 | SÍ |
| 0.5 | 32 | 0.1261 | 1.1e-15 | 1.0e-15 | 1.2e-15 | SÍ |
| 1.0 | 26 | 0.0896 | 4.6e-16 | 7.8e-16 | 3.1e-16 | SÍ |
| 1.0 | 32 | 0.0878 | 7.3e-14 | 1.5e-13 | 9.4e-14 | SÍ |
| 3.0 | 26 | 0.1313 | 1.1e-14 | 4.6e-16 | 1.7e-16 | SÍ |
| 3.0 | 32 | 0.1249 | 1.1e-15 | 5.1e-16 | 2.0e-15 | SÍ |

## (c) 45(d) 'nocounter' re-sintonizado con δ_osc=−g_x²/w (w_p=2(w−g_x²/w), d=w_p)

Predicción: quitar los términos de 3ω elimina el factor (1+1/9) ⇒ κ₁ baja a 1/(10/9)=0.900 de 'full'.

| g_z/κ | variante | P_c máx | tasa de paridad (ajuste) | tasa (espectral) | κ₁ respecto a full (ajuste / espectral) | κ₁/κ₂ |
|---|---|---|---|---|---|---|
| 5 | full | 0.99930 | 1.873e-05 | 1.879e-05 | 1.000 / 1.000 | 2.810e-03 |
| 5 | nocounter (sin re-sintonizar, 45d) | 0.99663 | 2.325e-05 | 2.317e-05 | 1.241 / 1.234 | 3.487e-03 |
| 5 | nocounter re-sintonizado | 0.99936 | 1.652e-05 | 1.657e-05 | 0.882 / 0.882 | 2.479e-03 |
| 12 | full | 0.99517 | 1.801e-05 | 1.805e-05 | 1.000 / 1.000 | 4.690e-04 |
| 12 | nocounter (sin re-sintonizar, 45d) | 0.99581 | 2.239e-05 | 2.230e-05 | 1.243 / 1.235 | 5.830e-04 |
| 12 | nocounter re-sintonizado | 0.99589 | 1.617e-05 | 1.621e-05 | 0.898 / 0.898 | 4.210e-04 |

## (d) Ancho de la resonancia: FWHM = c·G (G=2g_xg_z/w=√(κκ₂)/2), 7 valores de κ₂/κ

FWHM medio-altura de P_c(w_p) a κ₂t=120 (PCHIP monótona; d=12 fijo). G en unidades absolutas (κ=0.03).

| κ₂/κ | κ₂ | G | FWHM | FWHM/κ₂ | c=FWHM/G | pico P_c |
|---|---|---|---|---|---|---|
| 0.03 | 9.000e-04 | 2.598e-03 | 6.375e-03 | 7.084 | 2.454 | 0.9986 |
| 0.05 | 1.500e-03 | 3.354e-03 | 8.941e-03 | 5.961 | 2.666 | 0.9986 |
| 0.1 | 3.000e-03 | 4.743e-03 | 1.365e-02 | 4.551 | 2.878 | 0.9985 |
| 0.2 | 6.000e-03 | 6.708e-03 | 2.046e-02 | 3.409 | 3.050 | 0.9985 |
| 0.3 | 9.000e-03 | 8.216e-03 | 2.547e-02 | 2.830 | 3.100 | 0.9984 |
| 0.5 | 1.500e-02 | 1.061e-02 | 3.305e-02 | 2.203 | 3.116 | 0.9983 |
| 1 | 3.000e-02 | 1.500e-02 | 4.239e-02 | 1.413 | 2.826 | 0.9975 |

**c = FWHM/G: media 2.870, desviación estándar 0.246 (8.6%), rango [2.454, 3.116] (n=7).** Ajuste libre FWHM∝κ₂^0.550 (=G^1.101); ajuste con pendiente fija 1 en G (FWHM=cG por mínimos cuadrados en log): c=2.861.
Tendencia de c con κ₂/κ: 0.03→2.45, 0.05→2.67, 0.1→2.88, 0.2→3.05, 0.3→3.10, 0.5→3.12, 1.0→2.83.