# V4 — Reporte de verificación independiente de R4 (figura de mérito κ₁/κ₂ = (5/72)(κ/g_z)²)

Fecha: 2026-09-26. Código: `v4_espectral.py`, `run_v4.sh`. Datos: `res_v4/*.npz`, `res_v4/log_*.txt`.
Del trabajo solo se consultó `tarea42_resultados.md`, para las convenciones y los valores de los dos puntos comunes. No se leyó su código.

## Método (difiere del original)
- **Modelo completo en el marco de laboratorio, sin RWA**, con la convención de la Tarea 42: ω_q = ω_p = 2(ω − 4g_x²/3ω), κ = 0.03, |α|² = 4,
  Ω = |α|²G y G = 2g_xg_z/ω. Esto da α² = +4, un gato en ±2 reales.
- **Método espectral:** se calcula el propagador del superoperador sobre un período del drive, T_p = 2π/ω_p (el original usa 4π/ω_p en el marco rotante).
  Tasas r_k = −ln|λ_k|/T_p.
  - En el laboratorio el oscilador rota a ω_p/2, de modo que en un período |α⟩ pasa a |−α⟩. Por eso el modo de pozo tiene λ ≈ −1 y el modo de paridad tiene λ real ≈ +1.
  - El modo de paridad se identifica como el de mayor |Tr(P R_k)|/‖R_k‖ entre los 12 modos más lentos, con λ real positivo y peso de borde (n > N−6) menor que 0.5.
  - En todos los puntos la identificación es inequívoca: |Tr(P R)|/‖R‖ ≈ 1.40–1.41 para el modo de paridad, frente a ≤ 0.05 en el resto de modos lentos no estacionarios.
- κ₁ = r_par/(2|α|²), con |α|² = 4 nominal; κ₂ = 4G²/κ.
- **Confirmación temporal** en dos puntos: evolución período a período desde |0⟩|g⟩, muestreo estroboscópico en t = nT_p y ajuste A e^{−kt} + c con los puntos P_c > 0.99.
- N = 22 en todos los puntos; N = 28 en un punto para la convergencia.

## Resultados

| g_x | ω | g_z/κ | κ₂/κ | κ₁/κ₂ medido | predicho | **razón** | P_c estac. | tipo |
|---|---|---|---|---|---|---|---|---|
| 0.03 | 6 | 4 | 0.006 | 4.419e-3 | 4.340e-3 | **1.018** | 0.99954 | nuevo |
| 0.05 | 5 | 4 | 0.026 | 4.404e-3 | 4.340e-3 | **1.015** | 0.99932 | nuevo (ω=5) |
| 0.1 | 7 | 4 | 0.052 | 4.419e-3 | 4.340e-3 | **1.018** | 0.99958 | nuevo (ω=7), con evolución |
| 0.15 | 6 | 4 | 0.160 | 4.409e-3 | 4.340e-3 | **1.016** | 0.99909 | nuevo, con evolución |
| 0.05 | 6 | 5 | 0.028 | 2.818e-3 | 2.778e-3 | **1.014** | 0.99929 | común con T42 |
| 0.12 | 7 | 7 | 0.230 | 1.429e-3 | 1.417e-3 | **1.009** | 0.99890 | nuevo (ω=7) |
| 0.05 | 7 | 10 | 0.082 | 6.929e-4 | 6.944e-4 | **0.998** | 0.99779 | nuevo (ω=7, g_z/κ=10) |
| 0.08 | 5 | 10 | 0.410 | 6.757e-4 | 6.944e-4 | **0.973** | 0.99498 | nuevo (ω=5, g_z/κ=10) |
| 0.1 | 4 | 10 | 1.000 | 6.575e-4 | 6.944e-4 | **0.947** | 0.99095 | nuevo (saturado) |
| 0.05 | 6 | 12 | 0.160 | 4.702e-4 | 4.823e-4 | **0.975** | 0.99504 | común con T42 |

(Valores de κ₁/κ₂ de las filas 0.03/6/4, 0.1/7/4, 0.15/6/4, 0.12/7/7 y 0.08/5/10 calculados como razón × predicho; los valores exactos están en los logs.)

**Independencia de g_x y ω:** a g_z/κ = 4 fijo, con g_x ∈ {0.03, 0.05, 0.1, 0.15} y ω ∈ {5, 6, 7}, la razón está en 1.015–1.018 (dispersión 0.3%).
La fórmula captura la dependencia en g_x y ω; queda un exceso sistemático de ≈ +1.6%.

**Dependencia en g_z/κ:** la razón baja de ≈1.016 (g_z/κ = 4) a 0.97–0.95 (g_z/κ = 10–12).
El descenso se correlaciona con κ₂/κ: 0.998 con κ₂/κ = 0.08, 0.973 con 0.41 y 0.947 con 1.0.
Esto coincide con lo que el trabajo describe como pérdida lenta de exactitud al saturar.

## Comparación directa con la Tarea 42 (puntos comunes)

| punto | tasa de paridad espectral: mía / T42 | κ₁/κ₂: mío (espectral) / T42 (ajuste temporal) | razón: mía / T42 |
|---|---|---|---|
| g_x=0.05, ω=6, g_z/κ=5 | 1.8785e-5 / 1.879e-5 | 2.818e-3 / 2.810e-3 | 1.014 / 1.012 |
| g_x=0.05, ω=6, g_z/κ=12 | 1.8054e-5 / 1.805e-5 | 4.702e-4 / 4.690e-4 | 0.975 / 0.972 |

Las tasas espectrales coinciden con las de la Tarea 42 a 4 cifras (<3e-4 relativo), con una formulación distinta (laboratorio y T_p frente a marco rotante y 2T_p).
La diferencia de 0.2–0.3% en la razón viene de que el trabajo usa la tasa temporal y yo la espectral.

## Confirmación temporal (ajuste con piso)

| punto | ventana con P_c > 0.99 | k temporal | r espectral | k/r | piso c |
|---|---|---|---|---|---|
| 0.15 / 6 / 4 | t ∈ [1907, 20961] (κ₂t ∈ [9, 101]) | 1.6940e-4 | 1.6932e-4 | 1.0004 | +7.4e-4 |
| 0.1 / 7 / 4 | t ∈ [1953, 44892] (κ₂t ∈ [3, 70]) | 5.5419e-5 | 5.5414e-5 | 1.0001 | +6.9e-4 |

El piso de paridad (~7e-4) reaparece como en V3. Con él incluido, las tasas temporal y espectral coinciden al 0.04%.

## Validaciones y convergencia
- Estado estacionario (autovector de λ ≈ 1, normalizado): |Tr ρ − 1| ≤ 6e-16, ‖ρ − ρ†‖ = 0, mínimo autovalor ≥ −1.3e-14.
- Evoluciones temporales: |Tr ρ − 1| ≤ 4.8e-11, ‖ρ − ρ†‖ ≤ 6.5e-13, mínimo autovalor ≥ 0. Todas las tolerancias se cumplen.
- Convergencia en N para g_x=0.05, ω=6, g_z/κ=12:

  | cantidad | N=22 | N=28 | diferencia rel. |
  |---|---|---|---|
  | r_par | 1.80545e-5 | 1.80543e-5 | 1e-5 |
  | P_c estacionario | 0.99504 | 0.99504 | <1e-5 |
  | peso de borde del modo de paridad | 8.9e-7 | 1.9e-11 | — |

## Veredicto
**Confirma R4 dentro del rango declarado, con un matiz.**
- En los 10 puntos, 8 de ellos nuevos (ω = 5 y 7, g_z/κ = 4, 7 y 10), la razón está entre 0.947 y 1.018.
- La independencia de g_x y ω se sostiene al 0.3%.
- El matiz: el punto (0.1, 4, 10), con κ₂/κ = 1.0, da 0.947, por debajo del mínimo 0.963 reportado por el trabajo. Es un 5% bajo en régimen saturado.
- Para κ₂/κ ≲ 0.25 todos los puntos están dentro de ±2.5%.

## Dudas y aproximaciones
- κ₁ se define con |α|² = 4 nominal. En los puntos saturados ⟨a²⟩ estacionario es menor (3.85 en 0.05/6/12). Con |α|² efectivo las razones subirían en ~4%.
  Aquí no se hizo esa corrección, para seguir la convención del trabajo.
- El exceso de +1.6% a g_z/κ pequeño no se explica. Candidatos: correcciones de orden superior en g_x/ω, o el vestido del código. No se investigó.
- κ₁ viene del modo de paridad, que incluye la contribución de κ₂ a la paridad solo a través de fugas fuera del código. Esa contribución no se separó.
- No se midió la brecha, porque no se pedía en V4.
