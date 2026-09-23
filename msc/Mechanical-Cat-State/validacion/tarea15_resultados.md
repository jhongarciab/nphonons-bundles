# Tarea 15 — Patada de polarón y paridad

Script: `tarea15_patada_polaron.py`. Datos: `tarea15_resultados.npz`.
Qubit BASE físico. g_eff=0.36 fijo, ε/g_eff=2 fijo, Γ₂/κ=0.5184 fijo,
τ_max=28.94 (igual en todos los casos). g_z·g_x=180 constante; g_x
ajustado en cada caso: {12.0, 6.0, 3.0, 1.5} para g_z/ω_m={0.015, 0.03,
0.06, 0.12}.

## Tabla

| g_z/ω_m | (2g_z/ω_m)² (predicción) | paridad | 1−paridad | población impar | \|⟨a²⟩\| | F(gato) |
|---|---|---|---|---|---|---|
| 0.015 | 0.00090 | 0.9814 | 0.0186 | 0.00928 | 1.2992 | 0.9707 |
| 0.030 | 0.00360 | 0.9668 | 0.0332 | 0.01660 | 1.9276 | 0.9889 |
| 0.060 | 0.01440 | 0.9192 | 0.0808 | 0.04042 | 1.9473 | 0.9791 |
| 0.120 | 0.05760 | 0.7559 | 0.2441 | 0.12206 | 1.8104 | 0.9355 |

**Modelo efectivo (referencia, sin dependencia de g_z/ω_m)**: paridad=0.9956
(1−paridad=4.4e-3, atribuible al canal Γ₁ residual).

## Validación

Traza/hermiticidad perfectas; positividad OK (mín. autovalor ~1e-9 a
1e-22, todos dentro del umbral −1e-8).

## Contraste con la hipótesis (1−paridad ∝ (2g_z/ω_m)²)

**La hipótesis captura la tendencia cualitativa pero NO el valor
cuantitativo**: 1−paridad medido es sistemáticamente **mucho mayor** que
la predicción (2g_z/ω_m)² — por un factor ~20× en g_z/ω_m=0.015, bajando
a ~4.2× en g_z/ω_m=0.12.

Ajuste lineal en (g_z/ω_m)²:

    1 − paridad ≈ 15.72 × (g_z/ω_m)² + 0.0190

El término cuadrático (15.72) es ~4× más grande que el coeficiente 4
esperado de la hipótesis literal, y hay además un **intercepto no nulo**
(~0.019) que no depende de g_z/ω_m — comparable en magnitud al valor
medido en el punto más pequeño (0.0186). Esto sugiere que el modelo
(2g_z/ω_m)² captura solo una parte del efecto, y que hay una
**contribución adicional, aproximadamente constante**, a la pérdida de
paridad — posiblemente relacionada con el mismo piso residual identificado
en las Tareas 9 y 13 (no explicado por completo por Γ1 ni por el quench
del qubit), que aquí se manifiesta como un suelo en la pérdida de paridad
incluso para el acoplamiento longitudinal más débil probado.

## |⟨a²⟩| y fidelidad con el gato

\|⟨a²⟩\| se mantiene razonablemente cerca de la predicción del estado
oscuro (2.0) para g_z/ω_m∈{0.03,0.06} (1.93, 1.95), pero se degrada en
los extremos: 1.30 en g_z/ω_m=0.015 (acoplamiento longitudinal muy débil,
el "kick" polarónico es pequeño pero también lo es la generación efectiva
del proceso de dos fonones dentro del τ_max fijo — posible falta de
convergencia al estacionario en ese caso) y 1.81 en g_z/ω_m=0.12 (kick
grande degradando la coherencia). La fidelidad con el gato par sigue una
tendencia similar: alta (~0.97-0.99) en el rango intermedio, cayendo a
0.935 en el extremo de acoplamiento fuerte (g_z/ω_m=0.12).

## Conclusión Tarea 15

La dirección del efecto (más "patada" polarónica → menos paridad, menos
estado tipo gato) se confirma cualitativamente, pero la ley de escala
cuadrática simple (2g_z/ω_m)² **subestima considerablemente** la pérdida
de paridad real, y no captura el suelo aproximadamente constante presente
en todos los casos. Se recomienda no usar (2g_z/ω_m)² como estimador
cuantitativo de la infidelidad por paridad sin una corrección aditiva
adicional (~0.02 en estos parámetros) cuyo origen no se aisló por
completo en esta ronda.
