# Tarea 14 — Curva de validez limpia (qubit base, con/sin término de Lamb)

Script: `tarea14_curva_validez.py`. Datos: `tarea14_resultados.npz`.
Qubit inicial BASE físico (`basis(Na,1)`, corrección de la Tarea 13).
Incluye g_z×0.125 (Nb=20, 51 puntos, τ_max≈309 — no se omitió).
Término de Lamb: δ₁=g_x²[Im S₁₋+Im S₁₊] = **−0.048** (constante, no depende de g_z ya que g_x es fijo).

## Tabla

| g_z×f | Γ₂/κ | F_min (sin Lamb) | F_int (sin) | t90f/t90e (sin) | F_min (con Lamb) | F_int (con) | t90f/t90e (con) |
|---|---|---|---|---|---|---|---|
| 0.125 | 0.0324 | 0.7211 | 0.7535 | 0.387 | **0.9910** | 0.9923 | 0.888 |
| 0.250 | 0.1296 | 0.9628 | 0.9658 | 0.964 | 0.9686 | 0.9910 | 1.013 |
| 0.500 | 0.5184 | 0.8914 | 0.9545 | 1.352 | 0.8915 | 0.9587 | 1.356 |
| 1.000 | 2.0736 | 0.7597 | 0.8476 | 1.938 | 0.7598 | 0.8494 | 1.938 |

## Validación

Traza/hermiticidad perfectas; positividad excelente en casi todos los
casos (mín. autovalor ~1e-17 a 1e-20); una excepción marginal en g_z×0.5
a tiempos intermedios/finales (−5.7e-12, −2.5e-9), aún dentro del umbral
−1e-8.

## Hallazgo central: el término de Lamb es crítico en el régimen de acoplamiento débil

**Sin el término de Lamb, la tendencia de la Tarea 11 se INVIERTE en el
punto más débil** (g_z×0.125): F_min cae a 0.721, peor que g_z×0.25
(0.963) — rompiendo la monotonía esperada. **Con el término de Lamb, se
restaura la monotonía y F_min salta a 0.991** — la mejora es dramática
justamente donde τ_max es más largo (309 unidades κ): un pequeño error de
frecuencia (δ₁=−0.048) sin corregir se acumula en una fase óptica
significativa sobre un tiempo de integración muy largo, degradando la
fidelidad severamente. En los demás casos (τ_max mucho menor: 77, 19, 4.8),
el efecto de δ₁ es marginal (cambios de <1% en F_min), como se espera si
el error de fase acumulado escala con δ₁×τ_max.

## Respuesta a la pregunta: ¿a qué Γ₂/κ se logra F_min>0.99?

**Con el término de Lamb incluido: Γ₂/κ ≤ 0.0324 (g_z×0.125)** —
primer caso, en las Tareas 11+14 combinadas, que cruza el umbral F_min>0.99.

Ajuste de ley de potencia (con Lamb, los 4 puntos):

    1 − F_min ≈ 0.1534 × (Γ₂/κ)^0.800

## Conclusión Tarea 14

Con la corrección del estado inicial del qubit (Tarea 13) Y el término de
Lamb δ₁ de la Ec. (19), el modelo efectivo con g_eff=2g **sí reproduce la
dinámica completa con alta fidelidad (F_min>0.99)**, pero únicamente en
un régimen de acoplamiento sustancialmente más débil que el usado en las
figuras publicadas del paper (Γ₂/κ≈2.07, donde F_min≈0.76 incluso con
Lamb y qubit correctamente inicializado). El término de Lamb no es
opcional para comparaciones a Γ₂/κ pequeño con ventanas de tiempo largas
(τ_max~10/Γ₂ grande) — omitirlo produce una degradación de fidelidad
espuria que puede incluso revertir la tendencia de mejora esperada al
bajar el acoplamiento.
