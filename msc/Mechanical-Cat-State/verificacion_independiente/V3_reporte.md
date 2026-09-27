# V3 — Reporte de verificación independiente de R3 (phase-flip intrínseco)

Fecha: 2026-09-26. Código: `v3_paridad.py`. Salida: `res_v2/v3_salida.txt`. Datos: la serie de V2 `res_v2/wp11.9800_N22.npz`
(marco de laboratorio, Floquet sobre T_p = 2π/ω_p, muestreo estroboscópico t = nT_p, 601 puntos hasta Γt = 300).
No hubo cómputo nuevo ni se leyó código del trabajo.

## Método
- Paridad ⟨e^{iπa†a}⟩ muestreada en t = nT_p (fase 0 del drive), por lo que no hay micromovimiento en la serie.
- Ajuste 1: recta de ln(paridad) frente a t, solo con puntos P_c > 0.99 (se cumple desde Γt = 23.3).
- Ajuste 2: A e^{−kt} + c, con un piso constante c.
- Predicción: 2|α|²(Γ₁⁻ + Γ₁⁺), con Γ₁⁻ = g_x²κ/(ω² + κ²/4) y Γ₁⁺ = g_x²κ/(9ω² + κ²/4). Se evalúa con y sin el término κ²/4, y con |α|² = 4 o |α|² = |⟨a²⟩| = 3.9734.

## Predicciones

| variante | tasa |
|---|---|
| \|α\|² = 4, con κ²/4 | 3.33331e-4 |
| \|α\|² = 4, sin κ²/4 | 3.33333e-4 |
| \|α\|² = 3.9734, con κ²/4 | 3.31113e-4 |

Con los parámetros de Ma (κ/ω = 0.005) el término κ²/4 cambia la predicción solo en 5.7e-6 relativo. **Aquí no permite discriminar entre las dos formas.**
(En la Tarea 46, con κ/ω ≈ 0.9, el efecto era del 16%.)

## Resultados

| ventana Γt (P_c > 0.99) | ajuste | tasa medida | medido / pred (\|α\|²=4, con κ²/4) | medido / pred (sin κ²/4) | medido / pred (\|α\|²=3.973) |
|---|---|---|---|---|---|
| **[26, 152]** (ventana del trabajo) | ln lineal | **3.3072e-4** | 0.9922 | 0.9922 | 0.9988 |
| [26, 100] | ln lineal | 3.3203e-4 | 0.9961 | 0.9961 | 1.0028 |
| [100, 200] | ln lineal | 3.2452e-4 | 0.9736 | 0.9736 | 0.9801 |
| [60, 300] | ln lineal | 3.0821e-4 | 0.9246 | 0.9246 | 0.9308 |
| [150, 300] | ln lineal | 2.8859e-4 | 0.8658 | 0.8658 | 0.8716 |
| [26, 152] | A e^{−kt} + c | 3.3316e-4 | 0.9995 | 0.9995 | — |
| [26, 300] | A e^{−kt} + c | 3.3319e-4 | 0.9996 | 0.9996 | — |
| [60, 300] | A e^{−kt} + c | 3.3334e-4 | 1.0000 | 1.0000 | — |

Comparación directa con el trabajo, en la misma ventana y con el mismo criterio P_c > 0.99:

| cantidad | mío | trabajo | dif. rel. |
|---|---|---|---|
| tasa medida, Γt ∈ [26, 152] | 3.3072e-4 | 3.309e-4 | −5e-4 |
| tasa predicha | 3.3333e-4 | 3.333e-4 | 0 |

## Observación: piso de paridad
- La tasa del ajuste logarítmico baja si la ventana llega a tiempos tardíos (0.87× de la predicción en [150, 300]), y los residuos de ln crecen hasta 0.2.
- La causa es un piso estacionario de paridad, c = +6.4e-4 ± 0.6e-4. La paridad estroboscópica vale 1.65e-3 a Γt = 300 y el piso es del mismo orden que la población fuera del código (1 − P_c = 2.5e-3).
- Con el piso incluido, la tasa es 3.332e-4–3.333e-4 en todas las ventanas, al 0.05% de la predicción.
- La ventana [26, 152] del trabajo es lo bastante temprana para que el piso sesgue poco (−0.8%). Eso explica su razón de 0.99.

## Validaciones y convergencia
- Las de V2 aplican a esta misma serie: |Tr ρ − 1| ≤ 9.3e-11, ‖ρ − ρ†‖ ≤ 3.9e-13, mínimo autovalor ≥ 0.
- Convergencia en N (Γt = 60): la paridad pasa de 0.1939 (N=22) a 0.1937 (N=28). La corrida N=28 solo llega a Γt = 60, así que no se ajustó una tasa a N=28.

## Veredicto
**Confirma R3.** En la ventana del trabajo la tasa medida es 3.307e-4, frente a 3.309e-4 del trabajo y 3.333e-4 predicha (razón 0.992).
Con el piso estacionario incluido la razón es 1.000 en todas las ventanas. El término κ²/4 es irrelevante en este régimen (efecto de 6e-6).

## Dudas y aproximaciones
- No se identificó el origen físico del piso c ≈ 6e-4. La explicación del estado estacionario fuera del código (componentes vestidas y excitación residual del qubit) es plausible, pero no está verificada.
- Con |α|² = |⟨a²⟩| medido (3.973) en vez de 4, la razón en la ventana del trabajo sale 0.999. La elección de |α|² importa al 0.7%.
- Solo se ajustó a N = 22.
