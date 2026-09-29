# Tarea 23 — Acoplamiento óptimo (brecha de confinamiento vs Γ₂/κ)

Script: `tarea23_worker.py` (una celda por proceso fresco). |α|²=2 fijo,
g_x fijo, g_z escalado para 10 valores de Γ₂/κ log-espaciados entre 0.01
y 3. Nb=20 fijo (suficiente margen dado |α|²=2 constante). Modelo
efectivo corregido (amortiguamiento intrínseco siempre incluido).

## Tabla

| Γ₂/κ | Brecha completo | Brecha efectivo | razón eff/full |
|---|---|---|---|
| 0.0100 | 0.01519 | 0.01380 | 0.908 |
| 0.0188 | 0.03093 | 0.03389 | 1.096 |
| 0.0355 | 0.05683 | 0.07790 | 1.371 |
| 0.0669 | 0.10163 | 0.16210 | 1.595 |
| **0.1262** | **0.16090** | 0.31725 | 1.972 |
| 0.2378 | 0.15544 | 0.60499 | 3.892 |
| 0.4481 | 0.12854 | 1.14364 | 8.897 |
| 0.8446 | 0.09144 | 2.15656 | 23.59 |
| 1.5918 | 0.08901 | 4.06597 | 45.68 |
| 3.0000 | 0.12437 | 7.66742 | 61.65 |

## Máximo de la brecha completa

**Γ₂/κ óptimo ≈ 0.126, con brecha máxima ≈ 0.161** — identificado
directamente del máximo numérico entre los 10 puntos muestreados
(consistente con el crecimiento y decrecimiento vecinos: 0.102 antes,
0.155 después).

La brecha completa **no es monótona**: crece desde Γ₂/κ=0.01 hasta el
máximo en 0.126, decrece hasta un mínimo local en Γ₂/κ≈1.59 (0.089), y
**vuelve a subir** en Γ₂/κ=3 (0.124) — un comportamiento no capturado por
un modelo de un solo pico simple.

## Ajuste Δ_gap = A·x/(1+B·x²)

    A = 1.466 ± 0.354
    B = 16.66 ± 6.99
    R² = 0.465

**El ajuste es mediocre (R²=0.46)**, principalmente porque el modelo de
un solo pico no puede reproducir la subida final en Γ₂/κ=3 (el modelo
predice decrecimiento monótono para x grande, pero los datos suben de
nuevo ahí). El pico *teórico* del ajuste (x*=1/√B=0.245, valor=0.180) es
notablemente distinto del máximo *numérico observado* (x=0.126,
valor=0.161) — el ajuste, al intentar acomodar el comportamiento
completo (incluida la subida final), desplaza su pico hacia la derecha
respecto al máximo real de los datos centrales.

## La brecha del efectivo diverge sin control (confirma la Tarea 21)

La brecha efectiva crece monótonamente sin ningún máximo, de 0.014 a
7.67 en el rango probado — nunca se dobla ni decrece. Esto reconfirma
categóricamente el hallazgo de la Tarea 21: el modelo efectivo **no**
tiene ningún mecanismo que capture el aplanamiento/reversión de la
brecha de confinamiento real a acoplamiento fuerte.

## Conclusión Tarea 23

**Γ₂/κ óptimo ≈ 0.126** (brecha máxima ≈0.16, la formación más rápida
posible del gato según el modelo completo). El ajuste de ley de potencia
propuesto (A·x/(1+Bx²)) captura la forma cualitativa cerca del pico pero
**no describe bien todo el rango** (R²=0.46) debido a una subida no
modelada en Γ₂/κ grande — se recomienda, si se necesita una descripción
cuantitativa completa, un modelo con un término adicional (p.ej. una
segunda rama a Γ₂/κ≳1) en vez de la forma de un solo pico propuesta.
Independientemente del ajuste, el hallazgo central y robusto es que la
brecha del modelo efectivo **nunca** reproduce el máximo ni la forma de
la brecha real — diverge sin límite —, así que el criterio de
"acoplamiento óptimo" solo puede obtenerse de forma confiable con el
modelo completo (o el propagador de Floquet, que es la herramienta usada
aquí).
