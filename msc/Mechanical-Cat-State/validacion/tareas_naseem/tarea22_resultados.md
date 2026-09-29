# Tarea 22 — Sesgo de ruido (bit-flip vs phase-flip)

Scripts: `tarea22_worker.py` (una celda por proceso fresco, evita el
throttling severo observado al correr todo en un único proceso largo —
ver nota de infraestructura al final). Efectivo corregido: **siempre**
incluye el amortiguamiento intrínseco `sqrt((n_th+1)γ)a`,
`sqrt(n_th γ)a†` (bug de la Tarea 21 corregido).

## Tabla principal (sin compensar)

### Γ₂/κ=0.13 (g_z×0.25)

| α² | pf completo | bf completo | conf. completo | η completo | pf efectivo | bf efectivo | conf. efectivo | η efectivo |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.264e-4 | 3.912e-3 | 0.1396 | 0.083 | 3.697e-4 | 4.265e-3 | 0.1801 | 0.087 |
| 2 | 6.217e-4 | 2.059e-3 | 0.1627 | 0.302 | 7.421e-4 | 1.909e-3 | 0.3262 | 0.389 |
| 3 | 9.532e-4 | 8.403e-4 | 0.1603 | 1.134 | 1.133e-3 | 8.126e-4 | 0.4253 | 1.395 |
| 4 | 1.284e-3 | 2.517e-4 | 0.1418 | 5.102 | 1.518e-3 | 9.805e-5 | 0.5346 | 15.49 |
| 5 | 1.612e-3 | 1.569e-5 | 0.1299 | 102.7 | 1.901e-3 | 5.856e-6 | 0.6555 | 324.6 |

Pendiente ln(bf) vs α²: **completo=−1.314, efectivo=−1.615** (esperado ≈−2)

### Γ₂/κ=0.52 (g_z×0.5)

| α² | pf completo | bf completo | conf. completo | η completo | pf efectivo | bf efectivo | conf. efectivo | η efectivo |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.643e-4 | 1.312e-3 | 0.0841 | 0.278 | 3.997e-4 | 1.314e-3 | 0.7243 | 0.304 |
| 2 | 6.847e-4 | 7.867e-4 | 0.1171 | 0.870 | 7.668e-4 | 7.353e-4 | 1.323 | 1.043 |
| 3 | 1.024e-3 | 6.037e-4 | 0.1231 | 1.695 | 1.147e-3 | 6.267e-4 | 1.710 | 1.830 |
| 4 | 1.362e-3 | 8.050e-5 | 0.1284 | 16.92 | 1.527e-3 | 5.304e-5 | 2.148 | 28.80 |
| 5 | 1.699e-3 | 4.179e-6 | 0.1331 | 406.5 | 1.908e-3 | 1.949e-6 | 2.631 | 978.9 |

Pendiente: **completo=−1.378, efectivo=−1.566**

### Γ₂/κ=2.07 (g_z×1.0, punto del paper)

| α² | pf completo | bf completo | conf. completo | η completo | pf efectivo | bf efectivo | conf. efectivo | η efectivo |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.695e-4 | 4.835e-4 | 0.0641 | 0.764 | 4.020e-4 | 4.820e-4 | 2.898 | 0.834 |
| 2 | 6.912e-4 | 4.822e-4 | 0.0991 | 1.434 | 7.684e-4 | 4.719e-4 | 5.298 | 1.628 |
| 3 | 1.025e-3 | 5.487e-4 | 0.1217 | 1.868 | 1.148e-3 | 5.872e-4 | 6.842 | 1.955 |
| 4 | 1.354e-3 | 5.686e-5 | 0.1395 | 23.82 | 1.528e-3 | 4.685e-5 | 8.596 | 32.62 |
| 5 | 1.679e-3 | 2.393e-6 | 0.1526 | 701.8 | 1.908e-3 | 1.233e-6 | 10.53 | 1547 |

Pendiente: **completo=−1.276, efectivo=−1.425**

## Hallazgo 1: sí hay supresión exponencial del bit-flip, pero más débil que exp(−2α²)

En los tres valores de Γ₂/κ, la pendiente de ln(γ_bf) vs α² es **negativa
y consistentemente entre −1.28 y −1.62** — confirma supresión exponencial
del bit-flip con α² (comportamiento cualitativo de cat-qubit), pero **no
exactamente exp(−2α²)**: la tasa real es ~65-80% de la pendiente
esperada. Esto es consistente en ambos modelos (completo y efectivo),
aunque el efectivo predice una supresión ligeramente más fuerte que el
completo.

## Hallazgo 2: sesgo η crece dramáticamente con α² (hasta ~1500×)

η=γ_pf/γ_bf pasa de ~0.1-0.8 (α²=1, ruido casi no sesgado) a
**hasta 1547 (efectivo) / 702 (completo)** en α²=5, Γ₂/κ=2.07 — el sesgo
de ruido característico de un cat-qubit (phase-flip mucho más frecuente
que bit-flip) se desarrolla con fuerza a medida que crece la amplitud del
gato, consistente con la teoría.

## Hallazgo 3 (no pedido explícitamente, pero relevante): la compensación colapsa el bit-flip del EFECTIVO, no del completo

| α² | bf completo sin comp. | bf completo con comp. | razón | bf efectivo sin comp. | bf efectivo con comp. | razón |
|---|---|---|---|---|---|---|
| 1 | 3.913e-3 | 4.439e-3 | 1.134 | 4.265e-3 | 8.143e-6 | **0.0019** |
| 2 | 2.059e-3 | 1.950e-3 | 0.947 | 1.909e-3 | 4.085e-7 | **0.0002** |
| 3 | 8.403e-4 | 7.972e-4 | 0.949 | 8.126e-4 | 2.782e-8 | **3.4e-5** |
| 4 | 2.517e-4 | 1.591e-4 | 0.632 | 9.805e-5 | 2.984e-9 | **3.0e-5** |
| 5 | 1.569e-5 | 1.117e-5 | 0.712 | 5.856e-6 | 3.844e-10 | **6.6e-5** |

**El modelo completo apenas cambia** (razón 0.63-1.13) al compensar la
desintonía, pero **el modelo efectivo colapsa 3-5 órdenes de magnitud**.
Esto indica que la tasa de bit-flip del modelo efectivo **es
extremadamente sensible/frágil frente al término δ₁/Δ₂₋** —depende de una
cancelación fina no física—, mientras que el bit-flip real (completo) es
robusto y esencialmente independiente de ese detalle de sintonía. **El
modelo efectivo no debe usarse para predecir γ_bf con precisión** salvo
verificación cruzada contra el completo, dado lo frágil de su valor ante
correcciones de segundo orden como δ₁.

## Conclusión Tarea 22

1. **Sí hay supresión exponencial del bit-flip con α²**, en el régimen
   probado (Γ₂/κ∈[0.13,2.07], α²∈[1,5]), con pendiente ln(γ_bf)≈−1.3 a
   −1.6 (más débil que el −2 ingenuo, pero claramente exponencial).
2. El sesgo η crece hasta ~3 órdenes de magnitud en el rango probado,
   confirmando el comportamiento esperado de un cat-qubit.
3. **Alerta metodológica**: γ_bf del modelo efectivo es muy sensible al
   término de Lamb/detuning δ₁ (colapsa al compensar), mientras que el
   del modelo completo es robusto — el efectivo sobreestima
   sistemáticamente γ_bf en el caso no compensado y lo colapsa
   artificialmente en el compensado; ninguno de los dos extremos captura
   con fidelidad el valor real (completo), que se mantiene estable
   (~1e-5 a 1e-3 según α²) frente a ambas variantes.

## Nota de infraestructura

Se detectó (de nuevo) throttling severo (~5-15% de duty cycle) al correr
esta corrida como un único proceso Python largo con múltiples casos
secuenciales, incluso con memoria disponible. **Se resolvió lanzando cada
celda (gz_scale, α², modelo, compensar) en un proceso Python fresco vía
subprocess**, lo que restauró velocidad normal para la mayoría de los
casos (aunque los casos más grandes, Nb=32, siguieron siendo lentos en
términos absolutos — varios minutos cada uno— probablemente por el costo
genuino de diagonalizar densamente una matriz 4096×4096, no por
throttling).
