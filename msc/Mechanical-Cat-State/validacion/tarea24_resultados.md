# Tarea 24 — Floquet conmensurable y resonancia vestida

Scripts: `tarea24_verificacion.py`, `tarea24_worker.py`. Datos:
`tarea24_verificacion.npz`, `tarea24_cache/` (81+4 archivos),
`tarea24_grid_analisis.npz`.

## Corrección del bug de conmensurabilidad (Tareas 22-23)

Confirmado el diagnóstico: en el caso "compensar" de las Tareas 22-23,
el drive quedaba a `2(ω_m+δ₁)` mientras el acoplamiento mecánico seguía
oscilando a `ω_m` — dos frecuencias no conmensurables, por lo que
`propagator(H, T_m)` con `T_m=2π/ω_m` **no era un propagador de Floquet
válido** para ese `H(t)`. Esos resultados del modelo completo quedan
descartados (ya señalado en `HANDOFF.md`).

## Paso 0: verificación del marco conmensurable — PASA

Con δ_m=0, Δ_q=0, ω_r=ω_m (mismos parámetros que la Tarea 18-ii, ε=1.44,
Nb=20), el nuevo Hamiltoniano en el marco de frecuencia única ω_r
reproduce los 6 autovalores lentos de la Tarea 18-ii con **diferencia
relativa máxima 4.0e-10** (muy por debajo del umbral 1e-6 pedido).
Verificación exitosa — se procede con el resto de la tarea.

## Rejilla 9×9 en (δ_m, Δ_q), Γ₂/κ=0.52, |α|²=3

Centrada en (δ₁, 0) = (−0.048, 0), rango ±3|δ₁|=±0.144 en δ_m y Δ_q.
81 puntos, patrón worker (proceso fresco por celda), ~100-150s/celda.

Tabla completa en `tarea24_cache/*.npz` (ver script de análisis). Rango
de valores: γ_bf desde 9.4e-3 (peor, δ_m=−0.192) hasta 1.18e-4 (mejor,
δ_m=0.06, Δ_q=0.144, dentro de la rejilla pedida).

## Hallazgo: la resonancia vestida es una CRESTA en δ_m, no un punto aislado

**Im(λ_bf) = 0 casi exactamente (a precisión de máquina, ~1e-13 a
1e-14) en TODA la fila δ_m=0.06**, independientemente de Δ_q (9 de 9
puntos de esa fila dan Im(λ_bf)~1e-13). En ninguna otra fila de δ_m se
observa esto (todas las demás dan Im(λ_bf) de orden 1e-3 a 4e-2). Esto
identifica **δ_m=0.06 como la condición de resonancia vestida**, fijada
casi enteramente por el detuning mecánico, con dependencia muy débil en
Δ_q. Es la fila con los γ_bf más chicos de toda la rejilla (mínimo
global en el punto (0.06, 0.144): γ_bf=1.177e-4).

**Extensión exploratoria** (fuera del rango ±3|δ₁| pedido, para
verificar si el mínimo real está en el borde): a δ_m=0.06 fijo, γ_bf
sigue bajando lentamente al aumentar Δ_q más allá de 0.144
(Δ_q=0.18→1.163e-4, 0.216→1.148e-4, 0.288→1.121e-4, 0.36→1.094e-4) —
mejora marginal (~7% en 2.5× más rango), sin mínimo interior claro
dentro de lo explorado. **Para la Tarea 25 se usa el punto
(δ_m,Δ_q)=(0.06, 0.144), el mejor dentro del rango ±3|δ₁| especificado.**

## Comparación con (δ₁, 0)

| | δ_m | Δ_q |
|---|---|---|
| Predicción ingenua | −0.048 | 0.000 |
| Resonancia vestida hallada | **+0.060** | **+0.144** |
| Desplazamiento | +0.108 | +0.144 |

**La resonancia vestida real está muy desplazada de (δ₁, 0)** — no solo
en magnitud sino en **signo** para δ_m (δ₁ es negativo, la resonancia
real es positiva). δ₁ (el "Lamb shift" de la Ec. 19, un efecto de
1er orden en g_x²) **no predice** dónde ocurre la verdadera resonancia
vestida del sistema completo — hay una contribución adicional
significativa (posiblemente de orden g_z² o cruzada g_z·g_x, o efectos
no perturbativos del acoplamiento fuerte) que desplaza la resonancia
real lejos de la predicción de la teoría de perturbaciones a orden más
bajo.

## Conclusión Tarea 24

1. El bug de conmensurabilidad de las Tareas 22-23 está corregido y
   verificado (diferencia 4e-10 contra la Tarea 18-ii).
2. La resonancia vestida (mínimo de γ_bf, Im(λ_bf)≈0) en Γ₂/κ=0.52 está
   en **δ_m≈0.06** (una cresta, insensible a Δ_q) — **muy lejos** de la
   predicción ingenua (δ₁,0)=(−0.048,0). γ_bf en la resonancia
   (~1.2e-4) es **~8× menor** que en el punto de la rejilla más cercano
   a (δ₁,0) (γ_bf≈1.03e-3 en δ_m=−0.048,Δ_q=0) — la resonancia vestida
   real ofrece una supresión de bit-flip sustancialmente mejor que la
   estimación basada en δ₁ solo.
