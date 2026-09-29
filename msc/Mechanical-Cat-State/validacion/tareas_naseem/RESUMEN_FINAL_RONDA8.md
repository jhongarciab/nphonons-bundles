> **CORREGIDO en la Ronda 13 (Tarea 38):** el 'óptimo Γ₂/κ≈0.126 con brecha 0.161', la no monotonía y la subida en Γ₂/κ=3 de la Tarea 23 eran modos de borde de Fock. La brecha física es monótona y satura en ~0.23. Ver `RESUMEN_FINAL_RONDA13.md`.

# Resumen final — Ronda 8 de validación (Tareas 22-23)

Continúa de `RESUMEN_FINAL_RONDA7.md`. Corrección de bug: el modelo
efectivo ahora **siempre** incluye el amortiguamiento intrínseco
(sqrt((n_th+1)γ)a, sqrt(n_th γ)a†), omitido en la Tarea 21.

## Tabla por tarea

| Tarea | Método | Resultado clave |
|---|---|---|
| 22 — Sesgo de ruido | Barrido Γ₂/κ∈{0.13,0.52,2.07} × \|α\|²∈{1..5}, completo vs efectivo | Supresión exponencial del bit-flip confirmada (pendiente ln(γ_bf)≈−1.3 a −1.6, más débil que el −2 ingenuo). Sesgo η=γ_pf/γ_bf crece hasta ~1500× en el rango. **Hallazgo extra**: γ_bf del efectivo colapsa 3-5 órdenes de magnitud al compensar δ₁, mientras el completo apenas cambia — el efectivo es frágil/no confiable para γ_bf |
| 23 — Acoplamiento óptimo | 10 valores de Γ₂/κ log-espaciados en [0.01,3], \|α\|²=2 fijo | **Γ₂/κ óptimo≈0.126, brecha máxima≈0.161** (completo). Ajuste A·x/(1+Bx²): A=1.47±0.35, B=16.7±7.0, R²=0.46 (mediocre, no captura una subida no-monótona en Γ₂/κ=3). Brecha del efectivo diverge sin máximo (confirma Tarea 21) |

## Conclusiones explícitas (según lo pedido)

### (1) ¿Hay supresión exponencial del bit-flip? ¿En qué régimen?

**Sí, en todo el rango probado** (Γ₂/κ∈[0.13,2.07], \|α\|²∈[1,5]):
ln(γ_bf) decrece linealmente con α² con pendiente entre −1.28 y −1.62
(vs. −2 ingenuo) — supresión exponencial real, aunque ~35% más débil que
la fórmula de referencia simple. Es consistente en ambos modelos
(completo y efectivo) y en los tres puntos de Γ₂/κ probados, aunque el
efectivo predice supresión ligeramente más fuerte. El sesgo de ruido
η crece dramáticamente (hasta >1500×) con \|α\|², confirmando el
comportamiento cualitativo esperado de un cat-qubit real en todo el
rango.

### (2) ¿Cuál es el Γ₂/κ óptimo?

**Γ₂/κ≈0.126**, donde la brecha de confinamiento del modelo completo
alcanza su máximo (≈0.161) — este es el punto de formación más rápida
del gato según el modelo completo (Floquet, sin aproximaciones). Los
parámetros publicados del paper (Γ₂/κ≈2.07) están **muy por encima** de
este óptimo, en una región donde la brecha real ya cayó a ≈0.15 desde su
pico y el modelo efectivo predice —incorrectamente— que seguiría
creciendo sin límite.

## Síntesis con hallazgos previos (Tareas 20-21)

La Tarea 21 ya había mostrado que la brecha de confinamiento del
efectivo diverge sin control con Γ₂/κ; la Tarea 23 añade que la brecha
**real** (completo) tiene un máximo bien definido en Γ₂/κ≈0.126 y luego
decrece —el sistema completo tiene un "punto dulce" de acoplamiento que
el modelo efectivo no puede predecir en absoluto. Combinado con la Tarea
22 (sesgo de ruido bien descrito cualitativamente por ambos modelos,
pero γ_bf del efectivo poco confiable cuantitativamente), el cuadro
completo es: **el modelo efectivo sirve para diseño cualitativo del
sesgo de ruido, pero el diseño cuantitativo del acoplamiento óptimo y de
γ_bf preciso requiere el modelo completo o el propagador de Floquet**.

## Nota de infraestructura (importante para trabajo futuro)

Se confirmó que ejecutar muchos casos de propagador de Floquet en un
único proceso Python largo produce throttling severo e impredecible en
esta máquina (duty cycle tan bajo como 3-15%), incluso con memoria
disponible y conectado a corriente. **La solución fue lanzar cada celda
del barrido en un subproceso Python fresco** (`tarea22_worker.py`,
`tarea23_worker.py`, invocados vía un bucle de shell), lo que restauró
velocidad normal. Se recomienda usar este patrón (proceso fresco por
celda) para cualquier barrido futuro con múltiples llamadas a
`propagator`/`eigenstates` en esta máquina.

## Archivos generados esta ronda

- `tarea22_worker.py`, `tarea22_cache/` (40 archivos .npz), `tarea22_resultados.md`
- `tarea23_worker.py`, `tarea23_cache/` (10 archivos .npz), `tarea23_fit.npz`, `tarea23_resultados.md`
- `RESUMEN_FINAL_RONDA8.md` (este archivo)
