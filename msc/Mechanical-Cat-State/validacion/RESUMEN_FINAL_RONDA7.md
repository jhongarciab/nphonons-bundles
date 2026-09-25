> **CORREGIDO en la Ronda 13 (Tarea 38):** la brecha 'completa' de la Tarea 21 en Γ₂/κ ≳ 0.24 era un modo de borde de Fock; el cociente efectivo/completo en 2.07 es ~23×, no 53.5× (la divergencia del efectivo se mantiene). Ver `RESUMEN_FINAL_RONDA13.md`.

# Resumen final — Ronda 7 de validación (Tareas 20-21)

Continúa de `RESUMEN_FINAL_RONDA6.md`. Usa Floquet (`qutip.propagator`,
Tarea 18) y Liouvilliano estático (`qutip.liouvillian`) para clasificar
modos espectrales por proyección sobre operadores físicos (P, n, a, a²,
σ_z).

## Tabla por tarea

| Tarea | Método | Resultado clave |
|---|---|---|
| 20 — Identificar modos lentos (paper, ε=1.44) | 8 autovalores del propagador de Floquet, clasificados por overlap | Jerarquía confirmada: confinamiento (τ≈10κ⁻¹) ≪ phase-flip (τ≈1447κ⁻¹) ≲ bit-flip/coherencia lógica (τ≈2074κ⁻¹, el más lento). **Modo 4 confirmado como brecha de confinamiento** (overlap con n̂=2.94, dominante) |
| 21 — Espectro completo vs efectivo | Liouvilliano estático (g_eff=2g+δ₁) vs Floquet completo, barrido Γ₂/κ y \|α\|² | **Brecha de confinamiento del efectivo diverge catastróficamente** (ratio efectivo/completo: 1.3× en Γ₂/κ=0.03 → 53.5× en Γ₂/κ=2.07). **Phase-flip real ≈4× la fórmula 2\|α\|²(Γ₋+Γ₊)** (factor estable ~4×, no depende fuerte de \|α\|²). Coherencia lógica: acuerdo razonable a bajo acoplamiento, se degrada moderadamente (ratio~2.8× en Γ₂/κ=2.07) |

## Conclusión sobre el dominio de validez (Γ₂/κ) — versión final

**La brecha de confinamiento del modelo efectivo deja de describir la
del completo ya en Γ₂/κ≈0.13** (factor 2×), y falla por completo
(factor >10×, hasta 53.5× en los parámetros del paper, Γ₂/κ≈2.07) para
acoplamientos mayores. Esto identifica la **causa raíz específica**
detrás del bajo F_min encontrado en la Tarea 11/14 en el régimen de los
parámetros publicados: no es la descripción del estado estacionario
(bien reproducida, Tareas 6b/9b/12/18) ni principalmente los canales de
decoherencia lógica del cat-qubit (razonablemente descritos, con una
corrección ~4× necesaria para el phase-flip), sino específicamente **la
velocidad a la que el modelo efectivo predice que se forma/confina el
gato** — que se dispara sin control al aumentar el acoplamiento, muy por
encima de la tasa real (aproximadamente constante) del modelo completo.

## Implicación práctica

Para trabajo futuro con esta familia de modelos: si se necesita el
modelo efectivo para predecir **tiempos de formación del gato** o **F_min
dinámico** en el régimen de acoplamiento fuerte (Γ₂/κ≳0.1-0.5, que
incluye los parámetros ya publicados), **no es confiable** — hay que usar
el modelo completo (o el propagador de Floquet, que es rápido, ~20-90s
por punto). Para estimar **tasas de error lógico del cat-qubit una vez
formado** (bit-flip, y phase-flip con la corrección ~4× encontrada aquí),
el modelo efectivo sigue siendo razonablemente útil en todo el rango.

## Archivos generados esta ronda

- `tarea20_modos_lentos.py`, `tarea20_resultados.npz`, `tarea20_output.log`, `tarea20_resultados.md`
- `tarea21_espectro_completo_vs_efectivo.py`, `tarea21_principal.npz`, `tarea21_bonus.npz`, `tarea21_output.log`, `tarea21_resultados.md`
- `RESUMEN_FINAL_RONDA7.md` (este archivo)
