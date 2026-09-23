# Resumen final — Ronda 3 de validación (Tareas 9-12)

Continuación de `RESUMEN_FINAL.md` (Tareas 1-5) y `RESUMEN_FINAL_RONDA2.md`
(Tareas 6-8, que establecieron **g_eff=2g** como normalización correcta,
confirmada por álgebra exacta y medición coherente directa). Esta ronda
corrige el diagnóstico del "piso" de la Tarea 6b y explora el dominio de
validez dinámico del modelo efectivo. `atol=1e-10, rtol=1e-8`, positividad
con umbral −1e-9.

## Tabla por tarea

| Tarea | Método | Resultado clave |
|---|---|---|
| 9a — Aliasing, eps=0 | Muestreo estroboscópico (t=nT_m) | dn/dt cae 100× (−1.15e-2→−1.06e-4) pero el piso en ⟨n⟩≈0.022 **persiste** — el aliasing explica la deriva rápida, no el piso completo |
| 9b — Barrido ε/κ estroboscópico | Ídem, 7 valores de ε, τ_max=60 | dn/dt cae 120-810× en todo el rango. Para ε≳0.4κ: full/B(2g) = 0.99-1.06× (excelente). Para ε≲0.2κ persiste un piso no resuelto |
| 10 — Estado inicial del qubit | fig2 (ε=1.44) con qubit en base física vs excitada | F_min mejora de 0.509 (excitado, = fig2 original) a 0.769 (base) — mejora real pero el dip **no desaparece** |
| 11 — Dominio de validez dinámico | Escaneo Γ₂/κ con \|α\|²=ε/g_eff=2 fijo | F_min mejora monótonamente al bajar Γ₂/κ: 0.499 (Γ₂/κ=2.07) → 0.732 (0.518) → 0.902 (0.130). Ninguno alcanza F_min>0.99; el régimen de fig1/fig2/fig4/fig5 (Γ₂/κ≈2.07) es el de **peor** acuerdo dinámico probado |
| 12 — Estado oscuro | Estacionario en ε=1.44, comparado con gato par | \|⟨a²⟩\|=1.960 vs predicción 2.000 (2% de diferencia); F(ρ_ss, gato par)=0.967 — confirma el estacionario como estado oscuro tipo gato |

## Conclusión: dominio de validez en Γ₂/κ

**Dos preguntas distintas, dos respuestas distintas:**

1. **Normalización de g_eff** (¿g_eff=g o 2g?): resuelta con alta
   confianza en la Ronda 2 — **g_eff=2g**, válida en todo el rango de
   acoplamiento probado (confirmado aquí de nuevo en el estado oscuro,
   Tarea 12: acuerdo de 2% en \|α²\|).

2. **Validez del modelo efectivo para la DINÁMICA transitoria** (no solo
   el estacionario): depende fuertemente de Γ₂/κ.
   - **Estado estacionario** (⟨n⟩_ss, p_e_ss, ⟨a²⟩_ss): bien reproducido
     para ε≳0.4κ en todo el rango de Γ₂/κ probado (Tareas 6b, 9b, 12).
   - **Dinámica completa** (F_min alto en todo instante, no solo al
     final): requiere Γ₂/κ **sustancialmente menor que 1** — con
     Γ₂/κ=0.130 aún F_min solo llega a 0.90; extrapolando, F_min>0.99
     probablemente exige Γ₂/κ≲0.03-0.05, un régimen no verificado
     directamente en esta ronda (se omitió por costo computacional).
   - **Los parámetros del paper** (fig1/fig2/fig4/fig5, Γ₂/κ≈2.07)
     están en el régimen de **peor** acuerdo dinámico de los probados
     (F_min≈0.50 con la condición inicial original del qubit, mejorando
     a ≈0.77 con la condición inicial físicamente correcta) — el modelo
     efectivo describe bien el punto de partida y el punto de llegada
     (estados inicial/estacionario) de las figuras del paper, pero **no
     la trayectoria completa** en ese régimen de acoplamiento.

## Puntos abiertos (no resueltos en esta ronda)

1. El "piso" residual en ⟨n⟩ (~0.02-0.04 a ε pequeño) persiste incluso
   con muestreo estroboscópico limpio; su origen exacto no se aisló
   (Tarea 9). No afecta la conclusión sobre g_eff, pero merece más
   investigación si se necesita el límite ε≪κ con precisión.
2. El valor preciso de Γ₂/κ donde F_min cruza 0.99 no se determinó (se
   omitió el caso más costoso, g_z×0.125, Γ₂/κ≈0.032, por decisión
   explícita de acotar tiempo). Extrapolación sugiere que ese caso
   omitido probablemente todavía no alcanza 0.99.
3. Nota metodológica de infraestructura: lanzar cómputo largo vía el
   parámetro de backgrounding explícito de la herramienta produjo
   throttling severo (~1-3% de duty cycle) en esta máquina; ejecutar en
   primer plano (con paso automático a segundo plano solo al superar el
   timeout de la herramienta) funcionó de forma confiable en todos los
   casos de esta ronda.

## Archivos generados esta ronda

- `tarea9a_pasoA.py`, `tarea9_pasoA.npz`, `tarea9a_resultados.md`
- `tarea9b_pasoB.py`, `tarea9_pasoB.npz`, `tarea9b_output.log`, `tarea9b_resultados.md`
- `tarea10_estado_inicial_qubit.py`, `tarea10_resultados.npz`, `tarea10_output.log`, `tarea10_resultados.md`
- `tarea11_dominio_validez.py`, `tarea11_resultados.npz`, `tarea11_output.log`, `tarea11_resultados.md`
- `tarea12_estado_oscuro.py`, `tarea12_resultados.npz`, `tarea12_output.log`, `tarea12_resultados.md`
- `RESUMEN_FINAL_RONDA3.md` (este archivo)
