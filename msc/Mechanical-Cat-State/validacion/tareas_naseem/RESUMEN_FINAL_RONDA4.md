# Resumen final — Ronda 4 de validación (Tareas 13-15)

Continúa de `RESUMEN_FINAL_RONDA3.md`. Corrección clave de esta ronda:
**todas** las corridas previas salvo la Tarea 10 iniciaron el qubit en el
estado FÍSICAMENTE EXCITADO (`basis(Na,0)`, mal etiquetado "ground" en el
código original). De aquí en más se usa siempre `basis(Na,1)` (base
física real). `atol=1e-10, rtol=1e-8`, positividad con umbral −1e-8,
muestreo estroboscópico en todas las corridas.

## Tabla por tarea

| Tarea | Método | Resultado clave |
|---|---|---|
| 13 — Verificar quench | eps=0, τ_max=30, qubit base vs excitado | Base: ⟨n⟩final=6.2e-3 (NO llega a <1e-4 esperado). Excitado: 2.22e-2. Descomposición aditiva piso(excitado)≈(2g_z/ω_m)²+piso(base) ajusta con 8% de error |
| 14 — Curva de validez limpia | Qubit base, \|α\|²=2 fijo, g_z×{0.125,0.25,0.5,1}, con/sin término de Lamb δ₁ | **Con Lamb, F_min>0.99 se logra en Γ₂/κ=0.0324** (g_z×0.125) — sin Lamb, ese mismo caso da F_min=0.72 (peor que casos con más acoplamiento, rompiendo monotonía). Ajuste: 1−F_min≈0.153×(Γ₂/κ)^0.80 |
| 15 — Patada de polarón y paridad | Qubit base, g_eff/Γ₂ fijo, g_z/ω_m×{0.015,0.03,0.06,0.12} (g_z·g_x=cte) | 1−paridad crece con g_z/ω_m pero **mucho más** que (2g_z/ω_m)² predice (factor ~4-20× de exceso) más un intercepto ~0.019 no explicado |

## Conclusión: dominio de validez en Γ₂/κ — respuesta ahora completa

La Ronda 3 (Tarea 11) dejó abierta la pregunta de a qué Γ₂/κ se alcanza
F_min>0.99, sin poder afirmarlo por el costo del caso más débil y por el
sesgo de iniciar el qubit excitado. **Esta ronda la resuelve**:

**Γ₂/κ ≤ 0.032, con el qubit inicializado en su estado base físico Y con
el término de Lamb δ₁ (Ec. 19) incluido en el modelo efectivo, da
F_min>0.99.** Ambas correcciones son necesarias simultáneamente:
- Sin la corrección del qubit (Ronda 3): el "piso" de contaminación por
  quench oscurecía la comparación en todo el rango.
- Sin el término de Lamb (esta ronda, "sin Lamb"): incluso con el qubit
  correctamente inicializado, el caso de menor Γ₂/κ (el más favorable
  según la intuición perturbativa) daba **peor** fidelidad que casos de
  acoplamiento intermedio, por acumulación de fase no compensada sobre
  ventanas de tiempo largas (τ_max~10/Γ₂ grande cuando Γ₂ es chico).

Los parámetros de las figuras publicadas (Γ₂/κ≈2.07) siguen muy lejos de
este régimen validado (F_min≈0.76 incluso con ambas correcciones) — el
modelo efectivo describe bien el estado inicial y el estacionario ahí
(Tareas 6b, 9b, 12), pero no la dinámica transitoria completa.

## Puntos abiertos persistentes

1. **Piso residual** (~6.2e-3 en ⟨n⟩ a τ_max=30, qubit base, eps=0):
   confirmado que no es enteramente el artefacto de quench del qubit
   excitado (Tarea 13); su origen exacto sigue sin aislarse.
2. **Exceso de pérdida de paridad** sobre la predicción (2g_z/ω_m)²
   (Tarea 15): factor 4-20× de exceso más un intercepto ~0.019,
   posiblemente relacionado con el mismo piso residual del punto 1.
3. Ambos puntos abiertos comparten un orden de magnitud similar (~0.02),
   lo que sugiere que podrían tener un origen común no identificado —
   candidato para investigación futura si se requiere precisión mayor
   en el límite de acoplamiento débil.

## Nota de infraestructura

La máquina estuvo funcionando con batería (56%, sin cargador) durante
parte de esta ronda, causando throttling intermitente de los procesos de
cómputo largo (hasta ~50× más lento en algunos tramos). Se recomienda
conectar el cargador para corridas largas futuras.

## Archivos generados esta ronda

- `tarea13_verificar_quench.py`, `tarea13_resultados.npz`, `tarea13_output.log`, `tarea13_resultados.md`
- `tarea14_curva_validez.py`, `tarea14_resultados.npz`, `tarea14_output.log`, `tarea14_resultados.md`
- `tarea15_patada_polaron.py`, `tarea15_resultados.npz`, `tarea15_output.log`, `tarea15_resultados.md`
- `RESUMEN_FINAL_RONDA4.md` (este archivo)
