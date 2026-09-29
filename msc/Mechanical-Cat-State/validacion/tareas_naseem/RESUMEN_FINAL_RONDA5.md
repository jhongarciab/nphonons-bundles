# Resumen final — Ronda 5 de validación (Tareas 16-17)

Continúa de `RESUMEN_FINAL_RONDA4.md`. Qubit siempre en `basis(Na,1)`
(base física), muestreo estroboscópico, umbral positividad −1e-8.

## Tabla por tarea

| Tarea | Método | Resultado clave | Veredicto |
|---|---|---|---|
| 16 — ¿Piso numérico? | Escaneo (atol,rtol)×3, Nb×2, luego aislar g_x=0 / g_z=0 | Pendiente varía solo 0.08% entre tolerancias — **es físico, no numérico**. Requiere AMBOS g_z y g_x simultáneos (completo=6.19e-3, solo g_x=1.27e-4, solo g_z≈0) — efecto de término cruzado, ~5 órdenes de magnitud mayor que la predicción perturbativa Γ₂₊ | **Confirmado: piso físico, origen cruzado g_z·g_x** |
| 17 — Patada polarónica (tasas) | −dP/dt completo vs efectivo, δ₁ compensado por retuneo de ω_q, ajuste diff vs (g_z/ω_m)² | Serie "diff" no monótona, R²=0.052 (sin correlación), un caso (g_z/ω_m=0.06) falla validación de positividad | **No concluyente** — ni confirma ni descarta la hipótesis del canal de patada |

## Conclusión combinada

**Tarea 16 es un resultado sólido y limpio**: el piso residual en ⟨n⟩ que
venía persiguiéndose desde la Ronda 3 (Tareas 9, 13) es un efecto físico
genuino, no un artefacto de tolerancia/truncamiento, y requiere el
acoplamiento cruzado g_z·g_x (consistente con ser una manifestación no
perturbativa del proceso de dos fonones — pero **~10⁵× más grande** que
lo que predice la fórmula Γ₂₊ de la eliminación adiabática a orden más
bajo, en el régimen g_x,g_z≫κ del paper).

**Tarea 17 no logró conectar** este mismo mecanismo con el exceso de
pérdida de paridad de la Tarea 15 mediante una medición de tasas —el
experimento diseñado para aislarlo cuantitativamente dio resultados
ruidosos y no interpretables con confianza, con al menos un caso con
problemas numéricos genuinos (falla de positividad). **La pregunta
"¿el exceso de pérdida de paridad escala como (g_z/ω_m)²?" sigue abierta**,
pendiente de una implementación más cuidadosa (ventana de estabilización
adaptada a δ₁, mejor resolución numérica en el punto problemático).

## Estado general de los puntos abiertos (actualizado)

1. ~~¿El piso es numérico?~~ → **Resuelto: es físico** (Tarea 16).
2. ¿Cuál es el origen microscópico exacto del piso (más allá de "requiere
   g_z·g_x")? → Aún sin una fórmula cuantitativa que lo reproduzca
   (la Γ₂₊ perturbativa falla por 5 órdenes de magnitud).
3. ¿El exceso de pérdida de paridad (Tarea 15) tiene el mismo origen que
   el piso de ⟨n⟩? → **Sin resolver** (Tarea 17 no concluyente).

## Nota de infraestructura

Esta ronda sufrió throttling severo por batería (~1-3% duty cycle
durante ~1h15 en la Tarea 17); el usuario conectó el cargador hacia el
final de la corrida. Se reafirma la recomendación: conectar el cargador
**antes** de iniciar corridas largas de esta serie.

## Archivos generados esta ronda

- `tarea16_piso_numerico.py`, `tarea16_resultados.npz`, `tarea16_output.log`, `tarea16_resultados.md`
- `tarea17_patada_limpia.py`, `tarea17_resultados.npz`, `tarea17_output.log`, `tarea17_resultados.md`
- `RESUMEN_FINAL_RONDA5.md` (este archivo)
