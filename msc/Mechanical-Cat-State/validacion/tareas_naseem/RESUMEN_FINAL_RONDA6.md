# Resumen final — Ronda 6 de validación (Tareas 18-19, análisis de Floquet)

Continúa de `RESUMEN_FINAL_RONDA5.md`. Se abandona la medición por
mesolve de tiempo finito para el estado estacionario y se usa el
**propagador de Floquet exacto** (`qutip.propagator` sobre un período
T_m=2π/ω_m), mucho más rápido (~20-90s por caso) y sin ambigüedad de
convergencia.

## Tabla por tarea

| Tarea | Método | Resultado clave |
|---|---|---|
| 18(i) — Floquet, eps=0, parámetros del paper | Punto fijo exacto del propagador de un período | ⟨n⟩_fijo=**0.0236**, 6.6× la predicción (g_z/ω_m)²=0.0036 y 3.8× el piso no-convergido de la Tarea 16 (0.0062) → **es calentamiento genuino, no relajación a estado base vestido** |
| 18(ii) — Floquet, eps=1.44 (punto del paper) | Punto fijo + espectro de decaimiento, modo de paridad identificado por proyección | **Paridad verdadera del estacionario = 0.052** (¡no 0.87 como en la Tarea 12!) — existe un modo ultra-lento λ₃=6.912e-4κ (τ≈1447κ⁻¹), 24× más largo que cualquier τ_max usado en rondas previas |
| 19 — Origen del calentamiento | Escaneo Floquet g_z/ω_m×{0.015,...,0.12}, g_z·g_x=cte, identificación de modo dominante por proyección sobre n̂ | **λ_heat vs g_x²: R²=1.0000** (perfecto); vs (g_z/ω_m)²: R²=0.37 (malo). Intercepto=1.496e-4 ≈ γ_m/κ=1.500e-4 (0.3% de diferencia) — el canal es Γ1 estándar (∝g_x²+γ_m), no un efecto nuevo de g_z |

## Conclusiones explícitas (según lo pedido)

### (1) ¿El piso es relajación o calentamiento?

**Es calentamiento hacia un estacionario caliente**, no relajación a un
estado base vestido de baja ocupación. El punto fijo exacto (0.0236) es
varias veces mayor que la predicción de "relajación" (g_z/ω_m)²=0.0036.
Además, **las Rondas 3-5 completas nunca alcanzaron el verdadero
estacionario**: con drive (ε=1.44), el modo de relajación de paridad más
lento tiene τ≈1447 κ⁻¹, mientras que todas esas rondas usaron τ_max≤77.
Esto implica que las cifras de "estado estacionario" reportadas en
Tareas 6a, 6b, 9b, 10, 11, 12, 14 y 15 son en realidad propiedades de un
**plateau transitorio de larga vida**, no del verdadero punto fijo — una
distinción importante pero que **no invalida** esas conclusiones para
fines prácticos: el plateau es lo que se observaría en cualquier
experimento/simulación con tiempos razonables (κt~10-100), y de hecho es
un mejor cat state (paridad 0.87) que el verdadero estacionario final
(paridad 0.052).

### (2) ¿γ_phase-flip coincide con la tasa de calentamiento?

**No exactamente, pero sí en orden de magnitud** (γ_phase-flip=6.91e-4 vs
λ_heat=1.97e-4 en los mismos g_z,g_x, razón 3.5×). Más importante: **el
mecanismo identificado para λ_heat es el canal Γ1 estándar (∝g_x²+γ_m,
R²=1.0000)**, no un efecto nuevo de "patada polarónica" asociado a g_z —
la hipótesis de las Tareas 15/17 sobre un canal de patada escalando como
(g_z/ω_m)² **no encuentra respaldo** en este análisis más riguroso.

## Revisión de conclusiones previas

- La **Tarea 12** (paridad_ss=0.87 en ε=1.44) describe correctamente el
  plateau transitorio observado a τ_max=60, pero **no** el verdadero
  estado estacionario (paridad=0.052) — debe leerse con esa salvedad.
- La **Tarea 16** (aislamiento g_x=0/g_z=0) sigue siendo válida como
  medición de snapshot a τ_max=30, pero la interpretación mecanística se
  refina: el modo espectral dominante de relajación es g_x²-driven
  (Γ1 estándar), no un efecto cruzado exótico — la necesidad de "ambos
  términos" en esa tarea probablemente refleja la dinámica de corto plazo
  (transitorio), no el canal asintótico dominante.
- La normalización **g_eff=2g (Tareas 7-8) permanece intacta** — depende
  de la dinámica de corto plazo/coherente, no del estado estacionario de
  larguísimo plazo aquí revisado.

## Archivos generados esta ronda

- `tarea18_floquet.py`, `tarea18_resultados.npz`, `tarea18_output.log`, `tarea18_resultados.md`
- `tarea19_origen_calentamiento.py`, `tarea19_resultados.npz`, `tarea19_output.log`, `tarea19_resultados.md`
- `RESUMEN_FINAL_RONDA6.md` (este archivo)
