# Tarea 18 — Floquet: propagador de un período y estado asintótico exacto

Script: `tarea18_floquet.py`. Datos: `tarea18_resultados.npz`.
`propagator(H, T_m, c_ops, atol=1e-12, rtol=1e-10)`. Muy rápido: caso (i)
(Nb=16) ~20s, caso (ii) (Nb=20) ~50s — el cómputo del propagador de UN
período es barato aunque el espacio de Liouville tenga dimensión 1024-1600.

## Caso (i): eps=0, parámetros del paper (Nb=16)

Punto fijo validado: tr=1.000000, herm_err=0, mín. autovalor=−4.6e-13 (perfecto).

| Cantidad | Valor |
|---|---|
| ⟨n⟩ del punto fijo (Floquet, exacto) | **0.023602** |
| (g_z/ω_m)² (predicción "relajación") | 0.003600 |
| Piso medido en Tarea 16 (τ_max=30, no convergido) | 0.006190 |
| razón n_fijo / (g_z/ω_m)² | 6.556 |
| razón n_fijo / piso T16 | 3.813 |

**Diagnóstico: CALENTAMIENTO con estacionario caliente.** ⟨n⟩ del punto
fijo exacto (0.0236) es **6.6× mayor** que la predicción de "relajación
al estado base vestido" (g_z/ω_m)²=0.0036. Además, es **3.8× mayor** que
el piso ya medido en la Tarea 16 (0.0062 a τ_max=30) — esto confirma que
la Tarea 16 **no había llegado al verdadero estado estacionario**: el
sistema sigue calentándose lentamente más allá de τ_max=30 (consistente
con la pendiente positiva medida allí, 2.97e-4, que a ese ritmo tardaría
~60 unidades adicionales en cerrar la brecha hasta 0.0236 — del orden de
magnitud correcto).

## Caso (ii): eps=1.44 (punto del paper, Nb=20)

Punto fijo validado: tr=1.000000, herm_err=0, mín. autovalor=−1.4e-12 (perfecto).

| Cantidad | Valor |
|---|---|
| ⟨n⟩... (ver \|⟨a²⟩\|) | — |
| **Paridad del punto fijo (oscilador reducido)** | **0.0524** |
| \|⟨a²⟩\| | 1.9535 |

**Hallazgo mayor: la paridad verdadera en el estado estacionario exacto
es 0.052 — MUY lejos del 0.87 reportado en la Tarea 12** (que usaba
τ_max=60, sin llegar al verdadero estacionario). El motivo: existe un
modo de relajación **extremadamente lento**:

| Modo k | μ_k | \|overlap con paridad\| | λ_k (=−ln μ_k/T_m) |
|---|---|---|---|
| 1,2 | 0.999997±0.000044i | 6.5e-3 (chico) | 4.82e-4 |
| **3** | **0.999996** | **1.404 (dominante)** | **6.912e-4** |
| 4 | 0.999377 | 0.119 | 9.911e-2 |
| 5 | 0.898±0.437i | 9.0e-7 (nulo) | 0.158 |

El modo 3 domina por mucho la dinámica de la paridad (overlap 1.40,
un orden de magnitud sobre el resto), con **λ₃=6.912e-4 (unidades κ)** —
tiempo de relajación **~1447 κ⁻¹**, ¡24× más largo que el τ_max=60 usado
en toda la Ronda 3! Esto explica por qué las Tareas 6a/6b/9b/10/12
—todas con τ_max≤60— midieron un "estacionario" que en realidad es un
**plateau transitorio de larga vida**, no el verdadero punto fijo.

**gamma_phase-flip = λ₃ = 6.912e-4 (unidades de κ)**, identificado por
proyección de los autovectores del propagador sobre el operador de
paridad (criterio: mayor \|Tr[P†X_k]\|).

## Conclusión Tarea 18

1. **El piso en ⟨n⟩ es calentamiento genuino hacia un estacionario
   caliente**, no relajación a un estado base vestido de baja ocupación
   — confirmado con el punto fijo exacto de Floquet, sin ambigüedad de
   convergencia.
2. **Las cifras de "estado estacionario" de rondas anteriores (Tareas
   6a, 6b, 9b, 10, 12) subestiman sistemáticamente el tiempo de
   relajación real**: existe un modo de decaimiento de paridad con
   λ≈6.9e-4κ (τ≈1447κ⁻¹), no capturado por τ_max≤60. La paridad
   verdadera en el punto del paper (ε=1.44) es **0.052**, no 0.87 —
   revisión sustancial de la conclusión de la Tarea 12.
3. Esto no invalida la normalización g_eff=2g (que depende de la
   dinámica de corto plazo, ya bien establecida en Tareas 7-8), pero sí
   implica que la calidad del cat state generado en tiempos
   experimentalmente razonables (τ~10-60 κ⁻¹) es mucho mejor que la que
   tendría el sistema si se dejara relajar por completo — es decir, en
   la práctica interesa el **plateau transitorio**, no el verdadero
   estado estacionario a tiempos κt≫1000.
