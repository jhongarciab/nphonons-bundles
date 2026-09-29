# Tarea 16 — ¿El piso es numérico?

Script: `tarea16_piso_numerico.py`. Datos: `tarea16_resultados.npz`.
Qubit BASE físico, eps=0, τ_max=30, estroboscópico.

## Parte 1: escaneo de tolerancia y Nb

| Nb | atol | rtol | ⟨n⟩ final | pendiente (2da mitad) |
|---|---|---|---|---|
| 16 | 1e-10 | 1e-8 | 6.19667e-3 | 2.9743e-4 |
| 16 | 1e-12 | 1e-10 | 6.19013e-3 | 2.9721e-4 |
| 16 | 1e-14 | 1e-12 | 6.19010e-3 | 2.9721e-4 |
| 24 | 1e-10 | 1e-8 | 6.19694e-3 | 2.9743e-4 |
| 24 | 1e-12 | 1e-10 | 6.19014e-3 | 2.9721e-4 |
| 24 | 1e-14 | 1e-12 | 6.19010e-3 | 2.9721e-4 |

**Variación relativa de la pendiente entre las 6 configuraciones: 0.08%.**
La pendiente y el valor final son **esencialmente idénticos** al apretar
la tolerancia 4 órdenes de magnitud (1e-10→1e-14) y al aumentar Nb de 16
a 24. Ya convergido con la tolerancia más floja probada.

**Diagnóstico: EL PISO ES FÍSICO**, no un artefacto numérico ni de
truncamiento de Fock.

## Parte 2: aislar el término causante

| Caso | ⟨n⟩ final | pendiente |
|---|---|---|
| Solo g_z (longitudinal) | 1.765e-8 | 8.96e-10 |
| Solo g_x (transversal) | 1.270e-4 | 3.98e-6 |
| **Completo (g_z y g_x)** | **6.190e-3** | **2.972e-4** |

**Ninguno de los dos términos por separado reproduce el piso**: g_z solo
da un efecto despreciable (~1.8e-8, esencialmente cero); g_x solo da algo
pequeño (1.27e-4) pero **49× menor** que el completo. El completo
(6.19e-3) es **casi 49× mayor que la suma de los dos aislados**
(1.27e-4+1.8e-8≈1.27e-4). **El piso es un efecto de término cruzado**:
requiere g_z Y g_x simultáneamente, consistente con un proceso de dos
fonones genuino (∝g_z·g_x, la misma combinación que define g=g_z g_x/ω_m
en el código).

## Comparación con la fórmula perturbativa Γ₂₊

La tasa de calentamiento de dos fonones predicha por la eliminación
adiabática, Γ₂₊=2g²ReS₂₊ con g=0.36, ReS₂₊=x/(x²+(4ω_m)²)≈3.13e-8, da
Γ₂₊≈8.1e-9 — **~5 órdenes de magnitud menor** que la pendiente medida
(2.97e-4). La fórmula perturbativa de Γ₂₊ **subestima drásticamente** el
calentamiento real en este régimen (g_x=6κ, g_z=60κ, ambos ≫κ, fuera de
la validez estricta de la eliminación adiabática a orden más bajo) —
consistente con los excesos ya vistos en la Tarea 7 (corrimiento tipo
Bloch-Siegert), Tarea 14 (necesidad del término de Lamb) y Tarea 15
(exceso de pérdida de paridad sobre la predicción ingenua).

## Conclusión Tarea 16

El piso residual en ⟨n⟩ (identificado en las Tareas 9, 13, 15) es un
**efecto físico real**, no numérico, y se origina en el **acoplamiento
cruzado g_z·g_x** (el mismo que da lugar al proceso de dos fonones), pero
su magnitud es mucho mayor que la que predice la fórmula perturbativa
Γ₂₊ de la eliminación adiabática a orden más bajo — evidencia adicional
de que, en el régimen de parámetros del paper (g_x, g_z ≫ κ), las
fórmulas de tasas de la teoría efectiva de orden más bajo subestiman
sistemáticamente varios efectos (calentamiento pasivo, corrimientos de
fase, pérdida de paridad), aunque la normalización central g_eff=2g
(Tareas 7-8) siga siendo correcta para el término de squeezing dominante.
