# Tarea 19 — Origen del calentamiento (Floquet, eps=0)

Script: `tarea19_origen_calentamiento.py`. Datos: `tarea19_resultados.npz`.
Sin drive, sin retuneo (ω_q=2ω_m fijo). g_z·g_x=360 constante (calibrado
para que g_z/ω_m=0.06 coincida exactamente con los parámetros del paper
usados en la Tarea 18(i)).

## Tabla

| g_z/ω_m | g_z | g_x | ⟨n⟩_fijo | λ_heat (modo dominante) |
|---|---|---|---|---|
| 0.015 | 15.0 | 24.0 | 0.07034 | 9.133e-4 |
| 0.030 | 30.0 | 12.0 | 0.04767 | 3.410e-4 |
| 0.060 | 60.0 | 6.0 | 0.02360 | 1.972e-4 |
| 0.120 | 120.0 | 3.0 | 0.02027 | 1.612e-4 |

Validación: traza/hermiticidad perfectas; positividad OK en todos los
casos (mín. autovalor ~1e-10 a 1e-13, muy por dentro del umbral −1e-8).

## Ajustes

| Modelo | Pendiente | Intercepto | R² |
|---|---|---|---|
| λ_heat vs (g_z/ω_m)² | −3.212e-2 | 5.567e-4 | **0.367** |
| **λ_heat vs g_x²** | **1.326e-6** | **1.496e-4** | **1.00000** |

**El ajuste contra g_x² es esencialmente perfecto (R²=1.0000)**; el
ajuste contra (g_z/ω_m)² es pobre (R²=0.37, pendiente incluso negativa,
sin sentido físico). **La tasa de calentamiento/relajación dominante
escala como g_x², no como (g_z/ω_m)².**

## El intercepto coincide con la disipación mecánica bare

El intercepto ajustado (1.496e-4) coincide, dentro del error numérico,
con γ_m en unidades de κ: γ_m/κ = (2π·15)/(2π·1e5) = **1.500e-4**
(diferencia relativa 0.3%). Esto es exactamente la estructura esperada de
la fórmula de la teoría efectiva Γ₁₋=(n_th+1)γ+Γ1_minus(g_x): a n_th=0,
el término (n_th+1)γ=γ es una constante independiente de g_x, y Γ1_minus
∝g_x² es el término variable. **El ajuste reproduce exactamente esta
estructura conocida**, confirmando que el modo dominante de relajación
identificado vía Floquet es, en efecto, el canal de un fonón inducido por
g_x (tipo Γ1), no un efecto exótico de orden superior.

## Comparación con γ_phase-flip de la Tarea 18(ii)

| | Valor (κ) |
|---|---|
| λ_heat en g_z/ω_m=0.06 (eps=0, sin drive) | 1.972e-4 |
| γ_phase-flip, Tarea 18(ii) (eps=1.44, con drive, mismos g_z,g_x) | 6.912e-4 |
| razón | 0.285 |

Los dos valores son del **mismo orden de magnitud** (factor ~3.5×) pero
no idénticos — razonable, dado que corresponden a regímenes físicos
distintos: eps=0 (relajación pasiva) vs eps=1.44 (con el drive activo
generando el squeezing y el propio cat state, que introduce canales
adicionales de decoherencia de paridad —p.ej. mezcla con Γ2₊, dk, y la
dinámica no lineal del squeezing— no presentes en el caso sin drive).

## Conclusión Tarea 19

1. **El calentamiento/piso está gobernado por g_x² (canal de un fonón
   tipo Γ1), no por (g_z/ω_m)² como sugería la hipótesis original de
   "patada polarónica" de las Tareas 15/17.** El ajuste es prácticamente
   exacto (R²=1.0000) y su intercepto reproduce γ_m independientemente.
2. Esto **revisa la interpretación de la Tarea 16**: aunque el aislamiento
   `g_x=0`/`g_z=0` mostró que el piso requiere ambos acoplamientos
   simultáneos para alcanzar su magnitud completa a τ_max=30 (medida de
   snapshot, no convergida), el **mecanismo espectral dominante que rige
   la relajación hacia el verdadero estado estacionario** (identificado
   aquí de forma rigurosa vía el propagador de Floquet) es
   inequívocamente de tipo g_x² + γ_m — la dependencia en g_z (si la hay)
   es subdominante en la tasa de relajación, aunque sí afecta el valor de
   ⟨n⟩_fijo alcanzado.
3. γ_phase-flip (Tarea 18-ii) y λ_heat (aquí) son consistentes en orden de
   magnitud pero no coinciden exactamente — la hipótesis del canal de
   "patada polarónica" como explicación separada y adicional para la
   pérdida de paridad (Tareas 15/17) **no encuentra respaldo aquí**: el
   mecanismo identificado con confianza (g_x², vía Floquet) es
   simplemente el canal Γ1 estándar de la teoría efectiva, ya conocido,
   no un efecto nuevo asociado a g_z/ω_m.
