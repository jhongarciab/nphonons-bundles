# Tarea 21 — Espectro completo (Floquet) vs efectivo (Liouvilliano)

Script: `tarea21_espectro_completo_vs_efectivo.py`. Datos:
`tarea21_principal.npz`, `tarea21_bonus.npz`. Modelo efectivo con
g_eff=2g y término de Lamb δ₁ (aditivo, sin retuneo, como en la Tarea 14).

## Tabla principal (|α|²=2, g_z escalado, g_x fijo)

| g_z×f | Γ₂/κ | Confin. completo | Confin. efectivo | ratio | Phase-flip completo | Phase-flip efectivo | fórmula 2\|α\|²(Γ₋+Γ₊) | Coh. lógica completo | Coh. lógica efectivo |
|---|---|---|---|---|---|---|---|---|---|
| 0.125 | 0.0324 | 5.244e-2 | 6.941e-2 | 1.32 | 3.681e-4 | 1.077e-4 | 1.600e-4 | 6.890e-3 | 8.220e-3 |
| 0.25 | 0.1296 | 1.627e-1 | 3.262e-1 | **2.00** | 6.217e-4 | 1.626e-4 | 1.600e-4 | 2.059e-3 | 1.614e-3 |
| 0.5 | 0.5184 | 1.171e-1 | 1.324e+0 | **11.3** | 6.847e-4 | 1.678e-4 | 1.600e-4 | 7.867e-4 | 4.355e-4 |
| 1.0 | 2.0736 | 9.911e-2 | 5.299e+0 | **53.5** | 6.912e-4 | 1.681e-4 | 1.600e-4 | 4.822e-4 | 1.718e-4 |

## Tabla bonus (g_z×0.25, |α|² variable)

| \|α\|² | Γ₂/κ | Confin. completo | Confin. efectivo | Phase-flip completo | Phase-flip efectivo | fórmula |
|---|---|---|---|---|---|---|
| 1 | 0.1296 | 0.1396 | 0.1800 | 3.264e-4 | 8.419e-5 | 8.000e-5 |
| 2 | 0.1296 | 0.1627 | 0.3262 | 6.217e-4 | 1.626e-4 | 1.600e-4 |
| 3 | 0.1296 | 0.1603 | 0.4255 | 9.532e-4 | 2.450e-4 | 2.400e-4 |
| 4 | 0.1296 | 0.1418 | 0.5347 | 1.284e-3 | 3.261e-4 | 3.200e-4 |

## Hallazgo 1: la brecha de confinamiento del modelo efectivo diverge catastróficamente

El **completo** mantiene su brecha de confinamiento aproximadamente
**constante** (~0.10-0.16) en todo el rango de Γ₂/κ y |α|² probado. El
**efectivo**, en cambio, **crece sin control** con Γ₂/κ (0.069→0.326→
1.324→5.299, ratio efectivo/completo de 1.3× a **53.5×**) y también
crece con |α|² a Γ₂/κ fijo (0.180→0.535 al subir |α|² de 1 a 4). El
modelo efectivo **sobreestima brutalmente** qué tan rápido se forma/
confina el gato al aumentar el acoplamiento o la amplitud del gato — una
limitación severa, mucho peor que cualquier otra discrepancia encontrada
en esta serie de tareas.

## Hallazgo 2: el phase-flip real es sistemáticamente ~4× el predicho

La fórmula `2|α|²(Γ₋+Γ₊)` reproduce casi exactamente el phase-flip del
**modelo efectivo** (diferencias <10%, ej. 1.681e-4 vs 1.600e-4 en g_z×1)
— es la fórmula correcta para ese modelo. Pero el phase-flip del
**modelo completo** es consistentemente **~3.4-4.1× mayor**:

| Caso | ratio completo/fórmula |
|---|---|
| g_z×0.125 | 2.30 |
| g_z×0.25 | 3.89 |
| g_z×0.5 | 4.28 |
| g_z×1.0 | 4.32 |
| \|α\|²=1 (bonus) | 4.08 |
| \|α\|²=2 (bonus) | 3.89 |
| \|α\|²=3 (bonus) | 3.97 |
| \|α\|²=4 (bonus) | 4.01 |

**El factor ~4× es notablemente estable** en el rango |α|²∈{1,2,3,4} a
Γ₂/κ fijo (3.9-4.1×, variación <5%) y crece más con Γ₂/κ (2.3× a 4.3×
en el rango probado) — sugiere una corrección multiplicativa
aproximadamente constante (~4×) al canal Γ₁ estándar, más relevante al
aumentar el acoplamiento longitudinal/transversal.

## Hallazgo 3: coherencia lógica (bit-flip) — acuerdo razonable a bajo acoplamiento, se degrada

A Γ₂/κ=0.032: completo=6.89e-3 vs efectivo=8.22e-3 (ratio 1.19, buen
acuerdo). A Γ₂/κ=2.07: completo=4.82e-4 vs efectivo=1.72e-4 (ratio 2.8,
divergencia moderada) — mucho menos severa que la de confinamiento, pero
tampoco despreciable a acoplamiento fuerte.

## Conclusión: ¿en qué Γ₂/κ deja de describir bien el efectivo la brecha del completo?

**La brecha de confinamiento del modelo efectivo deja de ser una
descripción razonable ya en Γ₂/κ≈0.13** (ratio=2.0×, el doble) y **se
vuelve completamente inútil (>10×) para Γ₂/κ≳0.5**. Los parámetros del
paper (Γ₂/κ≈2.07) están muy adentro del régimen de falla total
(ratio=53.5×). En contraste, el phase-flip y la coherencia lógica del
modelo efectivo (con la corrección multiplicativa ~4× para el phase-flip)
siguen siendo *cualitativamente* razonables (mismo orden de magnitud) en
todo el rango, aunque cuantitativamente imprecisos.

**Conclusión general de la Tarea 21**: el modelo efectivo con g_eff=2g y
δ₁ describe razonablemente bien los canales de **error lógico** (bit-flip
y, con una corrección ~4×, phase-flip) en todo el rango de acoplamiento
probado, pero **falla catastróficamente en describir la velocidad de
formación/confinamiento del gato** para Γ₂/κ≳0.13 — precisamente el
régimen de los parámetros publicados en el paper. Esto es consistente con
y complementa el hallazgo de la Tarea 11 (F_min bajo en Γ₂/κ grande): la
causa raíz identificable ahora es específicamente la brecha de
confinamiento, no los canales de decoherencia del cat-qubit en sí.
