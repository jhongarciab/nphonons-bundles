# Ronda 13 — Tareas 37-38: baños térmicos consistentes y auditoría de la brecha

Tablas completas: `tarea37_38_resultados.md`; figuras: `tarea37_eta_mapa.png`, `tarea38_auditoria.png`.
Código: `tarea37_worker.py`, `tarea37_check.py`, `audit_worker.py`, `tarea37_38_analisis.py`, `run_t37_38.sh`
(`modelo_comun.full_propagator` ahora acepta `nm` y `qcons`).

## Tarea 38 — qué cambia en las Tareas 21, 23, 27, 28
Criterio: modo espurio si su peso en n>N−6 supera 1e-3, o si no tiene pareja estable (Re±2%, Im±2%+0.05) al pasar N=20→26.
**Todos los máximos, mínimos y subidas de la "brecha completa" para Γ₂/κ ≳ 0.13 eran modos del borde de Fock.** La brecha física es monótona y satura en ~0.23:
| Γ₂/κ | T23 original | T23 física | T27 original | T27 física |
|---|---|---|---|---|
| 0.13 | 0.163 | 0.163 | 0.168 (Im −7) | 0.168 (Im −0.3) |
| 0.24 | 0.155 | 0.192 | 0.136 | 0.192 |
| 0.45 | 0.129 | 0.208 | 0.123 | 0.208 |
| 0.84-0.85 | 0.091 | 0.220 | 0.094 | 0.220 |
| 1.6 | 0.089 | 0.228 | 0.093 | 0.228 |
| 3.0 | 0.124 | 0.234 | 0.131 | 0.234 |
- **Tarea 21**: la divergencia de la brecha del efectivo frente al completo sigue siendo cierta, pero el "53.5×" en Γ₂/κ=2.07 sale de un modo de borde; con la brecha física (0.231) la razón es ~23×.
- **Tarea 23**: **se retira el "óptimo Γ₂/κ≈0.126 con brecha 0.161", la no monotonía y la subida en Γ₂/κ=3 (y el ajuste A·x/(1+Bx²))**. La brecha física crece monótonamente y satura en ~0.23-0.24 (sin máximo).
- **Tarea 27**: se retira el máximo en 0.13, la caída y la subida en 3.0; la brecha física es monótona (0.022 → 0.234). El par con Im≈7-18 era edge (Fock).
- **Tarea 28**: 16 de 30 puntos tienen el 5º modo espurio (desde Γ₂/κ=0.132); "no hay coalescencia" se mantiene; la brecha física es monótona (0.043→0.222 en Γ₂/κ=1). Hasta 7 de los 12 modos más lentos por punto son de borde.
- **Sin cambio**: γ_pf y γ_bf (Tareas 22, 25, 26, 34, 36), ubicación de la resonancia vestida (Tarea 24), la escalera estática (31-32) y la conclusión de Tarea 35(c). Nota: la brecha física del Floquet completo (0.222 en Γ₂/κ=1) coincide con la del modelo estático explícito (0.234) al 5%.

## Tarea 37 — baños consistentes (Γ₂/κ=0.13; n_q=1/(e^{2hf/kT}−1), n_m=1/(e^{hf/kT}−1))
Solo cuenta hf/kT (las tres celdas (0.1,10), (0.2,20), (0.5,50) son idénticas, como debe ser). Sesgo η=γ_pf/γ_bf:
| f_m (GHz) \ T | 10 mK | 20 mK | 50 mK |
|---|---|---|---|
| 0.5 | 2.7 / 6.9 | 0.58 / 1.6 | 0.49 / 0.60 |
| 1.0 | 200 / 591 | 2.7 / 6.9 | 0.55 / 1.4 |
| 1.5 | 1.1e3 / 8.9e3 | 23 / 60 | 0.67 / 2.0 |
| 2.0 | 1.2e3 / 1.0e4 | 200 / 591 | 1.3 / 3.4 |
(η para α²=2 / α²=3.) η≥100 solo con hf/kT ≳ 4.8 (f=1 GHz a 10 mK, o 2 GHz a 20 mK); a 50 mK el sesgo se pierde en todo el rango, con P(código) ≤ 0.96 (α²=2). El punto más frío (2 GHz, 10 mK, n_q=4.6e-9, n_m=6.8e-5) da η=1.2e3 (α²=2) y 1.0e4 (α²=3), P(código)=0.9998.
- **Verificación del modo bit-flip** (|α⟩|g⟩ propagado por U^n, ajuste de ⟨sgn x⟩(t)): en frío la tasa ajustada coincide con γ_bf espectral a 1e-3 (α²=2) y 1e-4 (α²=3); en el punto intermedio (0.5 GHz, 20 mK) difieren 19% (α²=2) y 3% (α²=3); en el más caliente 24-26%. En frío el modo etiquetado es el bit-flip; en caliente el decaimiento de ⟨sgn x⟩ deja de ser una sola exponencial (mezcla con otros modos), así que η ahí es cualitativo.
- **Convergencia en N en el punto más caliente** (0.1 GHz, 50 mK): γ_pf 3-5%, γ_bf 17-18%, |⟨a²⟩| 40-58%, P(código) 5-8%: **los puntos con n_q ≳ 0.3 no están convergidos**; los fríos (η alto) sí.
- **Validaciones de ρ**: traza y autovalores pasan (mín ≥ −6e-13). **14 puntos fallan la hermiticidad antes de hermitizar** (1e-10 a 3e-9, todos en el lado frío/moderado, n_q ≲ 0.03, por el bit-flip casi degenerado y la precisión del propagador); los calientes cumplen (≲2e-11).
