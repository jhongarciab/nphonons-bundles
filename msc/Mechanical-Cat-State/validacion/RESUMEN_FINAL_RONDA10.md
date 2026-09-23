# Ronda 10 — Tareas 26-27: resumen

Código: `modelo_comun.py` (completo + efectivo), `tarea26_worker.py`, `tarea27_worker.py`, `run_t26_t27.sh`,
análisis `tarea26_27_analisis.py` → `tarea26_27_resultados.md` (tablas completas). Punto: (δ_m, Δ_q) = (0.048, 0.144).

## Tarea 26 — precisión de γ_bf
- γ_bf **converge** al apretar (atol,rtol) de (1e-12,1e-10) a (1e-14,1e-12) y (1e-15,1e-13): el cambio es ≤0.4% (p. ej. Γ₂/κ=0.13, α²=5: 2.803e-8 → 2.793e-8 → 2.793e-8; 2.07, α²=5: 4.629e-7 → 4.625e-7 → 4.625e-7). Los dos últimos coinciden a 4 cifras ⇒ **no hay piso numérico**; los valores de la Tarea 25 eran correctos (error ≤0.4%).
- U^10 (λ=−ln μ_10/(10 T_r)) reproduce el valor de un período a ≲1e-4 relativo.
- Resolución: γ_min ≈ rtol/T_r = 1.6e-8 (rtol=1e-10), 1.6e-10 (1e-12), 1.6e-11 (1e-13), con T_r=6.28e-3. Con la tolerancia base, Γ₂/κ=0.13 tiene γ_bf/γ_min = 1.8-7 (α²=5…3): al límite de resolución, aunque el sesgo observado es solo ~0.4%; con rtol=1e-13 la razón es ≥1700. Valores convergidos (rtol=1e-13, 1 período):
  | Γ₂/κ | α²=3 | α²=4 | α²=5 |
  |---|---|---|---|
  | 0.13 | 1.1127e-7 | 4.4681e-8 | 2.7930e-8 |
  | 2.07 | 1.8686e-6 | 8.7824e-7 | 4.6249e-7 |

## Tarea 27 — brecha robusta (5.º autovalor por Re λ ascendente, α²=2, Δ_2−=Δ_q en el efectivo)
- Los 3 modos lógicos (k=1..3) verificados por overlaps en las 9×2 corridas (uno de paridad, dos de tipo 'a'). Ojo: los overlaps usan solo P, n, a bosónicos, así que un modo de coherencia con parte de qubit puede tener overlap pequeño (visto en Γ₂/κ=0.13: 0.028); la verificación es débil para ese modo.
- γ_pf ≈ 7.5-7.8e-4 en todo el rango, completo y efectivo casi iguales.
- **Completo**: brecha 0.022 (0.01) → 0.062 (0.03) → 0.122 (0.07) → **máx 0.168 (0.13)** → 0.136 (0.24) → 0.123 (0.45) → 0.094 (0.84) → 0.093 (1.6) → **0.131 (3.0)**. Es decir **persiste el máximo en Γ₂/κ≈0.13 y existe la subida en 3.0** (mínimo ~0.093 en 0.84-1.6).
- Im λ del modo de la brecha: ≈0 (modo real) en 0.03, 0.07, 0.84, 1.6, 3.0; par complejo con |Im|≈7, 11, 18 en 0.13, 0.24, 0.45. El máximo en 0.13 coincide con el cambio de real a par complejo del quinto autovalor (posible cruce de modos), así que el "máximo" puede reflejar reordenamiento y no un solo modo continuo. Esto explica los modos con Im≈±7 que en la Tarea 25(b) se marcaron como espurios: eran este par.
- **Efectivo** (Δ_2−=Δ_q, sin δ₁): brecha monótona creciente 0.022 → 6.6, sin máximo ni subida propia; sobrestima al completo en 1-2 órdenes para Γ₂/κ ≳ 0.45. Solo concuerda (≲10%) hasta ~0.03.
