# Tarea 17 — Patada polarónica, versión limpia (tasas, no valores finales)

Script: `tarea17_patada_limpia.py`. Datos: `tarea17_resultados.npz`.
Qubit BASE físico, ε/g_eff=2, g_eff=0.36 fijo, ω_q=ω_d=2(ω_m+δ₁)
(compensación retuneando la resonancia), δ₁·a†a incluido en H_eff.
τ_max=40/Γ₂≈77.16, ventana de ajuste [15/Γ₂, 40/Γ₂]=[28.94, 77.16].

## Tabla

| g_z/ω_m | δ₁ | −dP/dt (completo) | −dP/dt (efectivo) | diff | \|⟨a²⟩\|_full | \|⟨a²⟩\|_eff |
|---|---|---|---|---|---|---|
| 0.015 | −0.1920 | 1.158e-3 | 8.280e-4 | 3.296e-4 | 1.997 | **2.617** |
| 0.030 | −0.0480 | 3.584e-4 | 1.640e-4 | 1.943e-4 | 1.993 | 2.037 |
| 0.060 | −0.0120 | 1.104e-3 | 4.052e-5 | 1.064e-3 | 1.952 | 2.002 |
| 0.120 | −0.0030 | 5.535e-4 | 1.014e-5 | 5.434e-4 | 1.811 | 2.000 |

## Validación

Traza/hermiticidad perfectas en todos los casos. Positividad: **falla en
g_z/ω_m=0.06** en los instantes intermedio y final (mín. autovalor
≈−6.8e-8, fuera del umbral −1e-8) — el resto de los casos pasan limpio
(~1e-9 o mejor).

## Resultado: NO concluyente

**La serie "diff" NO es monótona en (g_z/ω_m)²** como predice la
hipótesis: 3.30e-4 → 1.94e-4 (¡baja!) → 1.06e-3 (salta) → 5.43e-4 (baja
de nuevo). El ajuste por mínimos cuadrados da:

    diff = 0.0132 (±0.040) × (g_z/ω_m)² + 4.70e-4 (±3.0e-4)      R² = 0.052

**R²=0.052 — prácticamente sin correlación.** El criterio automático del
script ("intercepto ~0 si │intercepto│<2×error") se cumplió técnicamente
solo porque el error del ajuste es enorme (la pendiente tiene un error
relativo >300%), no porque el ajuste sea bueno. **Se contradice el
veredicto automático impreso por el script**: no se puede afirmar que el
canal de patada polarónica quede confirmado con estos datos.

## Posibles causas del ruido

1. **g_z/ω_m=0.06 falla validación de positividad** — indica dificultad
   numérica específica de ese punto que probablemente contamina su valor
   de "diff" (el más alto y más discordante de la serie).
2. **\|⟨a²⟩\|_eff=2.617 en g_z/ω_m=0.015** (30% por encima del objetivo
   2.0) — con g_x=12 (el más grande de los 4 casos, ya que g_x=g_eff·ω_m/
   (2g_z) crece al bajar g_z), el modelo efectivo mismo no parece haberse
   estabilizado bien en la ventana usada para ese caso, sugiriendo que la
   ventana [15/Γ₂,40/Γ₂] —fija en unidades de Γ₂, que no depende de
   g_z— podría no ser suficiente tiempo de relajación cuando δ₁ (que sí
   depende de g_x²) es grande.
3. La compensación retuneando ω_q=ω_d=2(ω_m+δ₁) no se verificó de forma
   independiente (a diferencia del término aditivo de la Tarea 14, que sí
   dio una mejora clara y consistente) — es posible que la implementación
   de esta variante tenga un problema no diagnosticado.

## Conclusión Tarea 17

**El experimento, tal como se ejecutó, no permite confirmar ni descartar
la hipótesis del canal de patada polarónica** (que la Tarea 15 había
propuesto para explicar el exceso de pérdida de paridad sobre
(2g_z/ω_m)²). Los datos son demasiado ruidosos/no-monótonos (R²=0.05) y
un punto falla la validación de positividad. Se recomienda, antes de
sacar conclusiones sobre esta hipótesis específica: (i) investigar y
corregir la falla de positividad en g_z/ω_m=0.06 (probablemente requiere
tolerancias más estrictas o Nb mayor en ese punto), (ii) verificar que la
ventana de estabilización sea suficiente para cada g_x (quizás escalar la
ventana con 1/δ₁ además de con 1/Γ₂), y (iii) repetir con más puntos y
promediando sobre semillas/ventanas para reducir el ruido antes de
reintentar el ajuste. La cuestión de si el "piso residual" físico
identificado en la Tarea 16 (efecto cruzado g_z·g_x) es exactamente el
mismo mecanismo que el exceso de pérdida de paridad de la Tarea 15 queda
**sin resolver**.
