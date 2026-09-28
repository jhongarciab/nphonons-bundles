# Pendientes, discrepancias y hallazgos (verificación independiente y figuras finales)

Registro consolidado de lo que requiere revisión o decisión, de los errores encontrados en el trabajo original y de las correcciones ya adoptadas.
Detalle en `../verificacion_independiente/V*_reporte.md`, `C6_polaron.md` y `README.md`. Actualizado: 2026-09-28.

## 1. Pendiente de decisión o revisión
| # | Tema | Estado | Dónde |
|---|---|---|---|
| P1 | **Regla universal (nueva Fig. 2):** el pico cae en \|x\| < 0.1 salvo ω = 8, κ₂/κ = 1 (x = +0.15, posible sesgo de la parábola por asimetría y paso 0.3). El FWHM depende de \|α\|² (≈3 para \|α\|² = 4, ≈3.7–4.2 para \|α\|² = 2) y la asimetría crece con κ₂/κ: **no hay colapso total de la forma**. | **Aceptado (colapso parcial)**; figura terminada | `data/principal_fig2_resumen.csv` |
| P2 | Curvas nuevas con \|α\|² = 4 calculadas con N = 16: subestiman P_max en ~4e-3 (N = 22 da 0.9983 frente a 0.9945). No afecta a los flancos ni al FWHM (1e-4). **Hecho:** recalculados con N = 22 (|x| ≤ 1). P_max 0.998–0.9999, salvo g_z/κ = 20 (0.9905; g_z/ω = 0.1). El pico de ω = 8, κ₂/κ = 1 sigue en x = +0.16. | **Resuelto** | README, nueva Fig. 2 |
| P3 | Panel (d) (gato transitorio): F máx = 0.741 en Γt = 16.9 (el trabajo decía ~0.72 en Γt ≈ 13; F(13) = 0.727, curva plana). Orden de paneles a, b, d, c. | Hoy en apéndice | `apendice_fig2_estados.py` |
| P4 | Déficit no adiabático de C1/C2: ~1% en κ₂/κ ≈ 0.2 y ~2% en 1. No tiene explicación analítica; solo se describe. Correlaciona con κ₂/κ (r = 0.89), no con P_e. | Abierto | Fig. 3, V4b |
| P5 | Desplazamiento polarónico óptimo en Ma: 2.4% menor que g_z/ω; el campo medio −g_z⟨σ_z⟩/ω explica un tercio. Efecto en P_c ≤ 1e-6. | Abierto, irrelevante en la práctica | `C6_polaron.md` |
| P6 | Exceso de +1.6% en V4 a g_z/κ pequeño: explicado por el término Γ₁⁺(\|α\|²+1) (V4b). En el punto (0.1, 5, g_z/κ = 2) el déficit residual es 0.7%, mayor que en sus vecinos. | Menor | Fig. 3 |
| P7 | Figs. 5 y 3 del plan nuevo (requisito térmico con plataformas; espacio de diseño) | Sin empezar | — |
| P9 | **Fig. 3 principal: el modelo completo confina más despacio que el modelo mínimo cuando g_x/ω crece.** conf_completo/Δ_mínimo ≈ 1.00–1.03 para g_x/ω ≲ 0.01; 0.93–0.86 para g_x/ω ≈ 0.014–0.02; 0.82 (0.017, κ₂/κ = 0.23); 0.64 (g_x/ω = 0.025, κ₂/κ = 0.16). Depende de g_x/ω más que de κ₂/κ: (0.05, 6, 12) con κ₂/κ = 0.16 da 0.99, y (0.1, 5, 2) con κ₂/κ = 0.026 da 0.88. En consecuencia ε_completo/ε_mapa sube hasta 1.55. **Causa confirmada: χn|e⟩⟨e|, χ = (8/3)g_x²/ω** (el modelo efectivo con χ reproduce el completo; al subir ω/κ a 500 y 1000 la desviación crece hasta la pérdida del gato). Umbral: χ\|α\|²/κ ≈ 0.3 (5%). Ma, con 2.7, queda fuera de la región cuantitativa. | **Resuelto**; curvas de validez en la Fig. 3 | `data/principal_fig3_puntos.csv` |
| P10 | **Figura central (b): κ₁ con filtro en régimen saturado.** Verificación con el modelo completo con filtro (κ_f = 0.3, ω/κ = 200). En (κ₂/κ, g_z/κ) = (0.03, 4), ε_completo/ε_mapa = 1.03 (bien). En (0.3, 12), 2.29: el confinamiento coincide (0.94 frente a 0.955 del mapa), pero **κ₁ completo es 2.26 veces el filtrado predicho**. Hipótesis (no verificada): un canal de paridad no filtrado. Con el desplazamiento polarónico ±g_z/ω, cada decaimiento real del qubit (a 2ω, resonante con el filtro) desplaza el oscilador en 2g_z/ω, con una tasa ~ κ·P_e·(2g_z/ω)²·O(\|α\|²). En el baño plano es despreciable frente a κ₁; con filtro, κ₁ cae ×1700 y este canal pasa a dominar. Escala con (g_z/ω)² y P_e: el punto saturado tiene g_z/ω = 0.06, frente a 0.02 del no saturado. Prueba propuesta: repetir el punto con g_z/ω menor (subiendo ω/κ) o medir la dependencia con g_z a κ₂/κ fijo. | **Abierto**: el panel (b) solo es fiable en el régimen no saturado; añadir el canal al mapa si se confirma | `data/figura_central_verificacion.csv` |
| P8 | Apagado automático del PC: sin permiso (polkit/sudo piden contraseña). Hace falta la regla de sudoers o apagar manualmente. | Acción del usuario | — |

## 2. Errores o discrepancias encontradas en el trabajo original
| # | Qué | Consecuencia |
|---|---|---|
| E1 | **Ma et al., Ec. (4):** omite el corrimiento del oscilador −(g_x²/ω)n y da 3g_x²/ω para el qubit, cuando en RWA sale g_x²/ω y en el cálculo completo 4g_x²/3ω. El término de pares es correcto. | Resonancia en 11.98, no en 12 (V1, V2) |
| E2 | **Fórmula de phase-flip:** faltaba el +1 del canal de ganancia: γ_pf = 2[Γ₁⁻\|α\|² + Γ₁⁺(\|α\|²+1)]. | +1.6% en V4; mejora con filtro subestimada un 4% (V5) |
| E3 | **Acuerdo del 0.05% en V3:** era una compensación casual (el +1 sobra un 2.5% y la saturación resta un 2%). Con la fórmula corregida la razón es 0.982. | Reportar V3 con la fórmula corregida |
| E4 | **"≤ 0.4%" de C3:** solo vale para κ₂/κ ≲ 0.05. | C3 reemplazado por una precisión continua en κ₂/κ |
| E5 | **Tasa de confinamiento (Tarea 43 y relacionadas):** el "5.º modo espectral" es un artefacto del truncamiento (\|Im λ\| crece con N y la tasa cambia 2.2%). La tasa dinámica es ~7.0e-3 en el baño plano, no 4.2e-3, y el filtro κ_f = 0.1 la reduce a ×0.53, no a ×0.23. | Fig. 4 de validación usa la tasa dinámica |
| E6 | **Código con α_eff:** no sirve para P_c (P_c = 0.967 en ω_p = 12 sin gato). | Código fijo D(g_z/ω)\|±α_nom⟩ |
| E7 | **Tarea 37:** 14 ρ fallan en hermiticidad antes de hermitizar (hasta 2e-9 > 1e-10); los puntos calientes no están convergidos (η cambia 16% entre N = 20 y 26). | Declarar al usar η(x) en la Fig. 5 |
| E8 | **Tolerancias de traza en T42/T43** excedidas por potencias del propagador (ya reportado por el propio trabajo). | Mis cálculos no usan potencias |

## 3. Correcciones adoptadas (C1–C9 y decisiones)
- **C1:** tasa de phase-flip con +1 y \|α\|² = \|α_eff²\| = \|⟨(a − g_z/ω)²⟩\|.
- **C2:** figura de mérito κ₁/κ₂ = (5/72)(κ/g_z)², con κ₁ = Γ₁⁻ + Γ₁⁺ (sin corrección).
- **C3 (revisado):** precisión ≲ 0.4% para κ₂/κ ≲ 0.05, ~1% en 0.2 y ~2% en 1, presentada como corrección no adiabática.
- **C6:** código fijo polarónico. Marco de laboratorio con t = nT_p y desplazamiento +g_z/ω.
- **C9:** confinamiento dinámico en lugar de la brecha espectral.
- **Interpolación:** PCHIP para los FWHM.

## 4. Limitaciones de los cálculos propios
- Convergencia con filtro solo N = 16 → 18 (N = 20 no cabe en 14 GB).
- La regla universal usa N = 14–16 en las curvas nuevas (ver P2).
- Fig. 2(a) de validación con N = 22; la primera versión con N = 20 subestimaba P_c en 1.4e-4.
