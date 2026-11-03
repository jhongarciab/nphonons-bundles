# Dossier ruta 3 — fuente de pares de fonones (transmón con escalera |f⟩ ↔ modo mecánico)

Estado: simulación y verificación hechas; **no** es un paquete "solo falta escribir". Etiquetas: [V] verificado por reimplementación independiente, [S] simulado (modelo propio), [A] supuesto no medido.

## 1. Mecanismo
- M1: Jaynes–Cummings de dos fonones a partir de H = ω a†a + (ω_q/2)σ_z + (a+a†)(g_xσ_x + g_zσ_z); G = 2 g_x g_z/ω (con reducción ~e^{-2λ²} si hay acoplo longitudinal fuerte). [V]
- M2: variante con escalera de transmón: |f,0⟩ → |g,2⟩ vía virtual |e,1⟩; resonancia ω_q = ω − α/2; V = ¼[√(α²/4+16g²) − |α|/2] (4g²/|α| sobreestima 14 %). [V]
- Escalado por fonón extra: V₃/V₂ ≈ 0.18, V₄/V₃ ≈ 0.10 (g/|α| ≈ 0.1–0.2). Con parámetros publicados: V₂=5.74, V₃=1.04, V₄=0.105 MHz. [V]

## 2. Resultados cerrados
- Eficiencia de par: η = 1/[1 + (κ_q/4G²)(κ_m + κ_q/2 + 2G²/κ_m)]; óptimo κ_m* = √2 G; η ≥ 0.95 ⇔ κ_q ≤ 0.0735 G. Closed form vs integración exacta: 1e-16 (fig3). [V]
- Pureza de cluster exacta por renovación. [V]

## 3. Simulación pulsada realista (parámetros de Arrangoiz-Arriola 1902.04681: g/2π=15.2 MHz, α/2π=−138 MHz, T1=1.14 µs, κ_m/2π=0.37 MHz medido)
- P(2) ≈ 0.72 ± 0.05 con parámetros publicados **asumiendo** una guía de onda con κ_m≈10 MHz (el dispositivo medido no la tiene). [S][A]
- P(2) ≈ 0.93–0.95 con parámetros mejorados (presupuesto de error en fig4). [S]
- Flujo: limitado por reset del qubit; activo vs pasivo en fig4, suponiendo detección ideal (escala η_det²). [S][A]
- N=3 no viable con parámetros publicados (V₃ ~1 MHz vs decoherencia). [S]

## 4. Novedad (informe_3.md)
Reclamable: instancia fonónica concreta, criterio de diseño en forma cerrada (κ_m*, κ_q/G), presupuesto realista, comparación con Bin et al. a igual λ/ω y Ω/ω.
No reclamar: "λⁿ como límite fundamental"; JC de dos fonones para gatos (Naseem), bundles de dos fotones y transición de doble excitación en escalera son conocidos. Citar Qiao 2503.03898 (inverso temporal), Bin PRL 124 y PRL 127, Bin–Wu–Lü.

## 5. No verificado / abierto
- Signo de la corrección del modelo completo a η: V obtiene −0.3…−2.5 pp, E obtuvo +0.2…+0.4 pp. Sin resolver; el cierre analítico es válido solo a nivel de modelo efectivo.
- T1(|f⟩), ocupación térmica, población p_e y κ_m con guía: no medidos [A].
- Detección: η_det no modelada en el flujo.

## 6. Registro de errores corregidos
Mal-lectura Hitchcock (0.42±0.13), factor 2χ, factor 2 en Tφ, métrica condicional vs incondicional, censura en first_cross, selección de autovalores en Vsplit, fórmula eta_num, `\Big`/`\sqrt2` en mathtext.

## 7. Reproducibilidad
`rutas/figs/pair_figs.py` (pair_fig1–4), `rutas/P/codigo/*`, `rutas/V/*`, `rutas/E/codigo/*`. Usar `OMP_NUM_THREADS=1`.

## 8. Evaluación honesta
Ruta 3 es un trabajo incremental de calidad moderada: mecanismo y fórmulas correctos y verificados, pero la novedad es de instancia + criterio de diseño y el rendimiento depende de una guía de onda no demostrada en el dispositivo de referencia. No asigno probabilidades de aceptación (mis estimaciones previas no estaban calibradas). Encaje más plausible: AQT o PRApplied/PRA como "propuesta de arquitectura con presupuesto de error"; riesgo principal: revisores pidiendo demostración de κ_m y T1(|f⟩).
Alternativa/complemento: reexaminar la molécula de dos QDs (carpeta `bsc`) con ME vestida en vez de local (la local da nube espuria de fonones κλ²/γ).
