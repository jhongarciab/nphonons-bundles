# Informe scout C: minería de huecos y propuesta de 5 ideas (PRA / PRB / AQT)

Etiquetas: [TEXTO] = leído en PDF descargado (arXiv); [RESUMEN] = solo abstract/metadatos de la API de arXiv; [NO VERIFICADO] = no leído, solo inferencia. Revistas tomadas del campo `journal_ref` de arXiv (APS/Wiley inaccesibles). Nº de figuras = conteo automático de pies "FIG." en el PDF arXiv (aproximado). Todo el código está en esta carpeta (`proto1_cat.py`, `proto1b_kerr.py`, `proto2_reset.py`, `proto2_map.png`); PDFs en `pdf/`.

## 1. Panorama de artículos (2025–2026)

| id | revista (journal_ref) | tipo de contribución | págs / fig. | problema abierto / trabajo futuro declarado | lectura |
|---|---|---|---|---|---|
| 2602.22205 | PRA 113, 053530 (2026) | Análisis conceptual: EP Liouvilliano (LEP) vs Hamiltoniano (HEP) en optomecánica con fonones térmicos; fórmula exacta de la posición del EP híbrido (formalismo thermofield) | 12 / 4-5 | Realización con medida débil y post-selección de trayectorias; sonda de n_th | [TEXTO] concl. |
| 2508.00353 | PRA 113, 013737 (2026) | Propuesta + solución analítica "DCE paramétrico" en régimen de disipación invertida; acuerdo analítico-numérico | 21 / ~7 | Control con T de criostato, profundidad de modulación | [TEXTO] concl. |
| 2509.24295 | PRA 113, 033707 (2026) | Esquema realizable de squeezing de magnones (3.7 dB), cerca de punto crítico; apéndices de fase y RWA | 8 / 5-6 | Ninguno explícito; "parámetros experimentales actuales" | [TEXTO] concl. |
| 2504.09695 | PRA 113, 063713 (2026) | Estudio exhaustivo (bound states dressed, WGM+átomo+guía); Markov vs no-Markov | 18 / ~13 | Extender a multiexcitación y a más nanopartículas | [TEXTO] concl. |
| 2502.08350 | PRA 111, 053517 (2025) | Aprendizaje por refuerzo para estados mecánicos no clásicos | 14 / ~5 | Verificación completa de entrelazamiento (witnesses) | [TEXTO] concl. |
| 2604.08655 | PR Applied 26, L031005 (2026) | Experimento+teoría breve: reset de transmón con resonador acústico multimodo (HBAR) | 9 / 4 | Acoplo más fuerte vs. interacción residual; vaciar niveles superiores del transmón con otros modos | [TEXTO] discusión |
| 2511.19128 | PR Research 8, 033034 (2026) | Teoría cascada cuántica: detección dispersiva de fotón de microondas con DQD | 23 / ~3 | Esquemas con inmunidad parcial a fluctuaciones térmicas | [TEXTO] concl. |
| 2512.06219 | PR Research 8, 033169 (2026) | Código gato en ensamble de qutrits V (estados oscuros), inmunidad a decaimiento | 21 / ~2 | Efecto de desintonía entre niveles excitados | [TEXTO] concl. |
| 2603.25264 | PRA 114, 032212 (2026) | Transferencia de estado en resonadores multimodo (pitch-and-catch), puramente numérico | 8 / 2 | Comprensión analítica del cruce monomodo-multimodo; ultrafuerte | [TEXTO] concl. |
| 2504.12677 | PRA 112, 043713 (2025) | Estados oscuros de muchos cuerpos en wQED; solución analítica por simetría | 9 / 6 | Limitada a simetría alta; geometría arbitraria | [TEXTO] concl. |
| 2606.10319 | Adv. Quantum Technol. 9, e70324 (2026) | Bloqueo de fotón no recíproco: condiciones ópticas de control (interferencia), mapas 2D y curvas analíticas | 10 / ~6 | Rutas a enrutado/aislamiento cuánticos | [TEXTO] concl. (+ figuras vistas) |
| 2509.21569 | PRL 136, 080401 (2026) | Correlaciones de N fotones con baño fonónico estructurado (QD), más allá del QRT | 10 / 2 (+SM) | Incorporar transformación de polarón (fuerte acoplo) | [TEXTO] concl. |
| 2508.10500 | PRA 113, 013732 (2026) | Gatos mecánicos por ingeniería de reservorio con un qubit conducido; eliminación adiabática (Nakajima-Zwanzig) | 23 / ~2 (Wigner) | Varios resonadores con un qubit común; (sin análisis térmico/Kerr sistemático) | [TEXTO] concl. |
| 2608.05503 | sin journal_ref (ago 2026) | Gatos mecánicos de dos modos con fase de hopping, optomecánica con disipación | 10 / – | Ruido térmico en entrelazamiento estacionario | [TEXTO] concl. |
| 2606.25511 | sin journal_ref | Gato de dos magnones vía qubit; proyección | 9 / – | Dark-matter axion sensing | [TEXTO] concl. |
| 2601.18894 | PR Research 8, 023336 (2026) | Escalado de difusión en preparación disipativa (SSH) | 33 / ~12 | Mínimos locales/complejidad | [TEXTO] concl. |
| 2602.18153 | PR Applied 26, 034068 (2026) | Teoría+experimento QD biexcitón-exciton en microcavidad (g2, indistinguibilidad) | 20 / ~4 | Dispositivos diseñados específicamente | [TEXTO] concl. |
| 2501.16939 | PRA 113, 023718 (2026) | Interferencia de dos fotones, fluorescencia de resonancia de QD | 13 / ~10 | – | [RESUMEN] |
| 2505.05127 | PR Applied 24, 064064 (2025) | Acústica cuántica multimodo con qubits (régimen de acoplo en transición) | n/d | descarga fallida | [RESUMEN] |
| 2601.13509 | PR Applied 26, 014045 (2026) | Resonadores Lamb LiNbO3 suspendido | n/d | – | [RESUMEN] |
| 2603.12723 | PRB 114, 245405 (2026) | Torsión de nanotubos por espines | n/d | – | [RESUMEN] |
| 2512.14405 | PRL 136, 046904 (2026) | Control acusto-óptico de espín en QDs | n/d | – | [RESUMEN] |
| 2502.18707 | PRB 112, 115131 (2025) | Modelo de pseudomodos purificado (acoplo no lineal sistema-baño) | n/d | – | [RESUMEN] |
| 2407.14462 | PRB 110, 115308 (2024) | Cavidad-QED con baño fonónico de dimensión reducida | n/d | – | [RESUMEN] |
| 2508.14750 | PRA 113, 022420 (2026) | Preparación no clásica por medida de paridad generalizada | n/d | – | [RESUMEN] |
| 2503.11052 | PR Research 7, 023242 (2025) | Estados magnónicos no gaussianos | n/d | – | [RESUMEN] |
| 2605.12677 | AQT 9, e00845 (2026) | Batería cuántica no recíproca (línea saturada) | n/d | – | [RESUMEN] |
| 2501.16425 | PRA 112, 022414 (2025) | Qubits protegidos vistos como gato (fluxonium) | n/d | – | [RESUMEN] |
| 2508.17575 | PRB 112, 214310 (2025) | Mpemba cuántico en sistemas PT (línea saturada) | n/d | – | [RESUMEN] |
| 2609.11034, 2610.28749 | sin journal_ref | Kerr-cat: confinamiento disipativo con refrigerador SISIS; entrelazamiento con sesgo preservado | n/d | – | [RESUMEN] |

(≥12 artículos con conclusiones leídas: 16 [TEXTO]. Figuras observadas renderizadas en `fig/`: 2606.10319, 2508.10500, 2509.21569, 2602.22205.)

### Estándar confirmado y lecciones
- Contribuciones aceptadas: (i) propuesta realizable con parámetros experimentales y comparación analítico-numérico (2508.00353, 2509.24295, 2606.10319); (ii) aclaración conceptual con fórmula cerrada (2602.22205: 4-5 figuras, 12 págs); (iii) solución analítica explotando simetría (2504.12677). Longitud típica 8-23 págs, 2-8 figuras en cuerpo principal.
- Figuras vistas: 2606.10319 (AQT) tiene esquema, niveles con caminos, curvas analíticas sobre numéricas y mapas 2D con condición superpuesta; 2602.22205 (PRA) es más austero (esquema + gráficos de autovalores + curva de parámetro). El PRA de gatos mecánicos (2508.10500) solo muestra Wigner + espectros: es un artículo teórico de 23 págs con apéndices largos, no de muchas figuras. Es decir, PRA acepta fórmulas cerradas bien motivadas sin mapas elaborados; AQT pide más "estética de dispositivo".
- Huecos explícitos usados abajo: (a) 2508.10500 no analiza robustez térmica ni Kerr; (b) 2604.08655 menciona multimodo y acoplo óptimo sin teoría; (c) 2602.22205: LEP vs HEP con fonones térmicos, sin plataforma acústica ni gato; (d) 2509.21569 pide polarón en fuerte acoplo; (e) 2603.25264 pide comprensión analítica multimodo.

## 2. Cinco ideas (fichas)

### Idea 1. Presupuesto de errores del qubit-gato mecánico inducido por un qubit auxiliar: acción conjunta Kerr + fonones térmicos
(a) Sistema: resonador mecánico HBAR/nanomecánico (5 GHz o decenas de MHz) acoplado a un qubit (transmón o QD) conducido a 2ω_m, con acoplo transversal+longitudinal (como 2508.10500). Tras eliminar el qubit: pérdida de dos fonones κ2, bombeo ε, Kerr K, pérdida de un fonón κ1 con n_th. Parámetros ilustrativos: κ1/κ2 = 0.02, n_th 0.01-0.5 (HBAR a 10 mK reporta <1e-4 [TEXTO 2604.08655], nanomecánicos más calientes), K/κ2 ~ 0-2 (el Kerr es intrínseco a la eliminación adiabática).
(b) Giro: el Kerr se cancela exactamente a n_th=0 (variedad exacta con |α|²=2ε/√(κ2²+4K²)), pero con fonones térmicos amplifica la tasa de bit-flip (prototipo: hasta ×55). Esto no está en 2508.10500 (PRA 2026) y convierte su propuesta en una arquitectura con presupuesto de ruido y punto de operación óptimo (ε, desintonía qubit-conducción) que minimiza bit-flip a κ2 fijo.
(c) Analítica: condición cerrada del manifold; gap Liouvilliano (bit-flip) y paridad (phase-flip ≈ 2κ1(1+2n_th)|α|², confirmado a ~1% en n_th=0); ley de escala térmica del bit-flip ∝ n_th·f(K/κ2) a derivar con perturbación sobre el manifold; el LEP entre la rama de bit-flip y la de decaimiento define el tamaño máximo útil del gato.
(d) Figuras: esquema + niveles; mapa log10 Γ_bf vs (|α|², n_th) y vs (K/κ2, n_th) con la línea analítica |α|²=… discontinua; autovalores Liouvillianos vs ε con LEP; Wigner; tabla de parámetros HBAR/QD.
(e) Novedad (más cercanos): 2508.10500 (PRA 2026; gato mecánico por qubit conducido; Kerr aparece pero sin estudio de bit-flip/térmico [TEXTO]); 2609.11034 (Kerr-cat, confinamiento disipativo, circuito, [RESUMEN]); 2610.28749 (Kerr-cat bias-preserving, [RESUMEN]); 2608.05503 (gatos de dos modos mecánicos con térmico, entrelazamiento, no bit-flip, [TEXTO]); 2501.16425 (cat/fluxonio, [RESUMEN]). Búsquedas de la API con "thermal + bit-flip + Kerr + cat" devolvieron solo estos; no concluyente por la rigidez de la API: [NO VERIFICADO] que no exista en superconductores (gato Kerr+disipativo tiene literatura previa que no pude leer).
(f) Riesgo: "ya hecho en circuitos" (medio) — mitigar centrando en la derivación Kerr inducida por el qubit y parámetros mecánicos. "Demasiado específico": bajo-medio (plataforma acústica más general que QDs).
(g) Esfuerzo: 4-6 semanas (modelo efectivo + validación con qubit+oscilador completo en QuTiP, derivación de la ley térmica, 5-6 figuras).

### Idea 2. Reset de qubit con baño acústico multimodo: óptimo Liouvilliano y ley N/(N+1)
(a) Transmón + HBAR (FSR ~ decenas de MHz; Chu: 2604.08655 [TEXTO]); modos con κ, qubit con g; temperatura fonónica baja.
(b) Giro: el EP de un modo (g=|κ-γ|/4, tasa κ/2) se generaliza a un peine; el óptimo g_opt ∝ FSR y la tasa máxima se acerca a κ (prototipo: 0.50, 0.75, 0.87, 0.93 κ para N=1,3,7,15; conjetura κN/(N+1)).
(c) Analítica: matriz no hermítica 1+N tipo "star graph"; la ecuación secular da condiciones de degeneración; derivar g_opt/FSR(N) y el límite regla de oro (2πg²/FSR) con corrección; población residual ∝ n_th/(tasa).
(d) Figuras: mapa tasa de reset vs (g/κ, FSR/κ) (hecho: `proto2_map.png`), autovalores complejos vs g con EP, residual vs n_th.
(e) Novedad: 2604.08655 (experimento, sin teoría de óptimo, [TEXTO]); 2301.13571 (EP para reset en tres resonadores, PRResearch 2023, [RESUMEN]); 2608.26897 (reset por Mpemba/entrelazamiento, [RESUMEN]); 2603.25264 (transferencia multimodo, numérica, [TEXTO]); 2305.12745 (relajación acelerada por LEP, [RESUMEN]).
(f) Riesgo: "Mpemba/EP" tema cercano a línea saturada del grupo; "demasiado simple" (modelo de 1 excitación). Bajo en especificidad.
(g) Esfuerzo: 3-4 semanas.

### Idea 3. Fonones térmicos como perilla del LEP en qubit–fonón (extensión de 2602.22205 a acústica cuántica)
(a) Qubit-resonador acústico (SAW/HBAR) con decaimiento; (b) giro: desplazamiento LEP/HEP (γ_eff=γ(2n_th+1)) en un sistema Jaynes-Cummings con cola de excitaciones, sonda de n_th; (c) analítica: espectro Liouvilliano en el sector de una excitación con baño térmico y mapa de la posición del EP vs n_th; (d) mapa g vs n_th con EP; (e) novedad: 2602.22205 (PRA 2026, solo optomecánica lineal [TEXTO]), 2305.12745, 2301.13571 [RESUMEN]; (f) riesgo alto de "incremento sobre 2602.22205"; (g) 3 semanas. Sin prototipar.

### Idea 4. Subespacio libre de decoherencia en pares de QDs con baño de fonones LA correlacionado: la longitud de onda fonónica fija el "punto dulce"
(a) Molécula de dos QDs (d ~ 10-30 nm); correlador cruzado J12(ω)=J(ω)·sinc(ωd/c_s) (c_s~5 km/s); (b) giro: el estado oscuro es inmune a dephasing de fonones cuando el factor de correlación es -1 (ωd/c_s=π en el pico espectral); mapa de coherencia vs (d, T); (c) analítica: ratio de factores de Franck-Condon polarónicos B_D/B_B; (d) mapas vs d y T; (e) novedad: 1103.6228 (decoherencia por fonones en TQD, 2011) [RESUMEN]; búsquedas no hallaron trabajo sobre DFS con J12 [NO VERIFICADO]; (f) riesgo muy alto: es el mismo tema del trabajo rechazado ("demasiado específico"); (g) 4 semanas. Sin prototipar.

### Idea 5. Eliminación adiabática con polarón para fotón único por Raman en QD-cavidad (corrección fonónica a las tasas efectivas de James-Jerke)
(a) QD en microcavidad con excitación Raman/STIRAP; (b) giro: tasas efectivas renormalizadas por B(T)² más tasa asistida por fonones, con condición cerrada de pulso óptimo; (c) analítica: tasas Γ_eff = 4g²B²/…; (d) mapas pureza/indistinguibilidad vs (Δ, T); (e) novedad: 1706.07521 (PRB 2017, adiabatic passage con fonones, [RESUMEN]), 1205.1883 [RESUMEN], 2509.21569 [TEXTO], 2602.18153 [TEXTO]; (f) riesgo medio-alto: metodológico; (g) 5 semanas. Sin prototipar.

## 3. Prototipos QuTiP

### Prototipo 1 (Idea 1): `proto1_cat.py`, `proto1b_kerr.py`
Modelo efectivo: H = ε(a†²+a²) + K a†²a², pérdida √κ2 a², √(κ1(1+n_th)) a, √(κ1 n_th) a†; κ1/κ2 = 0.02; espectro del Liouvilliano (N=34, shift-invert). Γ_bf = primer autovalor no nulo; Γ_par = segundo.

Sin compensación (ε = α²κ2/2, K varía): el |α|² real baja → bit-flip empeora (artefacto de convención, corregido abajo).
Con la condición cerrada |α|²=3 fijado (ε=|α|²√(κ2²+4K²)/2):

| K/κ2 | n_th | Γ_bf (κ2) | Γ_par (κ2) |
|---|---|---|---|
| 0 | 0 | 2.0e-6 | 0.1196 |
| 0 | 0.1 | 1.3e-5 | 0.1473 |
| 0.5 | 0 | 1.5e-6 | 0.1198 |
| 0.5 | 0.1 | 5.0e-5 | 0.1476 |
| 1 | 0 | 1.2e-6 | 0.1199 |
| 1 | 0.1 | 2.1e-4 | 0.1476 |
| 2 | 0 | 9.7e-7 | 0.1200 |
| 2 | 0.1 | 7.1e-4 | 0.1472 |

Referencia sin Kerr, |α|²=2,3,4: Γ_bf=3.7e-5, 2.0e-6, 1.8e-7 (n_th=0); con n_th=0.1: 1.2e-4, 1.3e-5, 1.6e-6; Γ_par ≈ 2κ1|α|²(1+2n_th) a ≤3%.
Lectura: la condición cerrada se verifica (a n_th=0 el Kerr no daña: Γ_bf constante o mejor, autovalor cero a 1e-15); con n_th=0.1 y K/κ2=2 el bit-flip es ×55 el valor K=0 (y ×350 el de n_th=0). Efecto robusto, ≥1 orden de magnitud en el rango realista (K/κ2~1). Pendiente: derivar la ley analítica (los datos no siguen (1+4K²) simple: factores 1, 3.8, 16, 55 para K=0, .5, 1, 2) y validar el modelo efectivo contra qubit+oscilador completo (no hecho).
Veredicto: la señal existe.

### Prototipo 2 (Idea 2): `proto2_reset.py`, `proto2_map.png`
Matriz no hermítica (qubit + N modos, un solo excitón, T=0), κ=1, γ_q=0.01; tasa = decaimiento de la rama más lenta.

| N | FSR/κ | tasa máx (κ) | g_opt/κ | g_opt/FSR |
|---|---|---|---|---|
| 1 | – | 0.505 | 0.26 (EP teórico 0.2475) | – |
| 3 | 1 | 0.752 | 0.59 | 0.59 |
| 3 | 3 | 0.752 | 1.74 | 0.58 |
| 7 | 1 | 0.865 | 0.82 | 0.82 |
| 7 | 3 | 0.865 | 2.46 | 0.82 |
| 15 | 1 | 0.925 | 1.13 | 1.13 |

Lectura: el EP de un modo (κ/2) coincide con la teoría; con N modos la tasa máxima sube hacia κ (≈κ N/(N+1) empíricamente) y g_opt escala con el FSR, independiente de κ. Es una señal limpia pero de modelo muy simple; no incluye temperatura ni multiexcitación (no validado con baño térmico).

## 4. Ranking y recomendación

1. Idea 1 (qubit-gato mecánico: Kerr + térmico). Encaja exactamente con las habilidades (gato con qubit auxiliar, eliminación adiabática, Liouvilliano/LEP, QuTiP), tiene un efecto cuantificado en prototipo (×55 de amplificación del bit-flip, condición cerrada verificada), un artículo-ancla reciente aceptado en PRA (2508.10500) y un hueco declarado por sus autores (más allá de lo ideal). Es una arquitectura realizable (HBAR/nanomecánico) y evita las líneas saturadas.
2. Idea 2 (reset multimodo): señal limpia, plataforma real (2604.08655), pero cercana a Mpemba/EP (saturación) y modelo simple.
3. Idea 3: incremental sobre 2602.22205.
4. Idea 5: metodológica; riesgo de "ya hecho" parcial (1706.07521).
5. Idea 4: repite el perfil "demasiado específico" que ya les rechazaron.

Recomendación: Idea 1 hacia PRA (el artículo ancla 2508.10500 y los papers de gatos de dos modos 2608.05503 y 2606.25511 muestran que PRA/PRResearch aceptan este tipo de propuesta de 9-23 págs). Alternativa: AQT si se añade esquema de dispositivo y mapas 2D (estilo 2606.10319). Reducir riesgo: (i) validar el modelo efectivo contra el modelo completo qubit+oscilador (con Kerr inducido real); (ii) derivar la ley de escala térmica Γ_bf(n_th, K); (iii) comparar con la literatura Kerr-cat de circuitos (no leída aquí, [NO VERIFICADO]); (iv) incluir tabla de parámetros HBAR/QD. Limitaciones de este informe: la API de arXiv es rígida con consultas conjuntivas, de modo que "no encontrado" es evidencia débil de novedad; no se leyeron los PDF de los trabajos Kerr-cat de 2026.
