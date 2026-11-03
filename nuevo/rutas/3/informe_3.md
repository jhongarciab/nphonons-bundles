# Ruta 3 — Auditoría de novedad: bundles de dos fonones (g_x,g_z efectivo y escalera del transmon)

Marcas: **[TEXTO]** leído en el PDF descargado (se da página del PDF/sección/ecuación); **[RESUMEN]** solo abstract; **[NO VERIFICADO]** no leído (solo visto en la bibliografía de otro PDF, o conocimiento propio sin fuente). **[CÁLCULO]** aritmética mía sobre fórmulas leídas. PDFs en `/tmp/ruta3_pdf/`. Sin commit ni push. Referencias: `refs_3.bib` (solo lo leído).

## 0. Resumen ejecutivo (duro)
1. **El ingrediente "JC de dos bosones por eliminación adiabática de un nivel intermedio" y su uso para bundles de 2 fotones ya está publicado** (Tang & Deng 2409.01737, Ec. (2), pp. 2–3: g = −g0²/Δ1; escalera/Λ óptica, emisión por κ de la cavidad) [TEXTO]. El acoplo efectivo de dos fonones con g_x+g_z y "excitación del qubit → par de fonones" ya está propuesto para fonones (Naseem 2508.10500, p. 2) [TEXTO] y para gatos (Sánchez Muñoz 1802.01306, p. 2) [TEXTO]. Un modelo Rabi con mezcla σ_x/σ_z (ángulo θ) que emite pares de fotones con una Ω_eff analítica y pureza Π2 ya está en Bin–Wu–Lü 2012.10926 (pp. 4–5) [TEXTO] (en régimen ultrafuerte).
2. **La emisión de un par desde la escalera f→g a un canal lineal, con un nivel intermedio virtual**, ya está para fotones: Gao et al. 2406.12414 (transmon-like de tres niveles "ladder-type giant atom", guía lineal; tasa de decaimiento a dos fotones en forma cerrada, Ec. (16), p. 4) [TEXTO].
3. **Lo que NO encontré en nada de lo leído**: (i) la eficiencia cerrada de par η y el óptimo κ_m=√2G con la condición κ_q≤0.074G; (ii) estadística exacta de clusters por matriz de renovación; (iii) |f,0⟩→|g,2⟩ emitido como par de *fonones* a una guía con parámetros medidos de cristal fonónico; (iv) la comparación cuantitativa "sin λⁿ" frente a Bin 2020. Las búsquedas en la API de arXiv (≈30 consultas, varias fallidas por 429/timeouts) pueden haberse dejado cosas: la ausencia **no** es prueba.
4. Veredicto: C1 PARCIALMENTE CONOCIDA; C2 NOVEDOSA (pero de profundidad modesta: es álgebra de un sistema de dos modos con pérdida y estadística de cuentas estándar); C3 PARCIALMENTE CONOCIDA (mecanismo conocido en fotones; la novedad es la instancia fonónica con parámetros medidos); C4 PARCIALMENTE CONOCIDA/defendible solo si es cuantitativa y honesta (la penalización λⁿ de Bin se compensa con potencia óptica Ω; no es una barrera fundamental, §4).
5. Hallazgo incómodo para C3: Qiao et al. (Cleland) 2503.03898 ya usa la escalera g–e–f de un transmon con fonones itinerantes de SAW para *capturar* estados de dos fonones (P(gf)≈0.19) [TEXTO, pp. 8–9]; es el proceso inverso al que se quiere reclamar, con intermedio real y tonos de modulación. Un árbitro lo señalará.

## 1. Fichas

Leyenda de columnas de las fichas: Sistema · Mecanismo · Resultado · fotón/fonón · ¿emisión a un canal? · ¿fórmulas cerradas de eficiencia/pureza?

### F1. Tang & Deng, arXiv 2409.01737 (Kerr-type two-photon JC, bundles) — leído intro, modelo, conclusiones
- Sistema: átomo de 4 niveles (|g⟩,|e⟩,|r⟩,|m⟩) en cavidad óptica con medio Kerr (χâ†â†ââ), bombeo de cavidad η y bombeo Raman de dos láseres (Ω1,Ω2). Fig. 1, Ec. (1), p. 2 [TEXTO].
- Mecanismo: en el régimen lejos de resonancia (|g0/Δ1|≪1) se eliminan |e⟩,|r⟩ y queda H = … + g(â²σ_gm + h.c.) + Ω(σ_gm+h.c.) con **g = −g0²/Δ1** (Ec. (2), pp. 2–3). Espectro En,± = nΔc+(n²−3n+3)χ ± √((2n−3)²χ²+n(n−1)g²) (Ec. (4), p. 3). La excitación total N = â†â+2σ_mm se conserva (p. 3).
- Resultado: bloqueo de 2 y 3 fotones y bundles de 2–3 fotones sintonizables por χ, desintonía, bombeos; el Kerr ensancha los splittings Δn,± (Fig. 1c). Conclusiones p. 9. Fotones, óptica.
- Canal: la salida es el decaimiento κ de la cavidad en la ecuación maestra, Ec. (5) (solo "emisión de cavidad", no una guía con eficiencia); estacionario con bombeo CW.
- Cerradas: solo el espectro; correlaciones g^(k)_n por simulación numérica de Ec. (5) y Ec. (6). Sin η ni pureza analíticas.
- Relevancia: **es la misma estructura de "JC de dos cuantos por intermedio virtual" que el transmon-f**, en óptica. Su intro (p. 1) agrupa "photons (phonons)" bundles y cita Bin 2020 y la literatura de bundles.

### F2. Sánchez Muñoz, Lara, Puebla, Nori, arXiv 1802.01306 (PRL 121, 123604) — leído abstract, modelo, resultados
- Sistema: qubit de espín (NV, S=1) sobre un oscilador mecánico; acoplo **cuadrático** H ≈ H_NV+H_M+g2(a†+a)²S_z (Ec. (1), p. 2), con imanes de Dy; ω_m/2π=1.8 MHz, T=10 mK (Fig. 2 pie, p. 3) [TEXTO].
- Mecanismo: JC de dos fonones bajo bombeo coherente del qubit → transición de fase disipativa tipo oscilador paramétrico degenerado; estado estacionario mixto = "gato saltarín" (abstract, p. 1; Fig. 5, p. 4).
- Resultado: estados mecánicos no clásicos (Wigner negativa) con escalas de fracciones de segundo. Fonones, sin canal de emisión, sin bundles (el decaimiento es del oscilador al baño, Lindblad estándar, p. 3).
- Cerradas: no hay η ni pureza; hay umbrales de transición y cooperatividad.

### F3. Naseem, arXiv 2508.10500 (PRA 113, 013732) — leído abstract, intro, modelo, conclusión, Ap. A
- Sistema: nanorresonador mecánico + qubit con acoplo transversal **y** longitudinal (g_x, g_z), qubit conducido a ω_d=ω_q=2ω_m (abstract; Ec. (2) y discusión pp. 2–3) [TEXTO].
- Mecanismo: transformación unitaria perturbativa + eliminación adiabática del qubit (κ≫Γ2); da pérdida de dos fonones, término de squeezing coherente, Kerr. "Virtual pathways combine the two couplings to convert a qubit excitation coherently into a pair of phonons (and vice versa)" (p. 2, intro) [TEXTO].
- Resultado: gato mecánico determinista. Jerarquía de validez: g_{x,z}≪ω_m, κ≫{Γ2^∓,Γ1,γ}, ω_q≃ω_d≃2ω_m (p. 4). Conclusión p. 6: extensión a varios resonadores. Ap. A (p. 7): plataforma CPB/transmon/fluxonio; "se ha propuesto teóricamente… (p. ej. [72,73])" el acoplo simultáneo L+T.
- Fonones, sin bundle, sin guía: la "pérdida de dos fonones" es disipación del modo hacia el baño (γ±), no emisión a un canal contado. No aparecen las palabras *bundle* ni *waveguide* en el PDF (búsqueda por texto) [TEXTO].
- Sin η. Cita como antecedentes del mecanismo: Ma, Xie, Li PRA 99, 022302 (microondas, qubit sin simetría de inversión) y Hou et al. PRA 110, 013711 (magnón, flux qubit) [NO VERIFICADO: solo bibliografía de este PDF, refs. [30],[35]]. También cita Lambert et al. PRB 97, 125429 (acoplo L+T amplificado/sintonizable) como "second-order drive-assisted photon-pair processes demonstrated" [NO VERIFICADO].
- Importante: **el mecanismo H_ef = G(σ+a²+h.c.), G=2g_xg_z/ω es de Naseem/Ma/Hou; no es nuestro.**

### F4. Bin, Wu, Lü, arXiv 2012.10926 — Parity-symmetry-protected multiphoton bundle emission — leído abstract, intro, factibilidad
- Sistema: oscilador + qubit (flux qubit en el diseño de circuito) en acoplo **ultrafuerte**, qubit conducido; parámetros de ejemplo ω_r/2π=5 GHz, ω_q/2π=25 GHz, λ/2π=1 GHz, Ω/2π=0.3 GHz, γ/2π=0.5 MHz, κ/2π=200 MHz (p. 4, "Discussion of experimental feasibility") [TEXTO].
- Mecanismo: giro del qubit por láser |0,g⟩→|0,e⟩ + transiciones del Rabi con simetría de paridad → super-Rabi |0,g⟩↔|2n,e⟩; con disipación solo salen bundles de 2n fotones (abstract, p. 1).
- **Mezcla de acoplos:** el ángulo de mezcla θ (cosθ = ε/√(Δ²+ε²), p. 4–5) controla σ_x/σ_z; con θ≠mπ/2 se rompe la paridad y *aumenta* la tasa efectiva de dos fotones (Fig. 2(e), con Ω_eff analítica en el Suplemento), pero se pierde la pureza del par (Fig. 4: "pairs … never occur for θ=π/6", p. 4) [TEXTO].
- Fotones; la salida es el decaimiento de la cavidad κ (bundle "fuera del sistema"). Pureza Π2 por trayectorias (Fig. 4a-b). Dice "close to 100 % two-photon emission" con parámetros factibles (p. 4).
- Sin fonones, sin guía explícita, sin η cerrada; la Ω_eff analítica está en el Suplemento (no leído) [NO VERIFICADO el detalle].

### F5. Bin, Lü, Laussy, Nori, Wu, arXiv 1907.12714 (PRL 124, 053601) — leído abstract, modelo, pureza, conclusiones
- Sistema: punto cuántico + nanocavidad acústica, acoplo electrón–fonón λ, bombeo láser en la banda lateral n-ésima (Stokes). Parámetros ω_b/2π=1 THz, ω_σ/2π=100 THz, Ω/2π=0.2 THz, λ/2π=0.1 THz, κ/2π=2 GHz, γ/2π=0.2 GHz, γ_φ/2π=0.4 GHz (p. 4–5) [TEXTO].
- Mecanismo: oscilaciones super-Rabi |0,v⟩↔|n,c⟩ a tasa **Ω_eff^(n) = Ω(λ/ω_b)^n/√n!** (perturbativa, p. 2) y **Ω e^{−λ²/2ω_b²}(λ/ω_b)^n/√n!** (desplazada, p. 3) [TEXTO].
- Resultado: bundles de n fonones con pureza >97 % (Conclusiones, p. 5); contaminación por otros procesos discutida (p. 4, "Discussions of purity…"). Fonones, cavidad que emite por κ; "la pureza se debe a emisión en cascada rápida tras el colapso" (p. 4–5).
- Sin fórmulas cerradas de eficiencia; sí Ω_eff.

### F6. Gao, Li, Wu, Liu, Wang, arXiv 2406.12414 (PRA 110, 053706) — leído abstract, intro, modelo, conclusión
- Sistema: átomo gigante de tres niveles "ladder-type" (f,e,g, tipo transmon) acoplado en N puntos a una **guía lineal** (sin medio no lineal) [TEXTO, pp. 1–3].
- Mecanismo: se diseña la secuencia de acoplos (programación cuadrática secuencial) para que |e⟩ quede fuera de la ventana de acoplo (suprime el fotón único) y f→g emita dos fotones correlacionados por proceso de segundo orden vía estado intermedio virtual. Resultado analítico para la amplitud de |f⟩: ċ_f = 4L²|g_k0|⁴δw c_f /(πc²(Δ²−δw²)) (Ec. (16), p. 4). Condición práctica: Δ−δw<0.1ω0 (p. 4). Caso quiral (dirección única) y sistema en cascada mediado por dos fotones (p. 9, conclusión).
- Fotones, canal = guía lineal; hay tasa cerrada de decaimiento a pares; no hay η/pureza cerradas ni estadística de clusters.
- Relevancia: es **la versión fotónica de C3** (emisión de pares f→g a una guía vía intermedio virtual con transmon-like).

### F7. Qiao et al. (Cleland), arXiv 2503.03898 — Acoustic phonon phase gates with number-resolving detection — leído abstract, Fig. 4, Métodos
- Sistema: dos transmones con transductores (UDT) y SAW; fonones itinerantes a 3.925 GHz [TEXTO, p. 8].
- Mecanismo: "two-phonon catch": se modula el qubit a la anarmonicidad χ=|ω_ef−ω_ge|=185 MHz (189 MHz) creando bandas laterales ±χ de g↔e y e↔f para que el estado |02⟩−|20⟩ (HOM) se absorba en |gf⟩ (p. 8).
- Resultado: P(|gf⟩)=0.19, P(|fg⟩)=0.18 con capturas de un fonón P(ge)=0.24, P(eg)=0.23 (p. 9); T1(e–f)=1.5 µs en el máximo local (p. 8) [TEXTO].
- Fonones, canal acústico real, **experimento**. Es captura (inverso temporal) con tonos de microondas, no emisión desde |f⟩ ni bundle; no vi que lo propongan [TEXTO: no aparece "emit … from |f⟩"; revisión por texto, no exhaustiva]. Sin fórmulas de eficiencia.

### F8. Fichas breves (menos cercanas)
| Trabajo | Sistema/mecanismo | Fotón/fonón; canal | Lectura |
|---|---|---|---|
| Liu, Huang, Tian 2210.03619 | átomo Ξ en Rabi ultrafuerte, STIRAP con pulsos gaussianos; bundles de 2–6 fotones deterministas | fotones, salida de la cavidad | [TEXTO] abstract + p. 1, 10. Mecanismo distinto (STIRAP+USC) |
| Bin et al. 2406.12631 | resonador giratorio (Sagnac), super-Rabi fotón–fonón/fotón–magnón; bundles de pares entrelazados | pares mixtos, no dos fonones | [TEXTO] abstract |
| Zhang, Zhao, Tang, Deng 2610.09971 | emisor Mollow tripartito (fotones + fonones motionales); pares fotón–fonón; bundles de 2 fotones de alta pureza con fonones acumulados | mixto | [TEXTO] abstract; el fonón no sale como par |
| Wei, Tang, Deng 2412.18133 | JC de alto espín (spin-3/2) | fotones | [TEXTO] abstract |
| Deng, Shi, Yi 2011.03886 | átomo atrapado (Sr, transición de reloj), bundles de N fonones del movimiento, régimen de resonancia átomo–fonón | fonones del movimiento (iones/átomos), sin guía | [TEXTO] abstract, p. 2 |
| Cosacchi et al. 2108.03967 | estadística de N fotones de bundle en QD-microcavidad y en qubit superconductor–resonador (JC bombeado); N=2 factible en circuitos con resonador de Q mediocre | fotones, distribución del número en cavidad (no cluster de clics) | [TEXTO] abstract, pp. 2, 5–6 |
| Ma, Xie, Ren, Li, Li 2203.15373 | Josephson fotónica: qubit de carga + dos resonadores no degenerados; "photon-pair blockade", pares antiagrupados | fotones, dos modos distintos | [TEXTO] abstract |

## 2. Matriz de coincidencia con C1–C4
Sí / parcial / no (con justificación). "Parcial" significa que una parte sustancial de la afirmación ya está publicada.

| Trabajo | C1 par de **fonones** a una guía vía acoplo efectivo de 2 fonones (g_x,g_z) | C2 fórmulas cerradas (η, κ_m*, κ_q≤0.074G, P2(τc) por renovación) | C3 fuente desde la escalera |f,0⟩→|g,2⟩ con parámetros de plataforma y escala por fonón | C4 comparación sistemática con Bin (sin λⁿ) |
|---|---|---|---|---|
| Tang–Deng 2409.01737 | parcial: JC de dos fotones con bundles (Ec. (2), p. 3); fotones, óptica, κ de cavidad | no (solo espectro, Ec. (4)) | parcial: g=−g0²/Δ1 (escalera con intermedio virtual, p. 3); sin parámetros de transmon ni fonones | no (cita a Bin solo en la lista de bundles, p. 1) |
| Sánchez Muñoz 2018 | parcial: JC de dos fonones (Ec. (1)), pero acoplo cuadrático y gato, no bundle ni guía | no | no | no |
| Naseem 2025/26 | **parcial-alto**: g_x+g_z, qubit→par de fonones (p. 2), pero disipativo/gato y sin canal contado | no (adiabático Γ2, Kerr; sin η) | no | no |
| Bin–Wu–Lü 2012.10926 | parcial: mezcla σ_x/σ_z (θ) con Ω_eff y pureza de pares (pp. 4–5), fotones, USC | parcial: Ω_eff analítica (Suplemento) pero pureza numérica | no | no (sin comparar con Stokes por λⁿ) |
| Bin 2020 (1907.12714) | parcial: bundles de n fonones, n=2 incluido (p. 2–5), pero mecanismo Stokes con λⁿ | parcial: Ω_eff^(n) (p. 2–3); sin η | no | — (es el comparado) |
| Gao 2406.12414 | no (fotones) | parcial: tasa de decaimiento a pares Ec. (16), p. 4 | **parcial-alto**: pares f→g a una guía vía intermedio virtual (fotones, átomo gigante) | no |
| Qiao 2503.03898 | no (captura, no emisión) | no | **parcial**: escalera de transmon con fonones SAW, P(gf)=0.19 (pp. 8–9); inverso temporal con tonos | no |
| Zhang 2610.09971 / Bin 2406.12631 | no (fotón–fonón) | no | no | no |
| Liu 2210.03619 | no | no | no | no |
| Wei 2412.18133 | no | no | no | no |
| Deng 2011.03886 | parcial: bundles de fonones (átomo atrapado), sin qubit–g_z | no | no | no |
| Cosacchi 2108.03967 | no | parcial: estadística N=2 de bundle (número en cavidad), numérica | no | no |
| Ma 2203.15373 | no | no | no | no |

## 3. Veredicto por contribución

**[C1] Emisión de pares de fonones a una guía por el acoplo efectivo g_x,g_z: PARCIALMENTE CONOCIDA.**
- Conocido: H_ef=G(σ+a²+h.c.) con G=2g_xg_z/ω y "qubit→par de fonones" (Naseem p. 2; antecedentes Ma/Hou [NO VERIFICADO]); JC de dos fonones (Sánchez Muñoz p. 2); bundles de fonones (Bin 2020, Deng 2021) y de 2 fotones con mezcla L+T (Bin–Wu–Lü p. 4).
- Posiblemente nuevo: el régimen "coherente" (κ_q≲0.07G, modo → guía con κ_m≈√2G) donde el par sale contado a un canal en vez de disiparse en un baño; fonones. No lo he visto en ningún PDF leído.
- Problema duro: **no hay g_z mecánico demostrado** (informe E, tabla de plataformas); el árbitro dirá que es una extrapolación de Naseem. Con g_x=g_z=15 MHz y ω=2.3 GHz, G/2π≈0.2 MHz, y T1=1.14 µs da κ_q/G≈0.7 (η≈0.64) [CÁLCULO de E].

**[C2] Fórmulas cerradas (η, κ_m*=√2G, κ_q≤0.074G, P2(τc) exacta): NOVEDOSA en lo que leí, con matices.**
- Ningún trabajo leído da una eficiencia de par cerrada ni una estadística de clusters exacta: Tang–Deng (Ec. 4), Bin (Ω_eff), Gao (Ec. 16, solo tasa), Bin–Wu–Lü (Π2 numérica por trayectorias), Cosacchi (distribución numérica).
- Matices: (i) η es la razón de ramificación de un sistema de dos modos con pérdida (álgebra de Lyapunov); es correcta y útil pero elemental; (ii) la renovación para waiting times/clusters es técnica estándar de conteo; lo nuevo es aplicarla al criterio de pureza P2(τc). (iii) No pude buscar en profundidad la literatura general de estadística de conteo/tiempos de espera; [NO VERIFICADO] que no exista un resultado equivalente.
- Defendible solo si se **verifica contra el modelo completo** (ya hecho en E hasta η completo +0.2–0.4 pp) y se acota su dominio (κ_q ignorando dephasing, 1 fonón por Purcell g_x).

**[C3] Fuente de pares desde |f,0⟩→|g,2⟩ con parámetros medidos y escala por fonón: PARCIALMENTE CONOCIDA.**
- Mecanismo (JC de dos cuantos por intermedio virtual, V=g²/Δ-tipo): Tang–Deng Ec. (2), p. 3 (óptica). Emisión de pares de fotones f→g a una guía con intermedio virtual y supresión del fotón único: Gao 2024, Ec. (16), p. 4. Transmon + fonones SAW + escalera g–e–f para dos fonones: Qiao 2025, pp. 8–9 (captura, intermedio real/tonos).
- Posiblemente nuevo: la versión *fonónica* con V=4g²/|α| y parámetros de cristal fonónico publicados (Arrangoiz-Arriola: g/2π=15.2 MHz, α/2π=138 MHz, T1=1.14 µs; pp. 3, 7, 10 [TEXTO]) → V/2π≈5.7 MHz [CÁLCULO de E] y la escala por fonón g/α≈0.1–0.2.
- Riesgos: la escala "por fonón extra" es el contaje de órdenes de perturbación típico, no un resultado nuevo; el modo de ese dispositivo tiene κ/2π=370 kHz (Tabla I, p. 7) y vida mecánica 430 ns (p. 10), no una guía: κ_m/2π≈8 MHz óptimo exige un acoplo a guía no demostrado en ese dispositivo [TEXTO + CÁLCULO]. Se preparan |f⟩ con dos pulsos π (usual); el "bundle" es en realidad un solo evento de par por ciclo de preparación, o un flujo con bombeo CW de baja potencia (E: flujo ~0.03–0.07 G).

**[C4] Comparación sistemática con Bin (sin λⁿ): PARCIALMENTE CONOCIDA/defendible con cuidado.**
- Bin 2020 da explícitamente la ley Ω_eff^(n)=Ω(λ/ω_b)^n/√n! (p. 2). Pero Ω es la frecuencia de Rabi óptica y puede ser grande (Ω/2π=0.2 THz con λ/2π=0.1 THz, p. 4–5): λⁿ se compensa con potencia. **No se puede decir que "paga λⁿ" como límite fundamental.** La comparación honesta es a igual λ/ω y igual Ω/ω: con λ/ω=0.006 (el g_x/ω de E) y Ω/ω=0.3, Bin n=2 da Ω_eff/ω≈0.3·(0.006)²/√2≈7.6×10⁻⁶ [CÁLCULO sobre p. 2], frente a G/ω≈7×10⁻⁴ (si g_z/ω=0.06) o V/ω≈2.5×10⁻³ (transmon-f, 5.7 MHz/2.3 GHz). Ahí sí hay ventaja de 2–3 órdenes de magnitud en la tasa coherente, pero con diferencia de plataforma (QD óptico a THz frente a circuito a GHz).
- Bin–Wu–Lü 2012.10926 es el pariente más cercano en *circuitos* (flux qubit, mezcla θ). Debe incluirse en la comparación, no solo Bin 2020.

## 4. Cómo plantear el artículo (qué reclamar y qué NO)
**Reclamar (defendible):**
1. Un **criterio cuantitativo de diseño** para fuentes de pares de fonones a una guía: η(κ_m,κ_q,G), κ_m*=√2G, κ_q≤0.074G (para η≥0.95) y P2(τc) exacta, verificados contra el modelo completo (E). Formularlo como "regla de diseño transferible" a cualquier JC de dos cuantos con pérdida (fotones, fonones).
2. **Instancia fonónica concreta con números de dispositivos publicados** (transmon-f en cristal fonónico: V≈5.7 MHz, η de par ≈0.92 con T1 medido, 0.99 con T1=10 µs, por E) y lista de lo que falta (guía acoplada al modo, T1).
3. Comparación **cuantitativa y simétrica** con Bin (misma λ/ω, misma Ω/ω) incluyendo requisitos distintos: Bin necesita láser óptico, QD y THz; esta ruta necesita microondas y un transmon.
**No reclamar:** (a) el acoplo efectivo de dos fonones G=2g_xg_z/ω (Naseem, Ma, Hou); (b) el JC de dos cuantos por intermedio virtual ni los bundles de 2 fotones por esa vía (Tang–Deng); (c) la emisión de pares f→g a una guía en general (Gao 2024 en fotones); (d) "primera fuente de bundles de fonones" (Bin 2020, Deng 2021); (e) ventaja "sin λⁿ" como barrera fundamental; (f) viabilidad experimental de g_z mecánico.
**Cálculos/experimentos adicionales que diferencian:**
- Orden N=3 (|h,0⟩→|g,3⟩ o JC de tres bosones) con la misma teoría cerrada: es lo que daría "bundles" de verdad frente a pares; coste ~(g/α)² adicional [CÁLCULO por hacer, aún sin verificar].
- Canal de salida explícito: simular el modo acoplado a una guía real (cascada, retardo, pérdida de transductor), y la detección (eficiencia finita) → g^(2) y P2 medibles.
- Hibridación g_z + escalera: qué pasa si ambos mecanismos coexisten (interferencia de amplitudes de |e,0⟩↔|g,2⟩ y |f,0⟩↔|g,2⟩).
- Dephasing y T2 del qubit, y modos mecánicos vecinos (el PnC de Arrangoiz-Arriola tiene varios modos 2.0–2.4 GHz, Tabla I, p. 7).
- Comparar con el proceso de captura de Qiao (inverso temporal): mostrar que la emisión del par desde |f⟩ es el "reverso" y estimar su P(par) con los mismos parámetros (T1(e–f)=1.5 µs).
- Una búsqueda bibliográfica dirigida (APS/Nature/PRX no accesibles aquí): transiciones f0–g1 asistidas por bombeo en circuitos y estados de Fock con la escalera del transmon; **no las leí, no se citan**.

## 5. Tres objeciones probables de un árbitro de PRA y respuestas
1. **"El acoplo efectivo de dos fonones y los bundles de 2 cuantos ya se conocen (Naseem, Sánchez Muñoz, Tang–Deng, Bin–Wu–Lü)."** Respuesta: aceptarlo. La contribución no es el Hamiltoniano sino (a) el régimen de emisión contada a una guía con qubit coherente, ausente en Naseem (régimen disipativo del gato), (b) las condiciones cerradas de eficiencia/pureza que ninguno da, (c) su verificación contra el modelo completo. Reescribir el abstract para que la novedad sea el criterio de diseño y la instancia fonónica.
2. **"No hay g_z mecánico demostrado y T1=1.14 µs; esto es especulativo."** Respuesta: separar las dos rutas. La ruta transmon-f usa solo parámetros medidos (g, α, T1: Arrangoiz-Arriola Tabla I, pp. 3, 7) y da P(par)≈0.92 por E; la ruta g_z se presenta como límite de plataforma futura con la tabla de requisitos (κ_q≤0.074G). Añadir la incertidumbre sobre κ_m (el dispositivo medido tiene κ/2π=370 kHz, no 8 MHz): decir claramente que se necesita ingeniería de acoplo a guía.
3. **"La comparación con Bin es injusta: λⁿ se compensa con potencia del láser (Ω/2π=0.2 THz con λ/2π=0.1 THz)."** Respuesta: reconocerlo (Bin p. 2–5) y comparar a igual λ/ω y igual Ω/ω, dar la razón de tasas coherentes (≈10²–10³ a λ/ω=0.006) y los requisitos distintos (óptico/THz frente a microondas/GHz). No decir "sin pagar λⁿ" como ventaja absoluta; decir "la tasa coherente no depende de λ/ω sino de g/α o g_xg_z/ω".

## 6. Limitaciones de esta auditoría
- No pude abrir APS/Wiley/Nature; no leí Suplementos; no leí a fondo todas las secciones de los PDFs largos (ver marcas por ficha).
- Las consultas de arXiv devolvieron resultados parciales (429/timeouts); no recuperé Ma–Xie–Li 2019, Lambert 2018, Kockum 2017 (virtual photons), Muñoz 2014 (emitters of N-photon bundles) ni Chang 2016 (down-conversion determinista), que aparecen citados en los PDFs leídos y son candidatos de cierre de prior art. Todos [NO VERIFICADO]. Un resultado de la última tanda (arXiv 1506.06363, "Engineering entangled microwave photon states via multiphoton interactions between two cavity fields and a superconducting qubit") solo se vio por título: candidato a leer, no citado.
- Datos de Arrangoiz-Arriola leídos en `/tmp/rutaE_pdf/1902.04681.pdf` (por la ruta E).
