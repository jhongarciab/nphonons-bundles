# Literatura verificada: bundles de N fonones con acoplo longitudinal

Fecha de la búsqueda: 2026-11-02. Fuente única: API de arXiv y PDFs de arXiv descargados en `/home/user/nphonons-bundles/nuevo/lit/pdf/` (APS, Wiley e IOP no accesibles).

Etiquetas de estado:
- **[TEXTO]**: leído en el PDF de arXiv descargado.
- **[RESUMEN]**: solo abstract o metadatos de la API.
- **[NO VERIFICADO]**: no confirmado con ninguna fuente.
- **[DERIVADO]**: cálculo mío a partir de números leídos.

El `refs_bundles.bib` contiene solo entradas con al menos [RESUMEN].

## (i) Tabla de referencias verificadas

### Bundles

| arXiv | Autores (1.º…) | Año | Revista (según metadato arXiv) | Estado |
|---|---|---|---|---|
| 1907.12714 | Q. Bin, X.-Y. Lü, F. P. Laussy, F. Nori, Y. Wu | 2019/2020 | PRL 124, 053601 (2020) | [TEXTO] 7 pp. El PDF **no incluye el Suppl.** (la ref. [42] remite al Suppl. y no está en el archivo). Título: los metadatos arXiv dicen "anti-Stokes", el PDF v2 dice "Stokes". |
| 2012.10926 | Q. Bin, Y. Wu, X.-Y. Lü | 2020/2021 | PRL 127, 073602 (2021) | [TEXTO] 9 pp. El Suppl. tampoco está en el PDF. |
| 1306.1578 | C. Sánchez Muñoz, E. del Valle, A. González Tudela, … , F. P. Laussy (ver bib) | 2013 | arXiv no da journal_ref. No cito revista: [NO VERIFICADO] | [TEXTO] |
| 2108.03967 | M. Cosacchi, A. Mielnik-Pyszczorski, … , V. M. Axt | 2021 | arXiv no da journal_ref | [TEXTO] |
| 2011.03886 | Y. Deng, T. Shi, S. Yi | 2020/2021 | Photonics Res. 9, 1289 (2021) | [TEXTO] |
| 2604.15604 | T. Jing, Y. Deng | 2026 | — | [RESUMEN] |
| 2604.15605 | J. Tang, Y. Deng | 2026 | — | [RESUMEN] |
| 2609.07454 | Z. Chen, J. Xu, D. Kong, X. Hu, F. Wang | 2026 | — | [TEXTO] parcial (intro y puntos analíticos) |
| 2603.27160 | Y.-X. Zhang, C. Wang, Q.-H. Chen | 2026 | Opt. Express 34, 21141 (2026) | [TEXTO] parcial |
| 2610.09971 | J. Zhang, G. Zhao, J. Tang, Y. Deng | 2026 | — | [TEXTO] parcial |
| PRResearch 2025 "Two-photon decay enhanced even photon bundle emission" | — | — | — | **[NO VERIFICADO]**: ocho búsquedas en la API de arXiv (título, abstract, palabras clave) no devolvieron ningún id. No se incluye en el .bib. Puede no estar en arXiv; hace falta DOI o autores. |

Otros trabajos de bundles hallados y pertinentes:

| arXiv | Qué es | Estado |
|---|---|---|
| 2304.11298 (Zou, Li, Liao; NJP 25, 043027) | Bundles dinámicos con STIRAP en circuito QED. "Highly pure" con la ayuda del decaimiento del fotón. | [RESUMEN] |
| 2112.03679 (Vargas-Calderón, Vinck-Posada, Villas-Boas) | QD (excitón oscuro/brillante) con cavidad acústica y láser, multifonón. Es la plataforma de Bin con más estructura electrónica. | [RESUMEN] |

### Maestra vestida / ultrastrong

| arXiv | Referencia | Estado |
|---|---|---|
| 1107.3990 | Beaudoin, Gambetta, Blais, PRA 84, 043832 (2011) | [TEXTO], coincide con la revista |
| 1206.0944 | Ridolfo, Leib, Savasta, Hartmann, PRL 109, 193602 (2012) | [TEXTO], coincide con la revista |
| 1807.06348 | Settineri et al., PRA 98, 053834 (2018) | [TEXTO], coincide con la revista |
| 1809.08749 | Di Stefano et al., "Resolution of Gauge Ambiguities…", Nature Physics (2019) | [TEXTO]. La cita "Nat. Phys. 15, 803" solo la confirma indirectamente el abstract de 2005.06499. |
| 2005.06499 | Stokes y Nazir, Nat. Phys. 20, 376 (2024): comentario crítico a Di Stefano | [RESUMEN] |
| 2107.07667 | Wang, Chen, Liao, PRA 104, 033701: qubit con acoplo longitudinal y cuadrático a un resonador, con "quantum dressed master equation" | [RESUMEN] |
| "Hu, Gasparinetti et al." | No identificable con búsquedas por autor y tema | **[NO VERIFICADO]** |

## (ii) Fichas (qué hace / qué no hace)

Leyenda: (a) fórmulas cerradas de pureza o flujo; (b) cota de pureza; (c) comparación de maestras local y vestida.

**Bin et al. 1907.12714 (PRL 124, 053601).**
- Modelo: es exactamente el de nuestro artículo: H = ω_b b†b + ω_σ σ†σ + λσ†σ(b†+b) + Ω(e^{iω_L t}σ + h.c.) [TEXTO].
- Mecanismo: resonancias Stokes con ω_σ − ω_L = −nω_b. Tras el desplazamiento polarónico, la resonancia pasa a Δ_n(λ) = λ²/ω_b − nω_b y se desplaza también con Ω. El Hamiltoniano reducido es análogo al de un ion atrapado, según el propio artículo.
- Frecuencia de super-Rabi perturbativa: Ω_eff^(n) = Ω(λ/ω_b)^n/√n! [TEXTO]. Es el límite de λ pequeño del factor de Franck–Condon exacto, que no aparece en el texto principal.
- Maestra: **local/desnuda**, dρ/dt = −i[H,ρ] + κL[b] + γL[σ] + γφ L[σ†σ] [TEXTO]. No hay maestra vestida ni polarónica en el texto principal.
- Pureza: Π_n = P̄_n/Σ P̄_{n_i}, con poblaciones muestreadas en ventanas de tiempo de trayectorias de Monte Carlo [TEXTO]. Se obtiene numéricamente, no en forma cerrada.
- Heurística: la Fig. 4 usa un "sweet point κ ≈ 10 Ω_eff" [TEXTO]. Es una receta numérica sin derivación y es el antecedente más cercano a un κ* (ver riesgos).
- Resultados: ≈99 % (2 fonones) y ≈97 % (3 fonones), con pureza >95 % en un rango amplio (Fig. 5), dependiendo de κ/ω_b y λ/ω_b.
- Dephasing: γφ "basicamente despreciable" porque el ciclo en cascada es rápido [TEXTO].
- Respecto a (a): solo Ω_eff perturbativo. (b): no. (c): no.

**Bin et al. 2012.10926 (PRL 127, 073602).**
- Modelo: Rabi extendido, ω_q σ_z/2 + ω_r a†a + λ(cosθ σ_z − sinθ σ_x)(a+a†), con bombeo en σ_x. Incluye explícitamente la componente **longitudinal** σ_z(a+a†) [TEXTO].
- Maestra: **vestida (renormalizada)**, dρ/dt = i[ρ,H] + κL[X] + γL[D] con X = Σ_{m>n}⟨ψ_n|(a+a†)|ψ_m⟩|ψ_n⟩⟨ψ_m| y D análogo para σ. Los saltos usan solo transiciones m>n (parte de frecuencia positiva) y se cita Beaudoin [57] [TEXTO].
- Pureza: Π₂ = P₂/P_tot por conteo de clics en trayectorias [TEXTO]. El Suppl. no está disponible.
- Parámetros: ω_r/2π = 5 GHz, ω_q/2π = 25 GHz, λ/2π = 1 GHz, Ω/2π = 0.3 GHz, γ/2π = 0.5 MHz, κ/2π = 200 MHz [TEXTO]. El pie de Fig. 1 da γ/ω_r = 1e-4 y κ/γ = 20; es inconsistente con κ/2π = 200 MHz (κ/γ = 400) y lo señalo como incoherencia del artículo.
- Respecto a (a): no (numérico). (b): no (solo parity-protection). (c): **no compara** local y vestida. Es la referencia más cercana al uso de maestra vestida con acoplo longitudinal y bombeo.

**Muñoz et al. 1306.1578.**
- Primer esquema de bundles: TLS bombeado en una cavidad JC con maestra local (Lindblad). Parámetros γ_a/g = 0.1 y γ_σ/g = 0.01 [TEXTO].
- Define la pureza de N fotones π_N a partir de las tasas λ_N de un modelo de conteo (λ_N = γ_a ⟨n⟩/N si hay supresión eficaz). La ecuación exacta se perdió en la extracción de texto [TEXTO parcial].
- Cita: ≈85 % de 2 fotones con γ_a/g ≈ 0.5, g ≈ 12 GHz en semiconductores; ≈90 % de 3 fotones en circuitos con γ_a/g ≈ 0.01, g ≈ 50 MHz.
- Respecto a (a): no cerrada. (b): no. (c): no.

**Cosacchi et al. 2108.03967.**
- Estudia la estadística de bundles en dos plataformas. En QD–microcavidad con fonones acústicos longitudinales usa path integral iterativo numéricamente exacto. El efecto de fonones es **perjudicial**: el **polaron shift** del excitón desplaza las resonancias y un γφ fenomenológico no reproduce el efecto de los fonones microscópicos [TEXTO].
- En el qubit superconductor el pure dephasing es irrelevante.
- Los fonones son un **baño** de un QD cuyo bundle es de fotones; no es un modo discreto con bundle de fonones.
- Respecto a (a): no. (b): no. (c): compara γφ fenomenológico con el modelo microscópico, que es otro tipo de comparación local/microscópica.

**Deng, Shi, Yi 2011.03886 (Photonics Res. 9, 1289).**
- Átomo de ⁸⁷Sr con láser de reloj dependiente de la posición: modelo Rabi generalizado con JC y anti-JC, es decir, acoplo **transversal** efectivo (no λσ†σ(b+b†)).
- Maestra local de Lindblad, justificada porque |g/ω| < 1e-10 en la aproximación RWA [TEXTO].
- Parámetros: ω hasta ≈2π·100 kHz, g_x/h = 9 kHz, κ_d/g_x = 0.005, γ_d/g_x = 5e-4.
- Observable: g^(n)_k generalizada (sin pureza). Respecto a (a)(b)(c): no.

**2604.15604 y 2604.15605 (Tang, Deng).**
- Arrays de átomos acoplados a cavidad, con fase relativa φ e interacción cavidad-mediada χ. Bundles de **fotones** con 3 átomos. Pureza y brillo numéricos [RESUMEN].
- Respecto a (a)(b)(c): no aparecen en el abstract.

**2609.07454 (Chen et al.).**
- Bundles de 2 fotones de dos emisores separados, mediados por un emisor auxiliar fuertemente bombeado (régimen Mollow). Maestra Lindblad estándar.
- **Analítico**: la fuga que degrada la pureza se suprime como g²/Ω² en bombeo fuerte [TEXTO], con compromiso pureza-tasa.
- Es el trabajo reciente con la afirmación analítica más cercana. Es de sistemas finitos de emisores, no de bosones, y no hay cota tipo Franck–Condon ni comparación de maestras.

**2603.27160 (Zhang, Wang, Chen).**
- Modelo de Rabi cuántico y sus extensiones, con **maestra vestida** (dressed master equation) y operador de detección vestido X⁺ a temperatura finita [TEXTO].
- "Stark" aquí es el término no lineal de Rabi–Stark, U a†a σ_z, no el desplazamiento Stark dinámico. Estadística G₂(0), sin bombeo ni pureza.
- Respecto a (c): solo usa la vestida. No compara con la local.

**2610.09971 (Zhang, Zhao, Tang, Deng).**
- Emisor bombeado, con acoplo tripartito a fotones de cavidad y fonones de movimiento (ion atrapado en cavidad), régimen Mollow. Pares foton–fonón y bundles de fotones seleccionados por la simetría de la disipación (κ_a/κ_b) [TEXTO].
- Maestra Lindblad local. Respecto a (a)(b)(c): no.

### Maestra vestida: qué dicen

**Beaudoin–Gambetta–Blais (1107.3990).**
- Deriva la maestra tras diagonalizar el Hamiltoniano de Rabi (sistema + sistema–baño) y "neglecting high-frequency terms". Esto equivale a una aproximación secular en la base de H_S y baños independientes si la anchura de línea es pequeña [TEXTO].
- El Lindbladiano vestido tiene saltos |j⟩⟨k| entre autoestados con tasas dadas por el ruido a la frecuencia Δ_kj, y a T=0 solo hay decaimiento hacia el estado fundamental real [TEXTO].
- Un baño σ_z produce un término de frecuencia cero que da **pure dephasing en la base de autoestados**, Φ_j D[|j⟩⟨j|], y además puede inducir transiciones entre niveles de la misma paridad ("dephasing-induced generation of photons") [TEXTO].
- Es acoplo transversal, sin bombeo ni acoplo longitudinal explícito.

**Ridolfo et al. (1206.0944).**
- Hamiltoniano más general, H₀ = ω₀a†a + ω_xσ⁺σ⁻ + g(a+a†)[cosθ σ_z − sinθ σ_x], con **componente σ_z (longitudinal) explícita** y bombeo coherente del modo (no del qubit) [TEXTO].
- El campo de salida se construye con la componente de **frecuencia positiva** de X en la base vestida, que "no es proporcional a a" [TEXTO]. Esto es lo relevante para "jumps con DC": el operador de salida de frecuencia positiva excluye la parte diagonal.
- El dephasing se omite en el texto principal y aparece en el apéndice como γ_deph⟨j|σ_z|j⟩ (Fig. 5).
- Es el único de los cuatro que trata σ_z(a+a†) **con bombeo**, aunque el bombeo va al modo y el observable es g^(2).

**Settineri et al. (1807.06348).**
- Maestra general para sistemas híbridos sin aproximación secular completa. Los términos con ω=ω'=0 (operadores diagonales) dan **contribuciones adicionales de pure dephasing**, y con niveles bien separados (ω ≫ Γ_i) se descartan los términos que oscilan a ±(ω+ω') [TEXTO].
- Ejemplos: Rabi y optomecánica (con DCE), con ruido térmico.
- No trata acoplo longitudinal qubit–oscilador ni bombeo del qubit.

**Di Stefano et al. (1809.08749).**
- Gauge en el modelo de Rabi truncado [RESUMEN]. No trata disipación. Un comentario posterior (2005.06499) afirma que la propuesta "rests on an incorrect mathematical assertion" [RESUMEN]. Es tangencial; recomiendo no citarla como apoyo de la maestra vestida.

**Maestra polarónica para σ_z(b+b†):** no encontré ningún artículo que la trate explícitamente. 2107.07667 (Wang–Chen–Liao) usa la maestra vestida con qubit acoplado longitudinalmente (sin bombeo, en transporte térmico) [RESUMEN]. Pregunta (2): ninguno de los cuatro estándar trata σ_z(b+b†) **con bombeo del qubit** y bundles; el más cercano es 2012.10926, que usa la vestida con θ general, incluido el caso longitudinal.

## (iii) Plataformas (λ/ω_b, κ/ω_b, γ/ω_b; unidades ω_b = 1)

κ es el decaimiento del modo, κ = ω_b/Q. La condición **κ ≪ λ²** significa κ ≪ λ²/ω_b (en unidades ω_b = 1, comparar κ/ω_b con (λ/ω_b)²). El cociente R = (λ²/ω_b)/κ se muestra abajo; R ≫ 1 indica que se cumple.

| Plataforma y fuente | ω_b/2π | λ/ω_b | κ/ω_b | γ/ω_b | R = λ²/(ω_bκ) | Notas |
|---|---|---|---|---|---|---|
| **Bin 1907.12714 (propuesta)** [TEXTO] | 1 THz | 0.1 (λ/2π = 0.1 THz) | 0.002 | 2e-4 (γφ: 4e-4) | **5** | κ ≪ λ² solo en un factor 5. Ω = 0.2 ω_b. |
| Bin 2012.10926 (circuito) [TEXTO] | 5 GHz | 0.2 | 0.04 (texto) o 0.002 (pie Fig. 1) | 1e-4 | 1 (con 0.04) o 20 (con 0.002) | Parámetros inconsistentes entre texto y pie de figura. Maestra vestida. |
| Spinnler 2311.05342 (QD InAs, cristal fonónico) [TEXTO] | 1.466 GHz | **0.002** (g_ep/2π = 2.9 MHz) | 4.8e-4 (Q_m = 2.1e3) | **0.092** (τ_R = 1.18 ns) [DERIVADO] | 0.008 | Resolved-sideband (γ<ω_b) con λ 50× menor que Bin. |
| DeCrescent 2205.01277 (QD InAs, SAW GaAs) [TEXTO] | ≈3.6 GHz | 3.3e-4 (g₀/2π = 1.2 MHz) | 1.1e-4 (411 kHz a 3.79 GHz; ≤1 MHz en general) | no extraído | ≈1e-3 | Sub-natural sideband. g₀ es "single-phonon optomechanical". |
| Kettler 2105.04223 (QD en microhilo) [TEXTO + DERIVADO] | **330 kHz** | **0.21** (g_m/2π = 68 kHz) | 6e-4 (Q = 1650) | **≈480** (τ_QD ≈ 1 ns) | **≈70** | κ ≪ λ² y λ≈0.2 **sí se cumplen**, pero γ ≫ ω_b: régimen no resuelto, no apto para el esquema de bundles (que requiere γ ≪ ω_b). |
| Imany 2112.09267 (SAW GaAs + QD InAs) [RESUMEN] | GHz | — | — | — | — | Solo abstract: no tengo parámetros. Es SAW, no nanobeam. |
| Ion Ca⁺, Lo et al. 1412.7100 [TEXTO + DERIVADO] | 2.1 MHz | ≈0.003 (fuerza Ω/2π = 13.25 kHz; λ = Ω/2 [DERIVADO]) | ≈10⁻⁶ (10 cuantos/s de calentamiento, aproximado) | spin dephasing: coherencia 180 µs, γφ ≈ 5.6e3 s⁻¹ | ≈13 con κ_heat, pero γφ ≫ λ²/ω_b | λ²/ω_b ≈ 130 s⁻¹ < γφ: el dephasing del spin mata el esquema. η ≈ 0.05. |
| Ion atrapado, Lv 1711.00582 [TEXTO] | ω_X = 2π·2.498 MHz | g/ω_m hasta 2.0 (con detunings efectivos de 10–40 kHz) | no extraído | no extraído | — | Es una **simulación** del modelo de Rabi (sidebands JC+anti-JC), no un acoplo longitudinal real. |
| Sr atrapado, Deng 2011.03886 [TEXTO] | hasta 2π·100 kHz | g_x/ω ≈ O(0.1–1) por construcción | κ_d/g_x = 0.005 | γ_e/g_x = 1.1e-8 | — | Acoplo transversal efectivo. Propuesta teórica. |
| Harpt 2407.08869 (qubit QD híbrido, longitudinal paramétrico) [TEXTO + DERIVADO] | 1.304 GHz | ≈2e-5 (g̃∥/2π ≈ 25 kHz, estimado de Γ_meas ≈ g²/κ ≈ 5 kHz) | 9.6e-5 (κ/2π = 125 kHz) | no extraído | ≈4e-6 | Longitudinal **paramétrico** (impulsado), débil. |
| Champain 2410.20217 (qubit de carga CMOS, longitudinal paramétrico) [TEXTO] | 526 MHz | no extraído (g₀/2π = 25 MHz es el acoplo carga–fotón) | 0.026 (κ/2π = 13.9 MHz) | no extraído | — | Solo se midió g∥ en figura; no tengo el valor. |
| Circuito σ_z–resonador, Richer–DiVincenzo 1511.06138 [RESUMEN] | — | — | — | — | — | Propuesta de circuito; sin g/ω extraído. |

Lecturas:
- **¿λ ≈ 0.1–0.2 es plausible?** En QD con fonones **discretos de GHz**, los experimentos que encontré tienen λ/ω_b ≈ 0.002 (Spinnler) y 3e-4 (DeCrescent): **la propuesta de Bin (λ = 0.1 en 1 THz) está 50 a 300 veces por encima** [DERIVADO] y es sobre-optimista con la tecnología actual. El caso de Kettler muestra λ/ω_b = 0.21 solo en un microhilo de 330 kHz, donde γ ≫ ω_b.
- **κ ≪ λ²** se cumple en el caso de Bin (R = 5, marginal) y en Kettler (pero con γ ≫ ω_b). No se cumple en Spinnler (R ≈ 0.008) ni en DeCrescent (R ≈ 1e-3).
- En iones, la fuerza dependiente del spin demostrada da λ/ω ≈ 3e-3 [DERIVADO] (limitado por el dephasing del spin, 180 µs).
- Superconductores: no encontré una demostración **estática** de acoplo longitudinal ultrafuerte con g/ω y κ/ω reportados. Los experimentos hallados son paramétricos y débiles. **[NO VERIFICADO]** que exista una demostración con λ/ω_b ≳ 0.1.
- Observación: en Bin 1907.12714, con κ/ω_b = 0.002 < λ²/ω_b = 0.01, el desplazamiento de los niveles inducido por el acoplo (≈λ²/ω_b) supera a κ. Es justo el régimen donde la maestra local deja de ser fiable según la lógica de Beaudoin y Settineri [INFERENCIA mía, no citada en las fuentes].

## (iv) Novedad: riesgos y veredicto

Búsquedas hechas en arXiv (títulos y abstracts): "bundle" con "purity" y "analytical/closed-form"; "N-phonon" con "purity"; "phonon bundle"; "bundle emission" con "Stark shift"; "Franck–Condon" con QD/cavidad/fonón/bundle; "shake-off"; y las combinaciones sobre maestra polarónica o longitudinal. Las consultas con paréntesis y OR no funcionan en la API y no devuelven nada; las reformulé sin OR.

| Elemento | Lo más cercano | Estado |
|---|---|---|
| Pureza en forma cerrada | 2609.07454: fuga ∝ g²/Ω² en emisores de dos niveles [TEXTO]; Bin 2020: Ω_eff^(n) = Ω(λ/ω_b)^n/√n! perturbativo [TEXTO] | Ninguno da una pureza cerrada para bundle de fonones con acoplo longitudinal. |
| Cota P ≤ e^{−λ²} (Franck–Condon) | Nada hallado. El efecto aparece como "Franck–Condon blockade" en transporte molecular (p.ej. 1708.01223, solo [RESUMEN] de la búsqueda) y el factor de Debye–Waller es estándar en iones | Sin precedente en bundles que yo haya visto. |
| κ* óptimo | Bin 2020: "sweet point κ ≈ 10 Ω_eff" (pie de Fig. 4) [TEXTO] y barridos de κ/ω_b en Fig. 5 | **Riesgo**: existe una receta numérica; el κ* analítico debe compararse con ella. |
| Comparación local y vestida | Bin 2021 usa vestida pero no la compara con la local; Cosacchi compara fenomenológica y microscópica (otro tipo) | No encontré una comparación sistemática en bundles. |
| Desplazamiento Stark dinámico | Bin 2020 dice que las resonancias se desplazan a lo largo de Δ_n(Ω) [TEXTO], sin fórmula exacta en el texto principal. El Suppl. no pude verlo. | **Riesgo** en el Suppl. de Bin (no accesible). |

**Veredicto (honesto).** No he encontrado un trabajo que contenga a la vez pureza en forma cerrada, la cota e^{−λ²}, un κ* analítico y la comparación local–vestida para bundles de fonones con acoplo longitudinal. El conjunto parece nuevo con respecto a lo que pude leer, pero no garantizo exclusividad:
1. La cobertura es solo arXiv, con la API limitada a consultas simples.
2. No pude leer el Suppl. de Bin 2020 (que podría tener una expresión analítica de Ω_eff y de los desplazamientos) ni el artículo de PRResearch 2025 sobre "two-photon decay enhanced even photon bundle emission".
3. Debe citarse con cuidado la fórmula perturbativa de Bin y la receta κ ≈ 10 Ω_eff, y mostrar que la fórmula exacta y el κ* analítico las generalizan o las corrigen.
4. El resultado de plataformas es incómodo: λ = 0.1 no está respaldado por datos experimentales de QD actuales (≈0.002), y el único caso con λ ≈ 0.2 (Kettler) no es resolved-sideband. La motivación debe replantearse como propuesta de futuro o como caso de iones/circuitos, con λ realista declarado.

Archivos descargados de apoyo: `/home/user/nphonons-bundles/nuevo/lit/pdf/`.
