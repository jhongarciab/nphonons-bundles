# Informe de literatura: ruido TLS correlacionado entre modos acústicos y memoria fonónica de doble carril

Fecha: 2026-11-02. Fuentes: arXiv (PDF completo descargado y leído con pdftotext, salvo indicación).
Etiquetas: **[TEXTO]** = pasaje leído en el PDF completo; **[RESUMEN]** = solo abstract/metadatos de la API; **[NO VERIFICADO]** = de memoria o inferido, no comprobado en fuente; **[DERIV. PROPIA]** = derivación mía a partir de ecuaciones verificadas (no figura en la fuente).
Limitación: la búsqueda por la API de arXiv es por palabras clave y ruidosa; puede haber omisiones. Las referencias de revista que la API no devolvió están marcadas "journal no verificado".

---

## PARTE A — Novedad

### A.1 Los ~12 trabajos más cercanos

| # | arXiv | Título (autores, año, revista) | Qué hace | Qué NO hace respecto a nuestra hipótesis |
|---|---|---|---|---|
| 1 | 2208.13410 | *Correlated frequency noise in a multimode acoustic resonator*. Tubsrinuan, Cole, Delsing, Andersson. 2022/2024. PRB 109, L241301 (2024) [TEXTO] | 14 modos SAW (GaAs, 2.37–2.40 GHz, FSR ≈ 2 MHz) medidos 7 h simultáneamente. Ruido de frecuencia por TLS correlacionado, con correlación (Pearson, Ec. 3) que cae con la desintonía "de la unidad a cero" en la ventana tranquila. Argumenta que, por "gran solapamiento en el espacio real", todos los modos ven el mismo ensemble de TLS con distinta desintonía (Ec. 1). Ráfagas lentas anticorrelacionadas (signo cambia a ~18 MHz): sugieren "un efecto geométrico derivado de la distribución espacial de uno o pocos defectos y la variación del campo acústico con el número de modo" (sin modelarlo). | No modela el perfil de deformación (strain) de los modos como causa de la correlación TLS; no propone ni evalúa codificación de qubit; ondas estacionarias SAW en superficie, no guía 1D con transductor + transmon. |
| 2 | 2501.12056 | *Time-resolved spectral diffusion of a multimode mechanical memory*. Fiaschi, Scarpelli, Korsch, Zivari, Gröblacher. 2025. PRL 134, 243604 (2025) [TEXTO] | Dos modos adyacentes (~5 MHz, modos Fabry-Pérot de guía fonónica "con perfil espacial similar") medidos en tiempo real. Frecuencias **no correlacionadas** (AA y BB correlacionados, AB no). Modelo Monte Carlo: baño de N TLS con acoplos g_{i,A} ≠ g_{i,B}, "hasta dos órdenes de magnitud" de diferencia por TLS. Discute explícitamente que en un esquema dual-rail la coherencia sería similar a la de un modo (no hay protección). | Es la evidencia más contraria a Hitchcock (Φ≈0.6–0.7 a 4 MHz). No explica la diferencia de acoplos por el perfil de strain (solo la postula); no hay qubit ni propuesta de memoria; mide a ~5 GHz y T = 50–800 mK con detección óptica. |
| 3 | 2501.08289 | *Frequency Fluctuations in Nanomechanical Resonators due to Quantum Defects*. Maksymowych, Yuksel, Hitchcock, …, Safavi-Naeini. 2025. Sin journal en API (preprint) [TEXTO] | Cristal fonónico en LN, 450 MHz–1 GHz. Señal telegráfica (RTS) de un único TLS; fórmula de acoplo g_x = (1/ħ)(Δ0/ε) γ:ξ_zp(**r**) con ξ_zp dependiente de la posición (S16); desplazamiento dispersivo y correlador (S20–S21). | Un solo modo por dispositivo; no estudia correlación entre modos ni perfil espacial como determinante de la correlación; no memoria. |
| 4 | 2610.18778 | *Deterministic and programmable generation of mechanical multipartite entangled states*. Franzen, Kern, Rahman, Kladarić, Marti, Chu, Wei, Fadel. 2026 (oct). Sin journal [TEXTO parcial] | HBAR (AlN, FSR 12.7 MHz, 4 modos de 5.023–5.061 GHz, T1,k = 75/82/57/117 µs, T2* = 149/159/112/216 µs; transmon T1 = 28 µs). Estados W, GHZ, cluster por isometrías de control óptimo. "Los estados W son libres de decoherencia bajo desfase colectivo"; sensado de ruido correlacionado con W vs GHZ: tasa de desfase colectivo Γc < 0.7 ms⁻¹ (95 %), "decoherencia dominada por procesos locales no correlacionados", T2* ≃ 2T1. | Modos separados 12.7 MHz (no 4 MHz); no es un qubit de doble carril con memoria/chequeo de borrado; no atribuye la ausencia de correlación a TLS ni al perfil de strain; no propone el encoding como memoria. |
| 5 | 2303.00730 | *Engineering phonon-phonon interactions in multimode circuit quantum acousto-dynamics*. von Lüpke, Rodrigues, Yang, …, Chu. 2023. journal no verificado [RESUMEN] | Interacción tipo beam-splitter sintonizable entre modos de un HBAR mediada por transmon parametricamente bombeado (pares y tríadas). | Provee la herramienta (gates entre modos) pero no analiza ruido correlacionado ni codificación. |
| 6 | 1804.07426 | *Creation and control of multi-phonon Fock states in a bulk acoustic wave resonator*. Chu et al. 2018. journal no verificado [RESUMEN] | Fock multifonónicos en HBAR + transmon. | Un modo; no ruido entre modos. |
| 7 | 1703.00342 | *Quantum acoustics with superconducting qubits*. Chu et al. 2017 (journal no verificado) [RESUMEN] | Primer acoplo qubit–HBAR en régimen cuántico. | No ruido correlacionado; no doble carril. |
| 8 | 2412.08006 | *A mechanical quantum memory for microwave photons*. Bozkurt, Golami, Yu, …, Mirhosseini. 2024. journal no verificado [TEXTO parcial (apéndices TLS)] | Nanomecánico de silicio, T1 ≈ 25 ms a 5 GHz, Q ≈ 0.8×10⁹; desacoplo dinámico vía qubit. Apéndice: ruido de frecuencia por TLS único, PSD Lorentziana con γ1 = 0.1 mHz, ρ_TLS ≈ 14 GHz⁻¹. | Un modo; ni correlación entre modos ni doble carril. |
| 9 | 2307.03169 | *Demonstrating a superconducting dual-rail cavity qubit with erasure-detected logical measurements*. Chou, Shemma, McCarrick, Chien, …. 2023 (Nat. Phys. 2024: journal no verificado) [RESUMEN] | Qubit dual-rail en dos modos fotónicos de cavidad; detección de borrado. | Fotones EM, no fonones; el ruido relevante es la pérdida, no TLS correlacionado; sin análisis de correlación de ruido de frecuencia. |
| 10 | 2311.04423 / 2406.14621 | Koottandavida, Tsioutsios, Kargioti, Smith, …(2023) *Erasure detection of a dual-rail qubit encoded in a double-post superconducting cavity*; de Graaf, Xue, Chapman, Teoh, …(2024) *A mid-circuit erasure check on a dual-rail cavity qubit…* [RESUMEN] | Variantes del dual-rail en cavidades 3D con chequeo de borrado a mitad de circuito (JPNS). | Idem: fotónico; muestran el chequeo de borrado que usaría una versión fonónica, pero no existe análogo acústico en lo que encontré. |
| 11 | 2505.12937 | *Doubling Qubits in a Trapped-Ion System via Vibrational Dual-Rail Encoding*. Kang, Chen, Kwon, Kim. 2025 [RESUMEN] | Propone/realiza qubit dual-rail codificado en dos modos vibracionales de iones que comparten un fonón. | Es el antecedente conceptual "fonón dual-rail", pero en iones atrapados, no en acústica piezoeléctrica; no TLS. |
| 12 | 2204.05066 / 2005.09311 / 2512.16582 / 2307.05544 | Zivari et al. (2022, Sci. Adv. 8, eadd2811) fonones viajeros; Bienfait et al. (2020) borrado cuántico SAW; Xiao et al. (2025) átomo gigante en guía de LN; Crump, Välimaa, Sillanpää (2023, APL 123, 134004) HBARs acoplados [RESUMEN] | Transporte/interferometría de fonones viajeros, guía de LN acoplada en dos puntos (retardo 125 ns), multimodo HBAR. | No ruido TLS entre modos. No encontré ninguna propuesta de **anillo/pista de ondas viajeras acoplado a qubit** ni **memoria time-bin fonónica** en arXiv con mis consultas (ver abajo). |

Consulta específica (iii) anillo/pista: solo aparecieron ítems ópticos/acusto-ópticos (p. ej. 2405.06274, 2603.27711, solo título) → sin resultados relevantes con qubit. Consulta (iv) time-bin fonónica: sin resultados relevantes (2204.05066 y 2005.09311 son lo más cercano). **[NO exhaustivo]**.

### A.2 Valoración honesta de novedad

1. **"El perfil espacial de los modos determina la correlación del ruido TLS entre modos"**: la *idea cualitativa* ya está insinuada en la literatura: (a) Tubsrinuan et al. (Ec. 1 + "gran solapamiento espacial" para correlación; "efecto geométrico… campo acústico con el número de modo" para anticorrelación) [TEXTO]; (b) Fiaschi et al. modelan g_{i,A} ≠ g_{i,B} por TLS y atribuyen la descorrelación a ello [TEXTO]; (c) Maksymowych formula g ∝ ξ_zp(**r**) [TEXTO]. **No encontré** un tratamiento cuantitativo que derive la correlación de Pearson entre modos a partir de los perfiles de strain (ondas estacionarias vs viajeras) y la densidad/posición de TLS, ni la comparación explícita ondas viajeras (perfil |ξ|² uniforme → acoplos g² idénticos para todos los modos) vs estacionarias. Eso sería la parte potencialmente novedosa. Los datos existentes son además **contradictorios** (Hitchcock Φ≈0.6–0.7 a 4 MHz y Tubsrinuan ρ→1; Fiaschi ≈0 a 5 MHz; Franzen: colectivo ≈0 a 12.7 MHz), lo que es a la vez oportunidad y riesgo.
2. **Memoria dual-rail fonónica con chequeo de borrado**: no encontré una propuesta para modos acústicos piezoeléctricos. Existen: dual-rail fotónico con chequeo de borrado (Chou, Koottandavida, de Graaf), dual-rail vibracional en iones (Kang), la discusión de "dual-rail" con dos dispositivos acústicos en Fiaschi (conclusión: sin protección si el ruido no está correlacionado) y estados W libres de desfase colectivo con medida de correlación en HBAR (Franzen). La novedad residual es la combinación: doble carril en modos adyacentes de guía + diseño que maximice la correlación (ondas viajeras / perfiles de strain solapados) + chequeo de borrado vía transmon. **Aviso**: Franzen (oct 2026) y Fiaschi muestran que la premisa "ruido de fase colectivo" puede ser falsa en estos dispositivos; el beneficio del subespacio libre de decoherencia debe cuantificarse con Φ medida.

### A.3 Riesgos de novedad
- R1: la descripción "TLS común + g_i distintos por modo → correlación dependiente de desintonía" ya aparece en Tubsrinuan (2024) y Fiaschi (2025); hay que citar y diferenciarse mediante un cálculo cuantitativo con perfiles de strain.
- R2: Fiaschi discute explícitamente dual-rail acústico y concluye que no ayuda para modos adyacentes (5 MHz) → posible crítica de "ya refutado".
- R3: Franzen et al. cubren "subespacio libre de decoherencia entre modos acústicos (W)" y la cota Γc (<0.7 ms⁻¹); reduce novedad del concepto de DFS, pero en modos con 12.7 MHz.
- R4: "Fonón dual-rail" existe en iones (Kang 2025); "dual-rail con borrado" existe en fotones.
- R5: no verificadas autoría/revista de varios (Chu 2017/2018, von Lüpke, Bozkurt, Chou); búsquedas pueden haber omitido trabajos de 2025–2026 (p. ej. 2610.10700 *Millisecond phonon lifetimes in piezoelectric nanomechanical resonators*, Maksymowych-grupo, solo [RESUMEN] y no leído).

---

## PARTE B — Parámetros TLS (con fuente y estado)

### B.1 Densidad y acoplo

| Magnitud | Valor | Fuente | Estado |
|---|---|---|---|
| P0 "típico de sólidos desordenados" | 1×10⁴⁴ J⁻¹m⁻³ (escrito "/Jm" en el PDF; unidad entendida como J⁻¹m⁻³) → ρ_TLS = V_h P0 ≈ 6.6 TLS/GHz para V_h ≈ 0.1 µm³ (capas de 5 nm en LN dañado, óxido Al, interfaz) | Maksymowych 2501.08289, Supl. | [TEXTO] |
| P_D superficie de Si amorfo | ≈ 5×10⁴⁴ (J m³)⁻¹; capa amorfa ~3 nm; V = 0.04 µm³ → ρ ≈ 14 GHz⁻¹ (δ/2π = 70 MHz entre TLS); Si volumétrico monocristalino con baja densidad de TLS | Bozkurt 2412.08006, Supl. | [TEXTO] |
| Densidad TLS en uniones AlOx | σ ≈ 0.4–0.5 /(GHz µm²) ≈ 10² /(µm³ GHz) (barrera 2–3 nm); dipolos máx. p_max ≈ 6–8 D; otras uniones 2.4/(µm² GHz) | Müller–Cole–Lisenfeld 1705.01108 | [TEXTO]. Conversión propia: 10²/(µm³GHz) ≈ 1.5×10⁴⁴ J⁻¹m⁻³ [DERIV. PROPIA] |
| Dipolo eléctrico típico | máximo de distribución 1–3 D, hasta ~8.3 D (Sarabi); TLS AlOx modelados ≈0.6 e·Å | 1705.01108 | [TEXTO] |
| Potencial de deformación γ | ≈1 eV (sílice, usado también para LN); strain RMS ≈ 1.5×10⁻⁹ → g_x/2π ≈ 0.36 MHz (Δ0/ε≈1); máx. observado en LN g_x/2π ≈ 2 MHz | 2501.08289 | [TEXTO] |
| Nanohaz Si | |γ̃·S̃|/h ≈ 1 MHz con strain ≈ 4×10⁻⁹ | 2412.08006 | [TEXTO] |
| Pérdida de LN | superficie: δ_TLS×d = (1.05 ± 0.47)×10⁻² (d = espesor de capa dañada); volumen: F δ_TLS⁰ = (8.78 ± 0.67)×10⁻⁸ (F = fracción de llenado); BAW Q_i hasta 6×10⁷ | Gruenke-Freudenstein et al. 2501.08291 | [TEXTO] (unidades de d no verificadas, probablemente nm) |
| P0 de SiO2, AlN, LiNbO3 (volumen), interfaces Si/LN | — | — | **[NO VERIFICADO]** no obtuve valor de fuente verificada. De memoria: SiO2 ~10⁴⁵ J⁻¹m⁻³ y tanδ ~10⁻³ (Phillips); no usar sin comprobar. |

### B.2 Relación pérdida–TLS (modelo estándar, tal como aparece)
- Müller et al. Ec. (20): `1/Q_{i,el} = π|p|² D0 / (3 ε_i)`, `1/Q_{i,ph} = π|γ|² D0 / (2 ρ v²)` (D0 = densidad de estados TLS constante, ε_i permitividad, ρ densidad másica, v velocidad del sonido, p y γ momentos dipolares eléctrico y elástico) [TEXTO; la transcripción del PDF de la 2.ª ecuación es ambigua en factores, verificar en la versión publicada].
- Ec. (22): `1/Q_int = Σ_i p_i tanδ_i · tanh(ħω_R/2k_BT) / (1+P_int/P_c)^β + tanδ_0` [TEXTO]. Análoga acústica (Gruenke-Freudenstein Ec. 3): `1/Q_i,tot = Fδ⁰_TLS tanh(ħω_r/2k_BT) / (1+⟨n⟩/n_c)^β + 1/Q_i,res`, con β = 0.5, n_c = 10⁶ asumidos [TEXTO].
- Ec. (23), corrimiento de frecuencia del baño: `Δε/ε = −(2D0 p²/3ε) Re[Ψ(1/2 + ħω/(2πi k_BT)) − log(ħω/k_BT)]`, con `Δf/f = −½ Δε/ε` [TEXTO].
- Ec. (9): `D(E) = ∫P dΔ0 = P0 ln(2E/Δ0min) ≈ D0` [TEXTO]; Ec. (8) distribución P(E,Δ0)=P0 E/(Δ0 √(E²−Δ0²)).

### B.3 Desplazamiento dispersivo y difusión espectral
- Tubsrinuan Ec. 1 (Jaynes-Cummings, régimen dispersivo): `f = f_r + g²/(f_r − f_TLS) σ̂_z` [TEXTO]. "Efecto neto = suma de desplazamientos dispersivos de todos los TLS".
- Maksymowych (S20): `H'_disp/ħ = (Δ + (gx²/Δ_ε,m) σ_z)(a†a + ½) …` con condición g_x/Δ_ε,m ≪ 1; si el TLS salta σ_z = ±1 a tasa Γs: `⟨δω0(τ)δω0(0)⟩ = A² e^{−2Γs τ}`, `A = 2gx²/Δ_ε,m` (S21); PSD Lorentziana `S_RTS(ν) = A²/(2Γs) · 1/(1+(πν/Γs)²)` (tomado del texto principal) [TEXTO].
- Bozkurt (J12): `S(ω) ∝ 2γ1/(γ1² + ω²)` [TEXTO].
- Interacción TLS–TLF (fluctuador térmico) como origen de la difusión espectral: `H_TLS-TLF = J Σ_z Σ_z^th`, `J ~ γ:γ^th/(ρ c_s² R³)` (S22–S23) [TEXTO]; Müller Ec. 17 `H = g_TLS σ_z^(p,1) σ_z^(p,2)` [TEXTO]. Burnett (Ec. 1): `S_y ∝ 1/(√(1+Ω²/Γ1Γ2)) · 1/T^{1+µ} · 1/f`, Γ2, Γ1 ∝ T^{1+µ}, µ ≈ 0.2–0.4 (0.34 medido), densidad de estados ρ(E) ∝ E^µ [TEXTO; Nat. Commun. 5:4119 (2014), arXiv:1311.1655].
- Müller: desplazamiento dispersivo "∝ ±g²/Δf según el estado del TLS"; la difusión espectral de TLS casi resonantes produce fluctuaciones discretas de frecuencia; ruido de fase domina sobre amplitud hasta 30 dB; Sy(f)∝f⁻¹ (Sφ∝f⁻³); S ∝ P_int^{-1/2}; ∝ T^{-1-µ} [TEXTO].

### B.4 Escalas temporales (T ~ 10–50 mK)
- Silicio nanohaz, 1 TLS dominante: γ1 = 0.1 mHz (Bozkurt) [TEXTO].
- LN cristal fonónico 10 mK: Γs/2π ≈ 1.8–2.2 kHz (α), término cuadrático β ≈ 60–117 kHz/K²; salto A/2π ≈ ver Fig. 2 (no extraído); RTS de un único TLS; 100× menos ruido a 1 K por exceder el ancho de banda de la cavidad [TEXTO].
- Conmutación térmica de TLS en resonadores/qubits: 75 µHz–2 mHz (cita en Tubsrinuan, no leída en origen: [TEXTO de segunda mano]); horas (Schlör, transmon) y cambios de ±100 % en T1 en minutos–horas (Müller, §IV) [TEXTO].
- Ancho de difusión espectral ∝ log T (Müller; Burnett/Faoro-Ioffe) [TEXTO]. Faoro–Ioffe (PRB 91, 014201, 2015) y Phillips / Anderson–Halperin–Varma: **[NO VERIFICADO]** (no descargados).
- Saturación: con 10⁴ fonones (Hitchcock) los TLS resonantes están saturados y domina el ruido de TLS lejanos (Fiaschi también: resultado igual a 800 mK y MC a 4 K) [TEXTO].

---

## PARTE C — Definición de Φ en Hitchcock et al., arXiv:2502.16426 [TEXTO]

**Modelo de campo (Sec. III, Ec. 3–6):**
- (3) `a(t) = a0 e^{−κt/2} e^{−iϕ(t)}`, `ϕ(t) = ∫0^t δω(t')dt'`.
- (4) `⟨|a(t)|²⟩ = |a0|² e^{−κt}` (energía; solo depende de κ).
- (5) `|⟨a(t)⟩|² = |a0|² e^{−κt} |⟨e^{−iϕ(t)}⟩|²` (amplitud promediada antes del módulo).
- (6) correlación a tiempos iguales: `|⟨a_j(t)a_k*(t)⟩| = |a_{j,0}a_{k,0}*| e^{−(κ_j+κ_k)t/2} |⟨e^{−i[ϕ_j(t)−ϕ_k(t)]}⟩|`.
- Ajuste de la amplitud: `B0 exp(−κt − Γt − Δ²t²) + B_bg` con κ fijo del ajuste de energía. "Γ y Δ surgen de procesos de fluctuación de frecuencia rápidos y lentos respectivamente (Apéndice H)". Definición: `(ΔTϕ)² + ΓTϕ = 1`, i.e. tiempo en que las coherencias caen a e⁻¹ por ruido rápido+lento.

**Apéndice H (procesos de ruido):** `⟨a(t)⟩ = a0 e^{−κt/2}⟨e^{−iϕ(t)}⟩` (H2) y expansión de Taylor `⟨e^{−iϕ}⟩ = Σ (i^k/k!)⟨ϕ^k⟩` (H3). Régimen rápido (τ ≪ t, τ tiempo de correlación): ⟨ϕ²⟩ ∼ t (ruido blanco, decaimiento exponencial); lento (τ ≫ t): ⟨ϕ²⟩ ∼ t² (decaimiento cuadrático/gaussiano en t). El ruido telegráfico aleatorio (RTN) coincide con el gaussiano en conmutación rápida; en conmutación lenta produce oscilaciones coherentes que a tiempos cortos se expanden en términos cuadráticos en t. (No hay ecuación del RTN en el apéndice: solo cita [42].)

**Apéndice J (Φ):**
- (J4–J5) `⟨e^{−iϕ(t)}⟩ = 1 − ⟨ϕ(t)²⟩/2 + … = e^{−f(t)}` (ϕ de media cero).
- (J6–J7) `⟨e^{−i(ϕ1−ϕ2)}⟩ = 1 − ⟨ϕ1²⟩/2 − ⟨ϕ2²⟩/2 + ⟨ϕ1ϕ2⟩ + … = e^{−f12(t)}`.
- (J8) `f12(t) = f1(t) + f2(t) − (⟨ϕ1(t)ϕ2(t)⟩ + …)`; si no hay correlación f12 = f1+f2; con correlación f12 < f1+f2.
- (J9) `|⟨a1a2*⟩| / √(⟨|a1|²⟩⟨|a2|²⟩) = e^{−f12(t)} = e^{−1}` define T_{ϕ,xc}.
- (J10) `|⟨a1⟩⟨a2*⟩| / √(⟨|a1|²⟩⟨|a2|²⟩) = e^{−f1(t)−f2(t)} = e^{−1}` define T_{ϕ,I} (desfase independiente).
- **(J11) `Φ = 1 − T_{ϕ,I}/T_{ϕ,xc}`**; Φ→1 correlación perfecta, Φ→0 independiente, Φ<0 anticorrelación.
- Experimento: ajuste de la correlación cruzada a la misma forma funcional que la amplitud; Φ promediada sobre ocurrencias y **nueve** frecuencias del modo de microondas (ωc), 19 modos en la matriz (Fig. 4d); la diagonal Φ=1 por definición; Fig. 4e: Φ promediada por desintonía Δω, con gran varianza y descenso medio con Δω. Las barras de error = desviación estándar de todos los pares. Los casos Φ=1 / Φ=0 de las Figs. 4a,b se simulan con la media de las caídas de energía / amplitud.

**Relación con el coeficiente de Pearson:** el paper **no menciona Pearson ni lo relaciona con Φ** (búsqueda en texto: sin coincidencias) [TEXTO]. Φ está definida sobre las fases acumuladas, no sobre δω. Derivación propia [DERIV. PROPIA, hipótesis gaussiana, varianzas iguales, ρ = ⟨ϕ1ϕ2⟩/σ²]: f_k = ⟨ϕ_k²⟩/2, f12 = ½⟨(ϕ1−ϕ2)²⟩ = σ²(1−ρ), f1+f2 = σ². Entonces
 - ruido rápido/blanco (f ∝ t): T_{ϕ,xc} = T_{ϕ,I}/(1−ρ) ⇒ **Φ = ρ**;
 - ruido lento (f ∝ t²): T_{ϕ,xc} = T_{ϕ,I}/√(1−ρ) ⇒ **Φ = 1 − √(1−ρ) ≤ ρ** (Φ = 0.6 ⇒ ρ ≈ 0.84);
 - para ruido telegráfico de un único TLS acoplado a ambos modos, δω_j = A_j σ_z(t) con A_j = 2g_j²/Δ_j (S21 de Maksymowych): |ρ| = 1 independientemente de |A1/A2|; el signo lo fija sign(Δ1Δ2) (anticorrelación si el TLS está entre los modos). Con varios TLS, ρ = Σ_i A_{i1}A_{i2}s_i / √(Σ A_{i1}² s_i · Σ A_{i2}² s_i), s_i = varianza del RTS i; por tanto ρ<1 por la competencia de los acoplos g_{ij} ∝ ξ_j(**r**_i) y de Δ_{ij}. Esto es hipótesis de trabajo, no verificada contra datos.
 - Nota [DERIV. PROPIA]: para TLS distribuidos uniformemente en una guía 1D con ondas estacionarias, g² ∝ cos²(k_j x): la covarianza espacial de cos²(k1x) y cos²(k2x) con k1≠k2 (armónicos ortogonales) es nula, mientras que para ondas viajeras |ξ|² es uniforme y los g² son idénticos entre modos. La correlación efectiva depende entonces de cuántos TLS dominan (N_eff pequeño ⇒ ρ alto, como infieren Hitchcock y 2208.13410).

**Números del texto sobre Figs. 3–4 (y dispositivo):**
- Fig. 3(a): modo ω_a/2π = 2.7996 GHz, microondas desintonizado ≈ 50 MHz; T1,in = 534 µs, Tϕ = 107 µs, ⟨n⟩ ≈ 1×10⁴; ωc/2π ≈ 2.85 GHz fijo para las medidas de 17 modos vs potencia.
- Fig. 3(b): T1,in ∝ ⟨n⟩^0.12, Tϕ ∝ ⟨n⟩^0.26; barras de error = error estándar de 24 medidas repetidas; T1,in supera a Tϕ en ≈ un factor 10 (T1,in/Tϕ ≈ 5 para el ejemplo).
- Fig. 4: ωc/2π ≈ 2.82 GHz; modos ω1 = 2.793, ω2 = 2.826, ω3 = 2.829, ω4 = 2.833 GHz; Δω14/2π = 40 MHz (Φ bajo, "predominantemente no correlacionado"), Δω34/2π = 4 MHz (correlación presente); 19 modos, 9 valores de ωc.
- Dispositivo: guía de Si, L = 700 µm, v_g = 5200 m/s, modos longitudinales 2.6–3.0 GHz, FSR diseñado ≈ 3.7 MHz, medido (3.3 ± 0.2) MHz; γ_WG > ω_FSR; f_k = ω_FSR γ_WG/2π ≈ 5.1 MHz; atenuación proyectada con T1,in = 100 µs, Tϕ = 10 µs: 0.7 y 7 dB/m.
- Valores numéricos de Φ (0.6–0.7 a 4 MHz) **no aparecen en el texto**; solo en la Fig. 4(d,e) (gráfico no leído): **[NO VERIFICADO]**.
- Interpretación del paper: ensemble de TLS distribuidos en frecuencia [32 = Maksymowych 2501.08289]; modos cercanos dominados por un único TLS (correlacionados), lejanos por distintos TLS; coherente con [33 = Tubsrinuan 2024].

---

## Tabla final de referencias verificadas

| id arXiv | Autores (1.º…) | Año | Journal | Estado |
|---|---|---|---|---|
| 2502.16426 | Hitchcock et al. (Safavi-Naeini) | 2025 | no verificado | [TEXTO] (Apéndices H, J, Secs. III–IV) |
| 2208.13410 | Tubsrinuan, Cole, Delsing, Andersson | 2022 | PRB 109, L241301 (2024) | [TEXTO] |
| 2501.12056 | Fiaschi, Scarpelli, Korsch, Zivari, Gröblacher | 2025 | PRL 134, 243604 (2025) | [TEXTO] |
| 2501.08289 | Maksymowych, Yuksel, Hitchcock, …, Safavi-Naeini | 2025 | no publicado (API) | [TEXTO] |
| 2501.08291 | Gruenke-Freudenstein, Szakiel, Multani, …, Safavi-Naeini | 2025 | no publicado (API) | [TEXTO parcial] |
| 2610.18778 | Franzen, Kern, Rahman, Kladarić, Marti, Chu, Wei, Fadel | 2026 | no publicado | [TEXTO parcial] |
| 1705.01108 | Müller, Cole, Lisenfeld (Rep. Prog. Phys. 2019) | 2017 | Rep. Prog. Phys. 82, 124501 (2019): journal de memoria, no verificado | [TEXTO] |
| 1311.1655 | Burnett et al. | 2013 | Nat. Commun. 5:4119 (2014) | [TEXTO parcial] (API confirma journal) |
| 2412.08006 | Bozkurt, Golami, Yu, …, Mirhosseini | 2024 | no verificado | [TEXTO parcial] |
| 1601.06837 | Behunin, Intravaia, Rakich | 2016 | PRB 93, 224110 (2016) | [RESUMEN] |
| 2303.00730 | von Lüpke, Rodrigues, Yang, …, Chu | 2023 | no verificado | [RESUMEN] |
| 1804.07426 / 1703.00342 | Chu et al. | 2018 / 2017 | no verificado | [RESUMEN] |
| 2307.03169 | Chou, Shemma, McCarrick, Chien, … | 2023 | no verificado | [RESUMEN] |
| 2311.04423 | Koottandavida, Tsioutsios, Kargioti, Smith, … | 2023 | no verificado | [RESUMEN] |
| 2406.14621 | de Graaf, Xue, Chapman, Teoh, … | 2024 | no verificado | [RESUMEN] |
| 2505.12937 | Kang, Chen, Kwon, Kim | 2025 | no verificado | [RESUMEN] |
| 2204.05066 | Zivari, Fiaschi, Burgwal, Verhagen, … | 2022 | Sci. Adv. 8, eadd2811 (2022) | [RESUMEN] |
| 2005.09311 | Bienfait, Zhong, Chang, Chou, … | 2020 | no verificado | [RESUMEN] |
| 2512.16582 | Xiao, Zhang, Zeng, Pan, … | 2025 | no verificado | [RESUMEN] |
| 2307.05544 | Crump, Välimaa, Sillanpää | 2023 | APL 123, 134004 (2023) | [RESUMEN] |
| 2301.11213, 2010.01025 | SAW/PnC en LN a baja T (Mater. Quantum Technol. 3, 021001; APL 118, 123501) | 2023/2020 | según API | [RESUMEN] |

No obtenidos/NO VERIFICADOS: Faoro–Ioffe PRB 2015; Phillips (Rep. Prog. Phys. 1987); Anderson–Halperin–Varma (1972); valores P0 de SiO2, AlN, LN volumétrico.
