# Línea base y estado del arte: memoria fonónica de doble carril con chequeo de borrado dispersivo

Fecha de la búsqueda: 2026-11-02. Fuente: arXiv (PDF completos descargados salvo indicación). Etiquetas: **[TEXTO]** = leído en el PDF completo; **[RESUMEN]** = solo abstract; **[DERIVADO]** = cálculo mío a partir de números [TEXTO]; **[NO VERIFICADO]** = no comprobado en un documento descargado.

## 0. Correcciones importantes al encargo

1. **arXiv 2204.05066 NO es de Cleland.** Es Zivari, Fiaschi, Burgwal, Verhagen, Gröblacher et al. (Delft/AMOLF), *Sci. Adv.* 8, eadd2811 (2022): fonones viajeros en una guía **optomecánica** (silicio, ~5,154 GHz) con entrelazamiento time-bin fotón–fonón. No hay qubit superconductor [TEXTO]. El trabajo de Cleland sobre fonones viajeros es otro (Bienfait 1903.05672; Dumur 2201.00729; Qiao 2503.03898).
2. **Hitchcock 2502.16426:** el abstract dice Tφ ≈ 60 µs; el valor Tφ = 107 µs (con T1,in = 534 µs) corresponde a **un modo concreto** (ωa/2π = 2,7996 GHz, ⟨n⟩ ≈ 1×10⁴) de la Fig. 3(a). Además es una **guía de onda** (FSR ≈ 3,7 MHz), medida con ringdown clásico de alta ocupación vía una cavidad de microondas, **sin qubit** [TEXTO]. El T1,in y Tφ a nivel de un fonón no están medidos.
3. **Mirhosseini et al.** = Bozkurt, Golami, Yu, Tian, Mirhosseini, arXiv **2412.08006** (v1, dic 2024). La referencia *Nat. Phys.* 21, 1469 (2025) viene del encargo; la API de arXiv no devuelve journal_ref [NO VERIFICADO la revista]. Los números de abajo son de la v1.
4. Franzen 2610.18778: el autor líder es Isabelle Franzen (Chu/Fadel, ETH) [TEXTO]. Confirma T1q = 28(1) µs, g/2π = 273 kHz, FSR = 12,7 MHz.

## 1. Memorias mecánicas con qubits superconductores (2023–2026)

**Bozkurt et al., 2412.08006 (Caltech).** Transmon galvánicamente conectado a un oscilador nanomecánico de silicio (cristal fonónico) con acople electrostático (no piezo). ωm/2π = 4,92/4,77 GHz (A/B); g_em/2π = 200/230 kHz; T1,m = 19,3/21,1 ms (intrínseco A: 25 ± 2 ms, Q ≈ 0,8×10⁹); T2* = 64/67 µs, T2,echo = 562/268 µs, T2 con CPMG-2 = 1,02 ± 0,15 / 0,54 ms; qubit pobre: T1,q = 1,70/1,55 µs, T2,q* = 1,30/1,05 µs; CT1 ≈ 1,3–1,7×10⁵, CT2 ≈ 132–147. Fidelidad de preparación de Fock: 0,964 ± 0,007 (|0⟩) y 0,711 ± 0,023 (|1⟩), limitada por la decoherencia del qubit. Acople vía swap resonante; los autores indican que el régimen **fuertemente dispersivo no se alcanzó** (lo plantean como trabajo futuro para bosónicos sesgados). Modo único por oscilador. [TEXTO]

**Yang, Kladarić et al., 2406.07360, "A mechanical qubit" (ETH, Chu; Science 386, 783 [2024] según la lista de refs de 2610.18778, no verificado en la fuente).** HBAR + transmon 3D, g/2π = 280 kHz; modo fonónico ωp/2π = 5,049 GHz con T1 = 104 µs, T2 = 205 µs; qubit T1q = 23,8 µs, T2* = 20,4 µs, eco 30,9 µs. Se opera en régimen dispersivo (Δ ≈ −0,7 MHz) para inducir una no linealidad de fonón α ≈ −17,3 kHz; lectura por **RPN resonante** (swap + lectura), no por splitting. [TEXTO]

**von Lüpke et al., 2110.00263 (Nat. Phys. 18, 794 (2022), según la ref. [13] de 2304.13589 [TEXTO de tercero]).** HBAR + transmon, **medición de paridad y number-splitting dispersivo**. g = 259,5 kHz, FSR ≈ 12 MHz; T1q = 10,2 µs, T2* = 10,5 µs, T2e = 11,6 µs; modo LG00: T1 = 81 µs, T2* = 138 µs; χ_Fock/2π = −147 kHz a Δ = −0,8 MHz (picos resueltos, sonda de ancho 10,6 kHz), χ_Ramsey = 70 kHz a Δ = −1,9 MHz; ruido de frecuencia lento del qubit (~10 kHz) comparable a χ. [TEXTO]

**von Lüpke et al., 2303.00730 (Chu).** Interacción *beam-splitter* paramétrica entre modos de HBAR mediada por transmon bombeado; transmon 3D 5,97 GHz, T1q = 9,5 µs, T2* = 7,2 µs; FSR ≈ 12,63 MHz; g_m = 257 kHz; τ_BS = π/(4g_BS) = 8,0 µs, es decir g_BS/2π ≈ 16 kHz [DERIVADO]; interferencia Hong-Ou-Mandel entre fonones y entrelazamiento de dos modos. Es la herramienta que haría falta para el "carril doble" (BS entre modos adyacentes). [TEXTO]

**Franzen et al., 2610.18778 (ETH).** HBAR (AlN), cuatro modos ωk/2π = 5,0232–5,0613 GHz (espaciado 12,7 MHz), T1,k = 75/82/57/117 µs, T2*,k = 149/159/112/216 µs (≈ 2 T1: dephasing puro despreciable), g/2π = 273 kHz, T1q = 28 µs, T2q* = 15 µs; fidelidad de isometría por tomografía de proceso 90,8(4) %; estados W/GHZ/cluster con GME hasta 4 modos. **Cota de desfase colectivo:** Γc < 0,7 ms⁻¹ (95 %), es decir sin evidencia de ruido de fase correlacionado entre modos HBAR; la decoherencia es local y por amortiguamiento. Solo acople resonante (JC); no se reporta χ. [TEXTO]

**Garcia-Belles, Anferov et al., 2602.22117.** HBAR con películas piezo diversas: T1,p hasta 396 ± 11 µs (Q = 1,3×10⁷), T2,p = 806 ± 24 µs (limitado por T1), g0/2π = 298 ± 3 kHz, CT2 = 1,13×10⁵; FSR no constante, dependiente del material (valor no extraído) [TEXTO parcial].

**Omahen et al., 2604.08655 (ETH).** Reset de transmon con HBAR multimodo, FSR ≈ 12,6 MHz, vidas de modo 60–250 µs (un modo de 400 µs), población residual < 10⁻⁴; el transmon se opera dispersivamente durante parte del protocolo. [TEXTO parcial / RESUMEN]

**Otros (solo abstract o tabla de terceros).** Bozkurt 2207.10972 (SOI electromecánico no piezo, g/2π = 1,1 MHz paramétrico, τ hasta 265 µs a 5 GHz) [RESUMEN]; quartz PnC 2509.07900 (100 MHz, vidas de ms a 8 K, acople paramétrico propuesto/evaluado con fluxonium/transmon) [RESUMEN]; Diego/Cleland 2601.13509 (Lamb LN, Q ≈ 6000 a un fonón) [RESUMEN]; Scigliuzzo/Scarlino 2505.24865 (cavidad SAW multimodo + resonador Kerr SQUID, "superstrong") [RESUMEN]; Wollack 2022 (Nature 604, 467) y Arrangoiz-Arriola 2019 (1902.04681, Nature 571, 537) aparecen en la Tabla I de 2304.13589 [TEXTO de tercero].

## 2. Tabla de parámetros transmon/qubit + fonones

| Plataforma (arXiv, año) | f (GHz) | g/2π | χ medido | T1q / T2q | T1 modo / T2 modo | FSR; nº modos | Dispersivo demostrado | Etiqueta |
|---|---|---|---|---|---|---|---|---|
| Si PnC nanomecánico + transmon (2412.08006, 2024) | 4,92 | 200–230 kHz (electrostático) | no resuelto | 1,7 µs / 1,3 µs (T2*) | 19–25 ms / 64 µs (T2*), 1,02 ms (CPMG) | 1 modo por oscilador | solo detección de n grande; **no** strong dispersive | TEXTO |
| HBAR + transmon, "mechanical qubit" (2406.07360, 2024) | 5,05 | 280 kHz | régimen dispersivo (Δ≈−0,7 MHz), α_fonón = −17,3 kHz | 23,8 / 20,4 µs | 104 / 205 µs | n/d | sí (no linealidad); lectura resonante | TEXTO |
| HBAR + transmon, paridad (2110.00263, 2021/22) | ~5 | 259,5 kHz | −147 kHz (Δ=−0,8 MHz); 70 kHz (Δ=−1,9 MHz) | 10,2 / 10,5 µs | 81 / 138 µs | ~12 MHz; varios | **sí, number-splitting** | TEXTO |
| HBAR + transmon, BS multimodo (2303.00730, 2023) | 5,97 (qubit) | 257 kHz | no extraído | 9,5 / 7,2 µs | n/d (ver paper) | 12,63 MHz; 2–3 modos | BS paramétrico g_BS≈16 kHz [DERIV.] | TEXTO parcial |
| HBAR AlN + transmon, multipartito (2610.18778, 2026) | 5,02 | 273 kHz | no reportado | 28 / 15 µs | 57–117 µs / 112–216 µs | 12,7 MHz; 4 modos | no (solo resonante/Stark) | TEXTO |
| HBAR, pérdidas (2602.22117, 2026) | ~5 | 298 kHz | n/d | n/d | hasta 396 µs / 806 µs | FSR variable; muchos | n/d | TEXTO parcial |
| HBAR, reset (2604.08655, 2026) | ~5 | n/d | n/d | n/d | 60–250 µs (400 µs un modo) | 12,6 MHz; ≥5 | parcial | TEXTO parcial |
| Fluxonium + PnC LN (2304.13589, 2023) | 0,69 | 13,5–14 MHz | 2χ = 2,23 MHz (Δ/g≈9–11) | 3,57 µs (T1q) / 0,33 µs | 0,85/4,1/29,6 µs / 3,9 µs | 1 modo | **sí, fuerte** | TEXTO |
| LN SAW, Bell entre 2 resonadores (2411.15726, 2024) | 3,03 / 3,30 | n/d (acoplador variable) | Ramsey dispersivo para tomografía | n/d | T1 380/270 ns; T2 709/527 ns | 1 modo por resonador | parcial (lectura Ramsey) | TEXTO parcial |
| LN SAW itinerante, fase/HOM (2503.03898, 2025) | 3,925 | acopladores variables | n/d | e-f T1 ≈ 1,5 µs | n/a (viajeros) | n/a | n/a; HOM V=0,981±0,007 | TEXTO parcial |
| SOI electromecánico (2207.10972, 2022) | 5 | 1,1 MHz (paramétrico) | n/d | n/d | 265 µs | n/d | n/d | RESUMEN |
| Lamb LN suspendido (2601.13509, 2026) | GHz | n/d | n/d | n/d | Q≈6000 | n/d | n/d | RESUMEN |
| Guía Si+LN (2502.16426, 2025), **sin qubit** | 2,8 | 1,1 MHz (cavidad–transductor) | n/a | n/a | T1,in = 534 µs, Tφ = 107 µs (un modo; abstract: ≈60 µs) | 3,7 MHz; 17–19 modos | n/a | TEXTO |

Lectura: **dispersivo con number-splitting demostrado**: HBAR (2110.00263), fluxonium+PnC (2304.13589) y, según la Tabla I de este último, transmon+PnC (Wollack 2022), transmon+SAW (Sletten 2019) [TEXTO de tercero]. **Solo resonante** (swap/JC): Franzen, Mirhosseini 2024, resonadores SAW de Cleland. El rango χ ≈ 100–300 kHz solicitado tiene un anclaje experimental directo: χ = 147 kHz en HBAR.

## 3. Qubits dual-rail con detección de borrado (calibración del chequeo)

| Ref. | Chequeo / duración | χ | Falsos neg. / pos. | Idle y dephasing | Ancilla / modos |
|---|---|---|---|---|---|
| 2307.03169 Chou (AWS, 2023): medición lógica *end-of-line* | secuencia total 4,8 µs; pulso π selectivo gaussiano σs = 300 ns (σ_us = 16 ns), desplazado χ | χ_cm/2π = −2,14 / −3,72 MHz | error de detección de fuga 7,7×10⁻³ (1 ronda), 1,2×10⁻³ (2 rondas, fracción de borrado 17 %); desasignación 1,8×10⁻⁴; fracción de borrado 6,03 % | tasa de desfase lógico Γφ = 1/(2,2 ms) Ramsey, 1/(2,7 ms) eco (≈0,023 %/µs); bit-flip < 0,002 %/µs (cota 1/66 ms); borrado ≈6× más rápido que el desfase | transmon T1 125/179 µs, T2R 92/206 µs, T2E 144/303; cavidades T1 592/350 µs, T2R 768/543 µs |
| 2406.14621 de Graaf (Yale, 2024): **chequeo JPNS a mitad de circuito** | pulso cuadrado Tp = 1,820 µs con beamsplitter |g_bs|/2π = 2,05 MHz (|g_bs/χ|≈1,04) | FN 3,7(2) %; borrado total 2,92(1) %/chequeo de los que 0,51 % son FP; fracción de borrado omitido 9,0×10⁻⁴ | Pauli por chequeo 0,31(1) % (0,12 % por transmon); tasa de pérdida de fotón del chequeo 2,41 % ≈ idle 2,43 % | χ/2π = 1,07 MHz; transmon ancilla T1 ≈ 42 µs, T2R ≈ 33 µs, T2E ≈ 63 µs (asignación de columna inferida); cavidades T1 347/109 µs, T2R 441/165 µs. Simulación con T1,anc=Tφ=200 µs, T1,cav=1 ms: borrado 0,49 %, Pauli del transmon 0,035 % |
| 2311.04423 Koottandavida (Yale/Devoret, 2023): cavidad de doble poste, χ igualado con bombeo | **pulso gaussiano 4σ = 8 µs**, circuito completo (lectura+reset) 12 µs | χ_aq = χ_bq = 0,52 MHz tras igualar | FN 0,28 %/medida; QND > 99,9 % | borrado 3,981 ms⁻¹ (4,8 %/medida); desfase residual ≤ 0,170 ms⁻¹ (≈0,2 %/medida) | transmon T1 147 µs, T2R 87 µs, T2E 130 µs; cavidades T1 225/299 µs, T2R 452/605 µs |
| 2503.10935 Quantum Circuits (2025): CZ dual-rail | puerta ≈ 500 ns, borrado ≈ 0,53(2) %/CZ; fidelidad Bell 99,60(1) % tras postselección | n/d | chequeos de borrado retardados, no tras cada puerta | estimación en el texto (no medición): cavidades T1 ~ 500 µs, Tφ,DR ~ 5 ms; transmon T1 ~ 50 µs, Tφ ~ 500 µs; asimetría de desfase control/objetivo ≥3,5 | [TEXTO] |
| 2601.21616 Mai (SUSTech, 2026): qubit "biased-erasure" {0,2} en una cavidad (no dual-rail) | ciclo 13,0 µs (intervalo 11,9 µs) | χ_qc = 1,69 MHz | FP 0,22 %; FN 0,69 %; omitido 0,004 % | desfase inducido por el chequeo 0,26 %/chequeo; tras postselección (6,2 ms)⁻¹ y (3,1 ms)⁻¹ | cavidad T1 0,47 ms, T2R 0,74 ms |
| 2510.08416 Dakis/Puri/Barnes (teoría, 2026) | pulsos robustos al desfase del transmon (cuarto orden); ej.: puerta ZZ(π/2) Tg = 15,7 µs para |χ| = 1 MHz | — | reduce errores del chequeo "tres órdenes" (simulado) | FP ∝ γ⁶ vs FN ∝ γ⁴ | teórico [TEXTO] |

**Cómo se manejan los errores de fase en estos trabajos [TEXTO]:** (i) el JPNS excita al ancilla solo si N = 0, de modo que el transmon permanece en |g⟩ dentro del código (2406.14621); (ii) ecos y desplazamiento de la frecuencia del pulso por χ para minimizar la asignación errónea de fuga al código (Chou); (iii) los eventos de transición del transmon se declaran como borrado (Chou, de Graaf) y se usa reset condicional; (iv) pulsos de control robustos al desfase del ancilla (Dakis et al.); (v) el desfase del dual-rail depende del ruido de ancilla (calentamiento, shot-noise), no del tiempo de defase de cada cavidad.

**Orden de magnitud para su modelo [DERIVADO, no medido para fonones]:** con T1 de modo ≈ 100 µs, un chequeo de 6–20 µs aporta un borrado intrínseco ≈ t/T1 ≈ 6–20 %, frente a 4,8 % en 2311.04423 (8–12 µs sobre T1 ≈ 225–300 µs) y 2,4 % en 2406.14621 (1,8 µs sobre T1 ≈ 109–347 µs). Un pulso gaussiano selectivo de 6–20 µs con χ ≈ 100–300 kHz es coherente con el ancho de banda 1/(2πσ) usado por Chou (σ = 300 ns con χ de MHz) y con el pulso 4σ = 8 µs a χ = 0,52 MHz de Koottandavida, pero con χ ≤ 300 kHz el ancho debe ser ≲ χ/3, lo que fuerza durar ≳ 10 µs. Con T1q = 28 µs (Franzen) o 10 µs (von Lüpke) el ancilla decae dentro del chequeo; las demostraciones dual-rail usan T1,anc ≈ 42–180 µs.

## 4. Almacenamiento time-bin de fonones

- **Zivari et al. 2204.05066** (optomecánico, Delft): qubit time-bin fotón–fonón con fonones viajeros en guía de silicio, τ/2 ≈ 63 ns entre bins, visibilidad V = 0,82 ± 0,04, CHSH S = 2,32 ± 0,08 (4σ), ~30 eventos/hora; fidelidad limitada por absorción óptica; eficiencia de recuperación hasta 30 % mencionada como proyección. Rendimiento medido por correlaciones/Bell, no por tomografía completa. [TEXTO]
- **Cleland, SAW itinerante:** Bienfait 1903.05672 (emisión y recaptura de un fonón en canal de 2 mm, ~500 ns de retardo; transferencia de estado y entrelazamiento remoto) [RESUMEN]; Dumur 2201.00729 (transferencia de información cuántica con fonones moldeados entre dos nodos) [RESUMEN]; Qiao 2503.03898 (control de fase determinista de fonones de 1 y 2 cuantos, HOM V = 0,981 ± 0,007, interferómetro Mach-Zehnder con visibilidad ≈ 0,98) [TEXTO parcial]; Chou 2411.15726 (Bell estacionario F = 0,872 ± 0,002, N00N F = 0,748 ± 0,008) [TEXTO parcial].
- **No encontré** (búsqueda de arXiv por palabras clave, limitada) almacenamiento time-bin de un fonón en un resonador piezo con qubit superconductor [NO VERIFICADO: ausencia].
- **Métricas típicas:** fidelidad de estado por tomografía, visibilidad de interferencia/HOM, probabilidad de transferencia (emisión-captura), CHSH. Para su memoria: probabilidad de recuperación = (1 − erasure) × fidelidad postseleccionada, y fidelidad condicionada a "no borrado".

## 5. Valoración honesta de novedad

**Combinación propuesta:** doble carril fonónico en dos modos adyacentes de un resonador multimodo + chequeo de borrado dispersivo JPNS con transmon + diseño de correlación de ruido entre modos.

Antecedentes que cubren partes:
1. **Wang, Qiao, Cleland, Jiang, 2411.00719:** propone *dual-rail* con fonones SAW (itinerantes) para detectar errores de pérdida en un QRAM, con heraldo limitado por el desfase del transmon [TEXTO parcial]. No es una memoria estacionaria ni usa chequeo dispersivo a mitad de circuito.
2. **Hann et al., 1906.11340 (PRL 123, 250501):** QRAM con modos fonónicos y acoples entre modos por bombeo fuera de resonancia del transmon [RESUMEN].
3. **von Lüpke 2303.00730:** *beamsplitter* entre modos HBAR ya demostrado (g_BS ≈ 16 kHz, lento) [TEXTO].
4. **von Lüpke 2110.00263 / Lee 2304.13589:** number-splitting dispersivo con fonones [TEXTO].
5. **Franzen 2610.18778 / Hitchcock 2502.16426:** ya miden correlaciones de ruido de fase entre modos mecánicos (colectivo < 0,7 ms⁻¹ en HBAR; correlación positiva entre modos cercanos en la guía de silicio, Φ decreciente con la separación) [TEXTO].
6. **Mansouri y Arai 2605.10932:** puertas holonómicas fonónicas con canales de borrado sesgado, pero para un centro NV (simulación) [RESUMEN]; no es dual-rail con transmon.

No encontré (búsquedas "dual-rail/erasure + phonon/mechanical/acoustic") un trabajo que combine los tres elementos. Esto es ausencia de evidencia en un sondeo de arXiv de alcance limitado, no una garantía [NO VERIFICADO].

**Lo realmente nuevo (probable):** (a) el chequeo JPNS en un oscilador mecánico multimodo con un único ancilla, (b) la memoria estacionaria con detección de borrado y su presupuesto de errores, (c) el diseño de correlación de ruido como regla de selección de los modos (pares con ruido de fase común). Esto último es lo más frágil.

**Riesgos:**
1. **Ancilla limitante.** T1q medidos con fonones: 1,7 µs (Caltech), 10–28 µs (ETH HBAR). Las demostraciones dual-rail usan ancillas de 42–180 µs. Un chequeo de 6–20 µs con T1q = 28 µs produce falsos positivos y desfase inducido ≫ 0,2–0,5 % por chequeo de los publicados.
2. **JPNS requiere beamsplitter** fuerte entre los dos modos (de Graaf: g_bs ≈ 2 MHz ≈ χ). En HBAR el BS medido es ~16 kHz ≪ χ; sin él, el chequeo debe ser un π selectivo con χ₁ ≈ χ₂ igualados (Koottandavida usó bombeo para igualar) y hay que verificar que ambos modos tengan el mismo χ, algo difícil con g y Δ distintos por modo (FSR de 12,7 MHz).
3. **Borrado intrínseco por chequeo** ≈ t_chequeo/T1 (6–20 % con T1 ≈ 100 µs [DERIVADO]) frente a 2,4–4,8 % en los dual-rail de cavidad; la ventaja de borrado exige T1 de modo más largos (Mirhosseini 25 ms, Garcia-Belles 400 µs) o chequeos más rápidos.
4. **Ruido de frecuencia del qubit** (~10 kHz, von Lüpke) comparable a χ ≈ 100–300 kHz: degrada la resolución de picos y el desfase inducido.
5. **Jerarquía de errores:** el dual-rail aprovecha T2 ≈ 2 T1 (Franzen lo muestra en HBAR), pero en silicio (Hitchcock) Tφ ≈ 60–107 µs < T1: el desfase no es despreciable y la correlación solo ayuda parcialmente (correlación medida a 4 MHz de separación; valor de Φ no extraído aquí). La correlación de ruido se mide en la guía con altas ocupaciones, no a nivel de un fonón.
6. **Correlación de ruido en HBAR:** el límite Γc < 0,7 ms⁻¹ sugiere que el beneficio de diseñar correlación está dentro del error (hay poco ruido colectivo que explotar); novedad posible, valor práctico incierto.
7. **Prioridad:** Mirhosseini (2412.08006) prevé explícitamente qubits bosónicos sesgados en mecánica de GHz; es probable que grupos de Caltech/ETH/UChicago estén cerca. Verificar versión publicada (Nat. Phys. 2025) antes de afirmar que "no existe".
8. **Datos faltantes:** FSR/χ de Garcia-Belles no extraídos; g y T1q de Chou 2411.15726 y Wollack 2022 sin leer; arXiv de Wollack no localizado.

## 6. Referencias verificadas

| arXiv / id | Autores (primeros) | Año | Revista | Estado |
|---|---|---|---|---|
| 2412.08006 | Bozkurt, Golami, Yu, Tian, Mirhosseini | 2024 | Nat. Phys. 21, 1469 (2025) [NO VERIFICADO; dato del encargo] | PDF leído (v1) |
| 2406.07360 | Yang, Kladarić, Drimmer, von Lüpke, ..., Chu | 2024 | Science 386, 783 (2024) [según ref. de 2610.18778] | PDF leído |
| 2110.00263 | von Lüpke, Yang, Bild, Michaud, Fadel, Chu | 2021 | Nat. Phys. 18, 794 (2022) [ref. de 2304.13589] | PDF leído |
| 2303.00730 | von Lüpke, Rodrigues, Yang, Fadel, Chu | 2023 | sin journal_ref en arXiv | PDF leído parcial |
| 2610.18778 | Franzen, Kern, Rahman, Kladarić, Marti, Chu, Wei, Fadel | 2026 | sin journal_ref | PDF leído parcial |
| 2602.22117 | Garcia-Belles, Anferov, Deeg, ... | 2026 | sin journal_ref | PDF parcial |
| 2604.08655 | Omahen, Storz, Kladarić, Chu | 2026 | Phys. Rev. Applied 26, L031005 | PDF parcial |
| 2304.13589 | Lee, Guo, Cleland, ..., Safavi-Naeini | 2023 | sin journal_ref | PDF leído parcial |
| 2411.15726 | Chou, Qiao, Yan, Andersson, ..., Cleland | 2024 | sin journal_ref | PDF parcial |
| 2503.03898 | Qiao, Wang, Andersson, Anferov, ..., Cleland | 2025 | sin journal_ref | PDF parcial |
| 1903.05672 | Bienfait, Satzinger, Zhong, Chang, ... | 2019 | no verificado | RESUMEN |
| 2201.00729 | Dumur, Satzinger, Peairs, Chou, ... | 2021/22 | npj Quantum Inf. 7, 173 (2021) | RESUMEN |
| 2204.05066 | Zivari, Fiaschi, Burgwal, ..., Verhagen, Gröblacher | 2022 | Sci. Adv. 8, eadd2811 | PDF leído parcial |
| 2411.00719 | Wang, Qiao, Cleland, Jiang | 2024 | sin journal_ref | PDF parcial |
| 1906.11340 | Hann et al. | 2019 | Phys. Rev. Lett. 123, 250501 | RESUMEN |
| 2207.10972 | Bozkurt, Zhao, Joshi, LeDuc, Day, Mirhosseini | 2022 | sin journal_ref | RESUMEN |
| 2509.07900 | (autores no extraídos) | 2025 | sin journal_ref | RESUMEN |
| 2601.13509 | Diego, Qiao, Kim, ..., Cleland | 2026 | Phys. Rev. Applied 26, 014045 | RESUMEN |
| 2505.24865 | Scigliuzzo, Peyruchat, ..., Delsing, Scarlino | 2025 | sin journal_ref | RESUMEN |
| 2502.16426 | Hitchcock, Mayor, Jiang, Maksymowych, Malik, Safavi-Naeini | 2025 | sin journal_ref | PDF leído parcial |
| 2307.03169 | Chou, Shemma, McCarrick, Chien, Teoh, Winkel, ... | 2023 | sin journal_ref | PDF leído parcial |
| 2406.14621 | de Graaf, Xue, Chapman, Teoh, ..., Schoelkopf | 2024 | sin journal_ref | PDF leído parcial |
| 2311.04423 | Koottandavida, Tsioutsios, Kargioti, ..., Devoret | 2023 | sin journal_ref | PDF leído parcial |
| 2503.10935 | Quantum Circuits, Inc. Team | 2025 | sin journal_ref | PDF leído parcial |
| 2510.08416 | Dakis, Puri, Economou, Barnes | 2025/26 | sin journal_ref | PDF leído parcial |
| 2601.21616 | Mai, Liu, Deng, Cai, Ni, ..., Yu | 2026 | sin journal_ref | PDF leído parcial |
| 2605.10932 | Mansouri, Arai | 2026 | sin journal_ref | RESUMEN |

Notas: "sin journal_ref" significa que la API de arXiv no devuelve revista; no implica que no esté publicado. Los números de revista marcados "según ref." proceden de listas de referencias de otros PDFs, no de la revista. Todos los PDFs están en /tmp/lit2/ (texto extraído con pdftotext).
