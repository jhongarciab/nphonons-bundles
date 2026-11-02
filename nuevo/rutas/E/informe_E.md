# Ruta E — Bundles de dos fonones por acoplo efectivo de dos fonones (g_x, g_z) y por escalera de transmon

Marcas: [TEXTO] leído en abstract/PDF descargado; [RESUMEN] leído parcialmente; [CÁLCULO] resultado propio (código en `codigo/`); [NO VERIFICADO] no comprobado. Sin commit ni push. PDFs en `/tmp/rutaE_pdf/`.

## 0. Resumen ejecutivo (veredicto en 6 líneas)
1. El mecanismo funciona en simulación completa: |e,0> -> |g,2> con G = 2 g_x g_z/ω (verificado por diagonalización exacta, cociente 0.956–0.993) emite pares por κ_m; pureza de ventana P2 ≈ 0.95–0.99 y g²(0) ≈ 27 [CÁLCULO].
2. Condición analítica nueva y exacta: eficiencia de par η = 1/[1 + (κ_q/4G²)(κ_m + κ_q/2 + 2G²/κ_m)], óptimo en κ_m = √2 G, y η ≥ 0.95 exige κ_q ≤ 0.074 G [CÁLCULO]. Con κ_q de Naseem (1.4 G) η = 0.45 (régimen "gato", no "bundle"); hace falta T1 ≳ 30 µs.
3. Bloqueo de la ruta literal: no hay g_z demostrado con un modo mecánico (solo propuestas: Naseem 2508.10500; fluxonio 2508.17105). Con g_x = g_z ≈ 15 MHz (PnC, ω = 2.3 GHz) G/2π ≈ 0.2 MHz: marginal.
4. Variante sin g_z (transmon multinivel, |f,0> -> |g,2> por |e,1> virtual, V = 4g²/|α|): con g/2π = 15.2 MHz y α/2π = −138 MHz (publicados, PnC) da V/2π = 5.7 MHz y P(par) = 0.92 con el T1 medido (1.14 µs) y 0.99 con T1 = 10 µs, con κ_m/2π ≈ 8 MHz [CÁLCULO]. Escala por fonón extra ~ g/α ≈ 0.1–0.2, no λ ≈ 0.002.
5. Novedad: no encontré bundles de fonones por este mecanismo (búsqueda por palabras clave, ~12 consultas; puede fallar). El ingrediente dos-fonones-JC existe para gatos (Sánchez Muñoz 2018, Naseem 2026); para bundles hay solo fotones.
6. GO condicionado, reformulado hacia la variante transmon-f; la ruta g_z literal es NO-GO con lo demostrado hoy (falta g_z y T1 alto).

## 1. Modelo y reducción
H = ω a†a + (ω_q/2)σ_z + (a+a†)(g_x σ_x + g_z σ_z) (+ bombeo Ω(σ+ e^{−iω_p t} + h.c.)), ω = 1.
Efectivo (James–Jerke, de `msc/HANDOFF_nuevo.md`): H_ef = (g_x²/ω)[(4n/3+1)|e><e| − (4n/3+1/3)|g><g|] − G(σ+a² + h.c.), G = 2 g_x g_z/ω.
Régimen "bundle": qubit coherente (κ_q ≪ G), κ_m ~ √2 G (a lo sumo 2√2 G). Régimen "gato" de la maestría: κ_q ≫ G, el modo no decae a una guía.

Verificaciones contra el modelo completo (`espectro.py`, `lab_check.py`, `t2.py`, `eta_check.py`):
| Verificación | Resultado [CÁLCULO] |
|---|---|
| G por anticruce exacto, G_num/(2 g_x g_z/ω), (g_x,g_z)/ω = (0.006,0.06), (0.01,0.1), (0.02,0.1), (0.03,0.15) | 0.993, 0.980, 0.980, 0.956 (reducción Franck–Condon ~e^{−λ²/2…}) |
| Resonancia ω_q* − 2ω | −3.93 g_x²/ω (H_ef predice −4 g_x²/ω) |
| Resonancia de bombeo g0->e0 | δ = −9.4e-5 vs −8 g_x²/3 = −9.6e-5 |
| Marco de laboratorio, bombeo sin RWA, H(t), salto `b` desnudo vs base vestida secular (g_x=0.01, g_z=0.1, κ_m=2.8G, Ω=0.3G, κ_q=0.05G) | <n> = 0.12639 vs 0.12643; p_e = 0.1197 vs 0.1200 |
| P2 y tasa, completo vs efectivo de pocos niveles (4 puntos) | P2: 0.9723/0.9720, 0.9848/0.9847, 0.9849/0.9849, 0.9440/0.9440 |
| η cerrada vs Liouvilliano efectivo vs completo (7 puntos) | idéntica al efectivo; completo +0.2 a +0.4 pp |

Simulación completa: base de autoestados vestidos de H sin bombeo (N = 8–10), bombeo con RWA solo en el bombeo (secular, Δν = 2) en marco rotante diagonal; disipadores vestidos y seculares por bandas (ω, 2ω, 3ω, DC), sin componentes que suben energía (T = 0). Esto es el "disipador consistente" pedido.
- DC (g_z/ω = 0.06, κ_m = √2 G < λ²ω): NO importa. Con DC plano: flujo −0.7 %, P2 0.9505 -> 0.9508, tasa del canal DC 0.005 G. Sí importa para la lectura del flujo: κ_m<n> sobreestima el flujo real de banda en 19 % a bombeo bajo (0.0319 vs 0.0268 G) por la nube polarónica estática (λ²). Usar el flujo de la banda ω.
- Canal de un fonón por g_x: es un fonón a 2ω (Purcell vía el modo), Γ_P = κ_m (g_x/ω)²; a κ_m = √2 G, Γ_P/G = √2 ε² = 5e-5 (ε = g_x/ω = 0.006), la tasa b2 medida en el completo (6e-5 G a p_e = 0.3) coincide con κ_m ε² p_e. Cota de pureza por ese canal: 1 − √2 ε² (irrelevante). Con una guía pasabanda en ω se anula.

## 2. Derivación (condiciones analíticas)
Subsistema {|e,0>, |g,2>} con V = √2 G, amplitud de |g,2> con decaimiento κ_m y de |e,0> con κ_q/2. Ecuación de Lyapunov para X = ∫c c† dt (resuelta a mano, `deriv_eta.py` es el esquema simbólico; sympy no está instalado):
- η (prob. de que |e,0> acabe en un par, no en pérdida por κ_q) = bV²/(aD+V²b), a = κ_q/2, b = κ_m, D = b(a+b)+V². Equivalente: 1/η − 1 = (κ_q/4G²)(κ_m + κ_q/2 + 2G²/κ_m).
- Máximo en κ_m* = √2 G: 1/η − 1 = κ_q/(√2 G) + κ_q²/(8G²). η ≥ 0.95 <=> κ_q ≤ 0.073 G (curva blanca, panel superior derecho). Para κ_m ≫ G recupera κ_q/Γ_2, Γ_2 = 4G²/κ_m.
- Tasa lenta de |e,0>: Γ_e = κ_m − √(κ_m² − 8G²) (sobreamortiguado); máxima, = 2√2 G, en el amortiguamiento crítico κ_m = 2√2 G.
- Bombeo débil: tasa de absorción W = 4Ω²/(κ_q + 4G²/κ_m); tasa de pares R = ηW.
- Fusión de pares (ventana τ_c = x/κ_m): P2 ≈ (1 − 2e^{−x})(1 − q), q ≈ (R/η)[τ_c − (1 − e^{−Γ_e τ_c})/Γ_e]. Contrastado con el mapa exacto: acierta a ~10 % para κ_m ≲ G y sobreestima 2–5x para κ_m ≳ 3G (cota conservadora). La curva Ω_max(κ_m) para P2 = 0.95 es la línea discontinua de los mapas. [CÁLCULO, semi-analítico]
- Estadística de clusters EXACTA (sin Monte Carlo): s_m = Tr[Φ^m ρ_c], Φ = J∫_0^τ e^{L0 t}dt; P2 = (s1 − 2 s2 + s3)/(1 − s1), <k> = 1/(1 − s1) (`purity.py`).
Definición de pureza: P2(τ_c) = fracción de clusters de clics (banda ω, hueco < τ_c) con exactamente 2 clics. κ_q NO entra en P2 (la pérdida es silenciosa): se reporta aparte como eficiencia η. Un flujo útil exige las dos.

## 3. Resultados (mapas)
`mapa_E.png`: fila superior κ_q/G = 0.05, inferior 1.4 (Naseem). Columnas: tasa de clics/G (Ω/G vs κ_m/G), P2, y (arriba derecha) η(κ_m, κ_q) con curvas η = 0.95, 0.90 y los 7 puntos verificados; abajo derecha: transmon-f. Líneas: roja κ_m = √2 G; naranja κ_m = 2√2 G; discontinua: Ω_max analítico; sólida: contorno numérico P2 = 0.95. Datos: `codigo/mapa_kq*_x*.npz` (22×22, κ_q/G ∈ {0.05,0.2,1.4}, x ∈ {3,5,8}).
Parámetros del mapa: g_x/ω = 0.006, g_z/ω = 0.06 (G/ω = 7.2e-4), N = 6 (efectivo validado).
Hallazgos: flujo de clics satura en ≈1.6 G; para P2 ≥ 0.95 (CW) solo Ω ≲ 0.1–0.2 G, flujo ≈ 0.03–0.07 G; con bombeo pulsado (sin fusión) ≈ 10x más [ESTIMACIÓN: 2η/(1/Γ_e + 5/κ_m)].

## 4. Parámetros reales (tabla, `tabla_out.txt`)
κ_m = √2 G, x = 5–12, Ω óptimo con P2 ≥ 0.95. ω/2π = 100 MHz, g_x = 0.6, g_z = 6 MHz (Naseem 2508.10500 [TEXTO]); segunda fila de casos hipotética (g_x = 2, g_z = 10 MHz).
| caso | κ_q/2π [kHz] | G/2π [kHz] | κ_m/2π [kHz] | κ_q/G | η | Ω/G | clics/s CW | P2 | pulsado clics/s (est.) |
|---|---|---|---|---|---|---|---|---|---|
| Naseem | 100 | 71.5 | 101.8 | 1.39 | 0.45 | 0.133 | 1.0e4 | 0.953 | 9.6e4 |
| Naseem | 30 | 71.5 | 101.8 | 0.42 | 0.76 | 0.096 | 8.8e3 | 0.959 | 1.6e5 |
| Naseem | 10 | 71.5 | 101.8 | 0.14 | 0.91 | 0.096 | 1.05e4 | 0.955 | 1.9e5 |
| Naseem | 3 | 71.5 | 101.8 | 0.042 | 0.97 | 0.096 | 1.12e4 | 0.953 | 2.1e5 |
| hipotético | 10 | 397 | 566 | 0.025 | 0.98 | 0.096 | 6.3e4 | 0.953 | 1.2e6 |
Ma 2019 (microondas, g/2π = 0.3 GHz, θ = π/4, ω = 6 GHz, κ_q/2π = 30 MHz; del informe de msc, no releído por mí [NO VERIFICADO]): G = 2 g_x g_z/ω ≈ 15 MHz, κ_q/G ≈ 2: tampoco régimen bundle.

Plataformas demostradas (de PDFs descargados):
| Plataforma | Dato [TEXTO] | Consecuencia |
|---|---|---|
| HBAR + transmon, Chu 2018 (1804.07426) | g0/2π = 350 kHz, modo ≈ 6.29 GHz, T1 fonón 64 µs (hasta 113) | f-escalera: V = 4g²/α ≈ 2 kHz (α = 250 MHz es [SUPUESTO]): inútil |
| HBAR colectivo (2603.20800) | g/2π ≈ 0.68–0.89 MHz a ≈ 4.77 GHz | idem, V ~ 10 kHz |
| PnC + transmon, Arrangoiz-Arriola 2019 (1902.04681) | g/2π = 13–16 MHz (15.2), banda 2–2.4 GHz, T1 qubit 1.14 µs, γ/2π ≈ 0.6 MHz, α/2π = 138 MHz | f-escalera: V/2π = 5.7 MHz |
| fluxonio–membrana 4 MHz (2505.21481) | acoplo resonante demostrado, >300 intercambios en 6 ms; g no extraído | candidato a g_z (con 2508.17105, teoría de acoplo longitudinal+transversal sintonizable) [NO VERIFICADO] |
Qué falta: (i) g_z mecánico demostrado (ninguno hallado); (ii) qubit sintonizable a ω_q = 2ω_b con T1 ≳ 30 µs a la vez que g_x, g_z del orden de decenas de MHz; (iii) canal de salida con κ_m/2π ≈ 0.1–8 MHz (guía fonónica acoplada al modo): no hallado en lo leído [NO VERIFICADO]; (iv) bombeo/reset pulsado.
Ejemplo PnC con g_x = g_z = 15.2 MHz, ω = 2.3 GHz: G/2π = 2 g_x g_z/ω ≈ 0.20 MHz; κ_m* ≈ 0.28 MHz (Q_ext ≈ 8000); η ≥ 0.95 requiere κ_q/2π ≤ 15 kHz (T1 ≥ 10 µs) [CÁLCULO].

### Variante transmon multinivel (sin g_z) — `transmon_f.py`, `run_transmon.py`
H = Σ_k E_k|k><k| + ω a†a + g(σ†a + h.c.), σ = Σ√(k+1)|k><k+1|, T1 en escalera (k κ_q), RWA en g (g/ω < 1 %). V = <g,2|H_ef|f,0> = 4g²/α, resonancia ω_q = ω − α/2. Probabilidades exactas P(M = 2 fonones, L = 0 saltos del qubit) desde |f,0> por cadena de bloques (sin Monte Carlo):
| caso (g, α /2π [MHz], T1) | V/2π [MHz] | κ_m* /2π [MHz] | P(par) máx | P(1 fonón) |
|---|---|---|---|---|
| 15.2, −138, 1.14 µs (publicado) | 5.74 (4g²/α = 6.70) | 8.3 | 0.918 | 0.039 |
| 15.2, −138, 10 µs | 5.74 | 8.3 | 0.989 | 0.006 |
| 15.2, −300, 10 µs | 2.96 | 3.8 | 0.977 | 0.012 |
| 15.2, −300, 50 µs | 2.96 | 3.8 | 0.995 | 0.003 |
| 5, −250, 50 µs | 0.40 | 0.36 | 0.979 | 0.005 |
| 0.35, −250 (supuesto), 50 µs | 0.0020 | 0.02 | 0.049 | 0.000 |
Fuga a un fonón: la componente virtual |e,1> (β = 2√2 g/|α| = 0.31 en el caso PnC) emite un fonón con κ_m; crece con κ_m y β (el mínimo de la curva P(par) vs κ_m, panel abajo derecha). Flujo: ciclo ≈ 2 pulsos π (~0.1 µs) + ≈ 1/Γ_e + 5/κ_m ≈ 0.5 µs: del orden de 1–2 MHz de pares [ESTIMACIÓN]. Escala con el orden (Duffing exacto, `escala_N.py`, V_N = |N,0>↔|g,N>): g/2π = 15.2, α = −138: V_2 = 5.74, V_3 = 1.04, V_4 = 0.105 MHz (razón ≈ 0.18 y 0.10 por fonón extra), frente a λ = 0.002 de Bin. Requiere preparar |N> con N−1 pulsos y pierde por T1 de niveles altos.

## 5. Novedad (lectura real)
| Ref. | Qué hace | Qué NO hace |
|---|---|---|
| Bin et al. 1907.12714 (PRL 124 053601) [TEXTO] | bundles de N fonones por proceso Stokes, pureza casi perfecta | fuerza ∝ λ^N/√N! (el problema de partida) |
| Bin, Wu, Lü 2012.10926 [TEXTO] | bundles de 2n fotones protegidos por paridad, régimen ultrafuerte (Rabi + bombeo del qubit) | fotones; necesita USC; sin acoplo g_z/g_x efectivo ni fonones |
| Tang & Deng 2409.01737 [TEXTO intro/modelo] | JC de dos fotones + Kerr, bundles de 2–3 fotones; átomo de 4 niveles con bombeo Raman | óptica; acoplo de dos fotones impuesto por láseres; sin óptimo analítico de pureza ni plataforma mecánica |
| Sánchez Muñoz, Lara, Puebla, Nori 1802.01306 (PRL 121 123604) [TEXTO abstract] | JC de dos fonones con qubit de espín y resonador mecánico, gato "saltarín" | régimen de gato/disipativo; no estudia bundles emitidos |
| Naseem 2508.10500 (PRA 113 013732) [TEXTO] | g_x+g_z, qubit conducido a 2ω, pérdida de dos fonones y gato | qubit con pérdida; sin emisión de pares a una guía |
| Ma, Xie, Li PRA 99 022302 [NO VERIFICADO, solo de la bibliografía de msc] | gato de microondas con qubit sin simetría de inversión | idem |
| 2610.09971 [TEXTO abstract] | pares fotón–fonón y bundles en sistema Mollow tripartito | un fotón + un fonón, no pares de fonones |
| 2406.12631 [TEXTO abstract] | bundles no recíprocos de pares fotón–fonón | idem |
| 2412.18133, 2210.03619 [TEXTO abstract] | bundles en JC de alto espín / Rabi con STIRAP | fotones, otros mecanismos |
No localizado: "Two-photon decay enhanced even photon bundle emission" (PRResearch 2025) [NO VERIFICADO]. Tampoco hallé (en ~12 consultas arXiv) bundles de fonones ni de fotones desde |f,0> -> |g,2> del transmon: el proceso de dos excitaciones en la escalera del transmon es de manual en cQED y probablemente conocido como transición, no como fuente de bundles [NO VERIFICADO]. Veredicto de novedad: moderada y frágil. Defendible: (a) régimen bundle del mecanismo g_x,g_z (complemento de Naseem) con η cerrada y óptimo κ_m = √2 G; (b) bundles de fonones sin λ^N desde la escalera del transmon con parámetros publicados.

## 6. Veredicto por criterios
| Criterio | g_z literal | transmon-f |
|---|---|---|
| (a) P2 ≳ 0.95 y flujo útil con parámetros ≤ 1 orden de lo demostrado | P2(ventana) ≥ 0.95 sí, pero η ≥ 0.95 exige κ_q ≤ 0.074 G ≈ 5 kHz (Naseem) y g_z no demostrado: NO | P(par) 0.92 con T1 medido, 0.99 con T1 = 10 µs; V = 5.7 MHz: SÍ (falta κ_m ≈ 8 MHz) |
| (b) no limitado por λ^n | sí (G ∝ g_x g_z/ω; solo N = 2) | sí (g/α ~ 0.1–0.2 por fonón) |
| (c) novedad | media (fonones + bundle + g_x,g_z) | media-baja (proceso conocido) |
| (d) condición analítica | sí: η, κ_m* = √2G, κ_q ≤ 0.074G, curva de fusión | parcial: V = 4g²/α, escala; el óptimo es numérico |
| (e) ventaja vs Bin | no depende de λ ni de Q alto del emisor; coste: g_z y T1 | plataforma real (PnC/SAW, g y α medidos); N = 2–3; coste: κ_m ingeniado, preparación de |f> |
Bloqueos numéricos: g_z mecánico = 0 demostrado; Naseem κ_q/G = 1.39 -> η = 0.45; PnC con g_x = g_z = 15 MHz: G/2π = 0.2 MHz vs κ_q/2π = 0.14 MHz (T1 = 1.14 µs): κ_q/G = 0.7, η ≈ 0.64.
Riesgos: dephasing del qubit no incluido en la tabla transmon (ensancha el cruce f0–g2; relevante si γφ ≳ κ_m); RWA en g del transmon (g/ω = 0.7 %); modelo de T1 en escalera supuesto (k κ_q); cutoffs N = 6–10; guía y detección ideales (eficiencia de recolección 1); filtro de ancho de banda del guía no modelado salvo J(2ω).

## 7. Código (`codigo/`)
`modelo.py` (completo vestido), `purity.py` (clusters exactos, efectivo), `espectro.py`, `lab_check.py`, `eta_check.py`, `deriv_eta.py`, `t1.py`, `t2.py`, `run_map.py`, `tabla.py`, `transmon_f.py`, `run_transmon.py`, `escala_N.py`, `figs.py`. Figura: `mapa_E.png`. Todo con OMP_NUM_THREADS=1.
