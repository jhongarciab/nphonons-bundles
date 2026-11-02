# Ruta D: qubit con acoplo longitudinal ingeniado en lugar del excitón (bundles de N=2 fonones)

Fecha 2026-11-02. Código y salidas en esta carpeta (`sistemas_D.py`, `verif_exact.py`, `multinivel.py`, `ingenieria.py`, `termico.py`, `*.txt`). Etiquetas: [TEXTO] leído en PDF descargado; [CÁLCULO] mío; [NO VERIFICADO]. Convención: λ = coeficiente de σ†σ(b+b†) (si la fuente da g_z σ_z(b+b†), λ=2g_z). Maestra vestida, n=2, ω_b=1.

## 0. Resumen ejecutivo (6 líneas)
1. **Veredicto: GO CONDICIONAL como propuesta teórica / sección de plataformas; NO-GO como experimento a corto plazo.** No lo recomiendo como artículo independiente.
2. Lo demostrado: λ/ω_b = 0.0027 (Potts, transmon+SQUID, modo EM, 4.35 GHz, γ/ω_b=5.5e-5) [TEXTO]. Con Ω≤0.3ω_b faltaría ×4.5 en λ para J≥1e-5ω_b; con el tope de transmon Ω≤α/3=0.03ω_b faltaría ×14 (≈1.1 órdenes) [CÁLCULO].
3. Para modo MECÁNICO no hay acoplo longitudinal qubit–mecánico medido (solo propuestas: λ/ω_b=0.06 y 0.12). Con acoplo 2508.17105 y coherencia de fluxonio medida (2505.21481) a ω_b=6 MHz, κ=3e-3ω_b, Ω=0.3ω_b: J=3.4e-4ω_b≈2 kHz, P2≈0.92 (exacto) [CÁLCULO].
4. Bloqueos reales: (i) dephasing por ruido de flujo, que cumple γφ/λ=σ_Φ/Φ_zpf (con σ_Φ efectivo de Potts, 43 µΦ0, γφ/ω_b=0.05 y P≥0.9 es imposible; hace falta σ_Φ≲5 µΦ0); (ii) ocupación térmica del modo (n_th=34 a 6 MHz y 10 mK; exige n_th≲0.02).
5. Novedad aparente pero modesta: no encontré bundles de N fonones con qubit y acoplo longitudinal ingeniado; sí hay trabajos vecinos (Cosacchi 2108.03967, Naseem 2508.10500).
6. Lo analítico nuevo es pequeño (relación λ/γφ=σ_Φ/Φ_zpf, criterio n_th≪J/κ, regla de resonancia 1→2 por |α|=(m−n)ω_b); sirve de sección, no de resultado principal.

## 1. Literatura leída (tabla de sistemas)
λ/ω_b con λ en convención de este proyecto. γ=1/T1, γφ=2/T2−γ (coherencia decae a (γ+γφ)/2), todo /2π.

| Sistema | arXiv / año | ω_b/2π | g (fuente) → λ | λ/ω_b | γ/2π (T1), γφ/2π | κ/2π o Q del modo | Banda lateral resuelta | Estado |
|---|---|---|---|---|---|---|---|---|
| A. Transmon + SQUID, acoplo intrínseco a resonador λ/4 (Hint=g0 a†a(b+b†)) | 2407.02024 Potts (PRL 134, 153603, 2025) | 4.347 GHz | g0=11.9 MHz (medido, estático) → λ=11.9 MHz | 0.0027 | T1=664 ns→0.240 MHz; T2*=235 ns→γφ=1.11 MHz; γ/ω_b=5.5e-5, γφ/ω_b=2.6e-4 | κint 28 kHz + κext 88.6 kHz = 0.117 MHz (κ/ω_b=2.7e-5) | sí | **medido**, modo EM; α=−388 MHz |
| B. Transmon inductivamente shuntado, diseño | 1708.04917 Richer 2017 (PRB 96, 174520) | ω_r 6–8 GHz | g_zx/2π=53 MHz (tabla de diseño; σ_z) → λ=106 MHz | 0.013–0.018 | no da; supongo coherencia de A (NO VERIFICADO) | no da | — | **diseño**, no medido |
| C. Qubit de carga + nanomecánico (CPB), cat de dos fonones | 2508.10500 Naseem (PRA 113, 013732, 2026) | 100 MHz | g_z/2π=6 MHz (σ_z; "paráms. accesibles", refs. de la fuente) → λ=12 MHz | 0.12 | decay qubit 100 kHz (γ/ω_b=1e-3); no incluye γφ | γ_m/2π=15 Hz (κ/ω_b=1.5e-7) | sí | **propuesta** teórica |
| D. Fluxonio + membrana/viga suspendida, flujo B∥ | 2508.17105 Nongthombam | 6 MHz | g_z/2π≈180 kHz (B∥=60 mT, m=0.75 pg, l=40 µm) → λ=0.36 MHz | 0.06 | su simulación usa Γ/2π=2 MHz, Γφ=0.9 MHz (γ/ω_b=0.33: NO resuelta) | γ_m/2π=5 Hz | no con sus Γ; sí con coherencia real (D2) | **propuesta** teórica |
| Coherencia fluxonio–mecánica (acoplo TRANSVERSAL) | 2505.21481 Gerashchenko | 4.4 MHz (membrana SiN) | vacuum Rabi g=1.50 kHz (transversal, resonante) | 3.4e-4 (transv.) | T1=22.4 µs (7.1 kHz), T2=17.1 µs → γφ=11.5 kHz | Q=1.62e5 (27 Hz); n_th≈47 a 10 mK | sí | **medido**; solo usado aquí por su coherencia |
| Espín NV + punta magnética (Hs=ħλ(a+a†)Sz) | 0806.3606 Rabl 2009 (PRB 79, 041302) | 5–7 MHz | λ/2π≈100–115 kHz (propuesta, G_m≈1e7 T/m) | ≈0.02 | δn~1/T2*~1 MHz (>λ; sin vestir); γ_r/2π=20 kHz (calentamiento a 100 mK, Q=1e5) | Q=1e5 | solo con estados vestidos | **propuesta** (2008, fuera de rango; espines 2015–26 no leídos: NO VERIFICADO) |
| Ion, fuerza dependiente de espín | 1412.7100 Lo (Nature 2015) | ω_z/2π=2.1 MHz | pulsos de banda lateral; λ no extraído | — | — | — | — | solo [TEXTO] parcial; no comparable |
| QD semiconductor-GHz (baseline grupo) | lit_bundles.md (Spinnler, Kettler) | ~GHz | — | ≈0.002 | γ/ω_b≈0.09 | — | no | medido |
| Qubit QD híbrido, long. paramétrico | 2407.08869 Harpt (npj QI 11, 5) | 1.304 GHz | g̃∥/2π≈25 kHz (derivado) | ≈2e-5 | — | κ/ω_b=9.6e-5 | sí | medido, débil |
| Qubit de carga CMOS, long. paramétrico | 2410.20217 Champain | 526 MHz | no extraído | — | — | κ/ω_b=0.026 | no | medido, débil |
| Lectura paramétrica long. (referencias de método) | 1504.04002 Didier; 1603.04424 Royer; 1511.06138 Richer-DiVincenzo | — | solo teoría, sin cifras GHz extraídas | — | — | — | — | contexto |

Respuesta (iii): con acoplo longitudinal, encontré enfriamiento por banda lateral (2508.17105), cat de dos fonones (2508.10500) y lectura, pero **ninguna propuesta de láser de fonones/bundles con qubit** (ver §5). No pude recuperar Billangeon 2015 ni Kounalakis 2018 (API arXiv con 429): [NO VERIFICADO].

## 2. Restricciones de bombeo
- Bundle exige Ω/ω_b≈0.05–0.3 (usado: 0.3 como tope de validez de la fórmula cerrada).
- Transmon: Ω≲α/3 si el bombeo fuese resonante; aquí el bombeo es de banda lateral azul (ω_d=ω_q+nω_b), a distancia nω_b+|α| de la transición 1→2, y el cálculo de 3–4 niveles (§3.3) muestra correcciones ≤8 %. El límite práctico queda en Ω≲α/3 por prudencia: A: Ω≤0.030ω_b (129 MHz de 388 MHz); B (supuesto α=250 MHz, NO VERIFICADO): 0.012ω_b.
- Resonancia espuria: si |α|=(m−n)ω_b (m entero) la banda 1→2 emite m fonones; evitar |α|/ω_b entero. Con ω_b=100 MHz y α=−250 MHz (2.5) no coincide.
- Fluxonio/otros con anarmonicidad ≫ ω_b: sin límite; a ω_b=6 MHz, Ω=0.3ω_b=1.8 MHz es trivial (≪ α, γ real 7 kHz). A 100 MHz, Ω=30 MHz, también alcanzable. En A (ω_b=4.35 GHz) Ω=0.3ω_b=1.3 GHz > α: **no alcanzable** con transmon.

## 3. Brecha con el mapa de requisitos (cálculo)
Cerrada: `flux_closed`, `P_closed` (maestra vestida), resonancia `delta_res`; optimización sobre κ y Ω≤Ωcap con P2≥0.9. Verificación con `exact_point` (Liouvilliano exacto, N=9): cerrada y exacto coinciden en P (≤0.015) y J (≤12 %) en todos los puntos usados (`verif_exact.txt`). P2 es por episodios, **no observable directamente**. Salida completa en `resultados_sistemas.txt`.

### 3.1 Parámetros reales y "κ, Ω óptimos" (n=2)
| Sistema | λ/ω_b | κ/ω_b real | γ/ω_b | γφ/ω_b | Ω (real) | J real (ω_b ; Hz) | P2 | κ* óptimo, Ω | J óptimo (ω_b ; Hz) | Factor en λ para J≥1e-5ω_b |
|---|---|---|---|---|---|---|---|---|---|---|
| A (Ω≤0.03) | 2.7e-3 | 2.7e-5 | 5.5e-5 | 2.6e-4 | 0.03 | 5.4e-10 ; 2.4 Hz (exacto 5.55e-10, P=0.991) | 0.99 | 1e-6 (borde de rejilla), 0.03 | 6.3e-10 ; 2.8 Hz | **×14** (λ/ω_b≈0.039) |
| A (Ω≤0.3, irreal para transmon) | 2.7e-3 | 2.7e-5 | 5.5e-5 | 2.6e-4 | 0.3 | 5.5e-8 ; 241 Hz | 0.99 | 2e-6, 0.3 | 6.4e-8 ; 278 Hz | ×4.5 |
| B (Ω≤0.012) | 1.5e-2 | 1.7e-5 | 3.4e-5 | 1.6e-4 | 0.012 | 1.3e-7 ; 0.9 kHz (exacto 1.6e-7, P=0.91 a κ=4e-6) | 0.99 | 4e-6, 0.012 | 1.5e-7 ; 1.0 kHz | ×3.5 (λ/ω_b≈0.053) |
| C (γφ=0 / =2γ) | 0.12 | 1.5e-7 | 1.0e-3 | 0 / 2e-3 | 0.05 | 1.0e-3 (P=0, bundles fusionados) | 0.00 | 6.3e-3, 0.3 | 1.5e-3 ; 147 kHz (exacto 1.32e-3, P=0.891) | 1 (ya cumple) |
| D (coherencia de su simulación) | 0.06 | 8.3e-7 | 0.33 | 0.30 | 0.05 | 2e-7 | 0.01 | sin solución | 0 | no hay (γ lo impide) |
| D2 (acoplo D + coherencia medida 2505.21481) | 0.06 | 8.3e-7 | 1.2e-3 | 1.9e-3 | 0.001 | 2e-8 | 0.90 | 2.5e-3, 0.3 | 4.6e-4 ; 2.8 kHz (exacto κ=3e-3: J=3.35e-4, P=0.919) | 1 |
| D3 (D2 con γφ=S·σ_Φ, σ_Φ=42.8 µΦ0 de Potts, S=7.2 GHz/Φ0) | 0.06 | 8.3e-7 | 1.2e-3 | 5.1e-2 | — | — | <0.1 | sin solución | 0 | no hay |
| D4 (D3 con σ_Φ=5 µΦ0) | 0.06 | 8.3e-7 | 1.2e-3 | 6.0e-3 | — | — | — | 2.5e-3, 0.3 | 3.3e-4 ; 2.0 kHz | 1 |
| E (QD GHz) | 2e-3 | 1e-3 | 0.09 | 0.18 | 0.05 | 5.9e-13 | 0 | sin solución | 0 | no hay (γ/ω_b=0.09 >6.6e-3: ningún λ lo salva) |

Ojo con C y D2: κ* es 3e-3–6e-3 ω_b, 4 órdenes sobre el κ intrínseco (Q≈100–300): hay que ingeniar disipación **fría** (n_th efectivo ~0). Ω=0.3 está en el borde de validez; con Ω=0.15, C da J=8e-4, P=0.938 (exacto).

### 3.2 Pureza y flujo útiles
En C y D2/D4: J≈3e-4–1.5e-3 ω_b, esto es ≈2 kHz (6 MHz) a ≈1e5 Hz (100 MHz), con P2≈0.9 por episodios. En A realista J≈2–3 Hz (inservible). Para A el cuello es λ^4: J∝(Ω_n)²∝λ^(2n).

### 3.3 Multinivel (`multinivel.py`)
Transmon de L=2,3,4 niveles, acoplo kλ, maestra de laboratorio local, misma optimización de Δ (el salto local solo es fiable si κ≳λ²; el cociente sí es informativo):
- A (α/ω_b=−0.089, Ω=0.03, κ=1e-4): J3/J2=1.000.
- C con transmon a 100 MHz, α=−2.5ω_b, Ω=0.15: J3/J2=0.972; con α=−2.0ω_b: 0.917 (el nivel 2 fuga algo); α=−10ω_b: 1.002.
Conclusión: corrección multinivel ≤8 % con bombeo de banda azul y α<0. [CÁLCULO; el caso α>0 no se ha explorado.]

## 4. Ingeniería para λ/ω_b≳0.01 (`ingenieria.py`)
Acoplo flujo–mecánico: λ=S·B·l·x_zpf (S=∂f_q/∂Φ en Hz/Φ0), x_zpf=√(ħ/2mω). Referencia medida: Potts S=26 GHz/Φ0, Φ_zpf=461 µΦ0.
| ω_b | m, B, l | x_zpf | Φ_zpf | S necesario (λ/ω_b=0.01 ; 0.03) | γφ implicado con σ_Φ=43 µΦ0 |
|---|---|---|---|---|---|
| 6 MHz | 0.75 pg, 0.06 T, 40 µm | 43 fm | 50 µΦ0 | 1.2 ; 3.6 GHz/Φ0 (=0.05 de Potts) | 0.05 MHz (0.009ω_b) |
| 100 MHz | 1 pg, 0.1 T, 40 µm | 9.2 fm | 18 µΦ0 | 56 ; 169 GHz/Φ0 (×2.2 y ×6.5 Potts) | 2.4 MHz (0.024ω_b) |
| 100 MHz | 1 pg, 0.5 T | — | 89 µΦ0 | 11 ; 34 GHz/Φ0 (×0.4 y ×1.3) | 0.5 MHz (0.005ω_b) |
| 300 MHz | 1 pg, 0.1 T | 5.3 fm | 10 µΦ0 | 293 GHz/Φ0 (×11) | 12.6 MHz (0.042ω_b) |
Escalado: λ/ω_b∝ω_b^(−3/2) (x_zpf∝ω^(−1/2)); de 100 a 6 MHz se gana ×68 pero el modo tiene n_th mayor.
Relación independiente de plataforma [CÁLCULO]: γφ/λ=σ_Φ/Φ_zpf (Potts: 1.11/11.9 → σ_Φ efectivo 43 µΦ0, Φ_zpf=461 µΦ0). A ω_b bajo Φ_zpf es menor y el dephasing relativo empeora; la banda lateral resuelta exige γφ/ω_b≲0.02 (mapa).
Órdenes de magnitud frente a lo demostrado: λ/ω_b: demostrado 0.0027 (EM); objetivo 0.011–0.03 → **0.6–1.0 órdenes** en λ/ω_b (siempre que el bombeo Ω=0.3ω_b sea posible). En modo mecánico no hay λ/ω_b longitudinal medido: la brecha es "demostrar el acoplo" (0 → 0.01), no cuantificable en órdenes. Campo requerido: B∥=0.06–0.5 T en el plano compatible con Al/Nb (NO VERIFICADO; la fuente usa 60 mT).
Térmico (`termico.txt`): n_th(6 MHz,10 mK)=34, n_th(100 MHz,10 mK)=1.6, n_th(300 MHz,10 mK)=0.31 (coincide con n_th≈47 de Gerashchenko a 4.4 MHz). Criterio [CÁLCULO]: bundle visible si n_th≪J/κ≈0.1–0.2 (C, D2), i.e. n_th≲0.02 → T≲1.2 mK a 100 MHz, 3.7 mK a 300 MHz, 0.07 mK a 6 MHz. Solo la disipación ingeniada fría o enfriamiento activo de banda lateral lo permiten. Mi modelo no incluye bombeo térmico (T=0).

## 5. Novedad (trabajos más cercanos)
1. 1907.12714 Bin et al.: bundles de N fonones, QD con acoplo longitudinal; sin qubit ingeniado ni requisitos de plataforma (numérico).
2. 2012.10926 Bin-Wu-Lü: modelo Rabi con θ general, maestra vestida; no cierra pureza ni compara con plataformas SC.
3. 2108.03967 Cosacchi et al.: estadística de N-fotones con qubit SC en resonador (acoplo transversal) y QD; no longitudinal ni fonones mecánicos.
4. 2304.11298 Zou-Li-Liao: bundles dinámicos STIRAP en circuit QED; fotones, no longitudinal.
5. 2508.10500 Naseem: qubit con acoplo transversal+longitudinal a nanomecánico, intercambio resonante de dos fonones, cat; objetivo estado cat, no flujo/pureza de bundles; cifras λ/ω_m=0.12.
6. 2610.09971 Zhang-Zhao-Tang-Deng: emisor de dos niveles con fotones y fonones motores (Mollow), pares y bundles de fotones/fonones; mecanismo Mollow, no longitudinal ingeniado ni requisitos.
7. 2107.07667 Wang-Chen-Liao: qubit con acoplo longitudinal y cuadrático, maestra vestida (transporte térmico); sin bombeo ni bundles.
8. 2508.17105 Nongthombam: fluxonio–mecánico con acoplo longitudinal; enfriamiento, no bundles.
Veredicto de novedad: no encontré ninguna propuesta de bundles de N fonones con qubit y acoplo longitudinal ingeniado ni un mapa de requisitos para ellos; la búsqueda fue limitada (API con 429, solo títulos/abstracts): [NO VERIFICADO] exclusividad.

## 6. Veredicto con criterios
- (a) Brecha ≤1 orden en λ/ω_b: **parcial**. EM: ×4.5 (Ω irreal) a ×14 (Ω≤α/3) respecto a lo medido; fluxonio/CPB mecánico: sin demostración longitudinal (propuestas ya en 0.06–0.12).
- (b) Flujo y pureza útiles: **sí en papel** (≈2 kHz a 6 MHz, ~1e5 Hz a 100 MHz, P2≈0.9 por episodios), **condicionado** a σ_Φ≲5 µΦ0, κ≈3e-3ω_b frío, n_th≲0.02, Ω=0.3ω_b.
- (c) Novedad: **aparente, modesta** (Cosacchi y Naseem solapan).
- (d) Algo analítico nuevo: **poco** (γφ/λ=σ_Φ/Φ_zpf; criterio térmico; regla |α|=(m−n)ω_b; regla λ/ω_b∝ω_b^(−3/2)).
Qué parámetro bloquea: en A, λ/ω_b junto con el tope de Ω (J∝λ^4Ω²); en mecánico, **γφ por ruido de flujo** (γφ/λ=σ_Φ/Φ_zpf≈0.09 con σ_Φ=43 µΦ0 a 6 MHz → γφ/ω_b=0.05) y la **ocupación térmica**. Decisión: GO solo como sección "plataforma qubit" del trabajo ya desarrollado (con D4 como objetivo), no como proyecto aparte.

## 7. Riesgos
Todo λ mecánico es propuesta; ruido de flujo de 1/f no es gaussiano-Markov (mi γφ es un ajuste a T2*); Ω=0.3 en el límite de la cerrada (verificada contra exacto en 2 puntos); κ* ingeniado requiere baño frío; el campo B∥ degrada fluxonio (no evaluado); α>0 y fuga a niveles altos con L>4 no evaluados; λ convención ×2 puede cambiar C en un factor 2.
