# Informe WP2: verificación independiente de H1-H4 (bundles de fonones, emisor + modo, acoplo longitudinal)

Unidades ω_b=1. Código propio en este directorio (`lib.py`, `A_grid.py`, `B_purity.py`, `B2_episodes.py`, `C_micro.py`, `C2_extrap.py`; logs `*.log`, `A_grid.json`). Uso: `python A_grid.py` (≈1 min), `python B_purity.py` (≈10 min), `python B2_episodes.py`, `python C_micro.py`, `python C2_extrap.py` (≈5 min); 1 proceso cada uno. Las partes A-C se hicieron sin abrir scoutB/ ni bundles/ (D al final).

## Convenciones y comprobaciones previas
- |g>=0, |x>=1, σ=destroy(2). Maestra en el marco del laboratorio: H=δ_L n+b†b+λ n(b+b†)+Ω(σ+σ†), saltos √κ b, √γ σ, √γφ n.
- Polarón T=exp[-λ n(b†-b)] con **H_p=T†HT**: b→b-λ n, σ→σ D(-λ), D(-λ)=exp[-λ(b†-b)], H_p=δ n+b†b+Ω(σD+σ†D†). (La especificación escribe σ†D+σD†: es solo un cambio de signo de λ en D; |d_{mk}|² no cambia. Hay que fijar la convención de signos al publicar.)
- Comprobación: flujo en el estacionario, lab (local) = polarón con salto b-λn: 3.84529e-4 vs 3.84529e-4 (igual a 1e-11). Vestida: 3.363e-4.
- Pureza: tres definiciones, ver B.

## A. H1 y H2 contra la maestra exacta (rejilla propia, 22 puntos, n=2,3)
**Derivación de ε_n.** Marco de polarón, H0=δ n+b†b, V=Ω(σD+h.c.). Con δ=-n: |g,0> (energía 0) se acopla con amplitud Ω d_{m0} a |x,m> (energía m-n); |x,n> (energía 0) se acopla con Ω d_{nm} a |g,m> (energía m). A 2º orden S_g=Ω²Σ_{m≠n} d_{m0}²/(n-m), S_x=-Ω²Σ_{m≥1} d_{nm}²/m, y la condición E_g+S_g=E_x+S_x da
δ_L = λ² - n + ε_n, ε_n = Ω²[Σ_{m≠n} d_{m0}²/(n-m) + Σ_{m≥1} d_{nm}²/m] ≈ 2Ω²/n (λ→0; términos m=0 y m=n). Para λ=0.14, Ω=0.1, n=2: ε=0.01019 (no 0.0100): 0.5% de corrección por λ. (Estos términos incluyen polarón y Stark dinámico; no hay otro término de 2º orden: el acoplo resonante Ω d_{n0} no desplaza el centro.)

**Resultado** (columna: error del pico del flujo exacto respecto a la fórmula, en unidades 1e-4; anchura de línea ~ (nκ+γ+γφ)/2 ≈ 30-100e-4):

| n | Ω | κ | λ | err pico lab(local) | err pico vestida | err 2Ω²/n solo | J_vestida/J_H2-1 | J_local/J_H2-1 |
|---|---|---|---|---|---|---|---|---|
| 2 | .07 | .0015 | .12 | 0.1 | 0.1 | 0.8 | -2.4% | +2.7% |
| 2 | .07 | .004 | .17 | 0.3 | 0.1 | 1.7 | -1.5% | +26% |
| 2 | .13 | .0015 | .17 | 1.3 | 1.1 | 5.9 | -6.1% | +3.8% |
| 2 | .13 | .004 | .17 | 1.3 | 1.0 | 5.9 | -2.0% | +25% |
| 3 | .07 | .004 | .17 | 0.4 | 0.0 | 0.9 | +0.5% | +18% |
| 3 | .13 | .004 | .17 | 0.6 | 0.2 | 2.3 | -0.9% | +17% |
| 2 | .2 | .01 | .2 | 6.5 | 5.4 | 21 | -0.1% | +94% |
| 3 | .2 | .01 | .2 | 3.6 | 1.3 | 9.1 | -1.9% | +61% |
| 2 | .1 | .003 | .14 | 0.5 | 0.3 | 2.4 | -2.1% | +12% |

(Tabla completa en `A_grid.log`.) **H1: confirmada.** Error ≤1.3e-4 para Ω≤0.13 (≲0.02 anchuras), 6e-4 en Ω=0.2; ~4× mejor que 2Ω²/n; el resto es 4º orden en Ω. Con Ω=0.2 el error crece como Ω⁴.
**H2: confirmada para la maestra vestida**: |error| ≤ 6.1% (típico 1-2%; el peor caso es κ pequeño con Ω grande, donde 1/Γ0 no es ≫ tiempo de Rabi coherente). Con la maestra local (salto b del laboratorio) el flujo **excede** la fórmula en +3% a +94% (crece con κ y λ², ver H4).
**Por qué (n+λ²):** en el marco vestido el flujo es κ<b†b>. Un ciclo (excitación Raman a tasa Γ0, permanencia ~1/γ en |x̃,0> sin emisión, decaimiento) emite n fonones al caer |x̃,n>→|x̃,0> y, tras σD, el oscilador queda en el estado coherente de amplitud λ (Poisson(λ²), media λ²), que se vacía emitiendo λ². Ciclos entre saltos radiativos exactos: media de fonones/ciclo = 2.0217 (n=2, base; n+λ²=2.0196). Γ0=4Ω²d²/(nκ+γ+γφ) es la tasa de bombeo incoherente (2g²/γ_coh con γ_coh=(nκ+γ+γφ)/2); 1/Γ0+1/γ es el tiempo de ciclo (dos etapas). Precisión: en la local, el decaimiento de coherencia lleva κ(n+λ²)/2 en vez de κn/2 (≈1%).

## B. Pureza
Definiciones (todas exactas, sin Monte Carlo; función generadora de conteo con FFT y renovación):
1. **Ciclos**: P_n^cic=P(N=n), N=saltos del modo entre dos saltos radiativos consecutivos (estacionario; estado post-salto en punto fijo de la aplicación de renovación).
2. **Episodios (clustering temporal)**: un episodio = saltos del modo con huecos < τ_c (operacionalización de "retorno al vacío": el vacío no es un observable proyectable sin retroacción, así que no lo defino mediante proyección). Renovación exacta con K_c=J L0⁻¹(e^{L0 τ}-1), K_>=-J L0⁻¹ e^{L0 τ}. P_n^ep=fracción de episodios con n fonones; además la fracción de fonones en episodios de n: n P_n/<N>.
3. (la del scout, ver D).
Resultados (marco vestido, resonancia con ε_n):
- Base n=2: P^cic=[.033,.057,**.805**,.070,.032]; n=3: P3^cic=.68 (fondo de excitaciones fuera de resonancia, Γ0 pequeña). Media 2.0217.
- Máximo de P_n^cic en rejilla κ∈[3e-4,3e-2], Ω∈[.03,.3], γ∈[2e-5,2e-3] (75 puntos): n=2: λ=.14: **0.9616** (κ=.01,Ω=.03,γ=2e-5) vs e^{-λ²}=0.9806; λ=.2: 0.9459 vs 0.9608; n=3: λ=.14: 0.880; λ=.2: 0.919. **Nunca supera e^{-λ²}**, y el límite se aproxima por abajo.
- Episodios por clustering, base (κ=.003): máx P2=0.798 (τ=3/κ). Depende fuertemente de τ_c (P2=0.25, 0.45, 0.70, 0.80, 0.79, 0.67, 0.51 para τ=0.5,1,2,3,5,8,12 /κ) porque aquí 1/κ≈333 no está bien separado de 1/Γ0≈880.
- Episodios en el óptimo (κ=.01,Ω=.03,γ=2e-5): P2^ep=0.974 (τ=8/κ), P1^ep=0.025 (los fonones de ring-down). Fracción de fonones en episodios de n=2: **0.9854** (τ=8/κ), > e^{-λ²}=0.9806 (cota asintótica 2/(2+λ²)=0.9903; cota para P^ep: 1/(2-e^{-λ²})=0.9810).

**H3, juicio.** Cierto para ciclos: tras σ, con el oscilador en |0> en el marco vestido, el estado del laboratorio es coherente D(λ)|0>, y el número total de fonones que emite al vaciarse es Poisson(λ²) (se demuestra por distribución de número del estado coherente + canal de pérdida pura). N=N_x+N_ring con N_ring~Poisson(λ²) ⇒ P(N=n)≤max_j pmf(j)=e^{-λ²} (λ²<1) si N_x es independiente; si el oscilador no está en |0> al caer el emisor (m≥1) la distribución de número es la de D(λ)|m>, |<n|D|m>|²≤e^{-λ²} (comprobado numéricamente para λ=0.1-0.9: max_{n,m}|<n|D|m>|²=e^{-λ²}) y la cota se mantiene: numéricamente P_n^cic≤0.9616<0.9806 siempre. **No es cierta "para cualquier definición"**: si el ring-down se cuenta como episodio aparte (retorno al vacío entre el bundle y el ring-down), pureza por episodios vale hasta 1/(2-e^{-λ²}) (excede e^{-λ²} en 4e-4) y la fracción de fonones en bundles de n llega a 0.985 (>e^{-λ²}). Hay que citar: "P_n^cic ≤ e^{-λ²} (definición por ciclos radiativos)". Para episodios: P_n≲1/(2-e^{-λ²}), fracción de fonones ≲ n/(n+λ²).

## C. Prueba microscópica de H4 (emisor+b+c, sin RWA)
Modelo: H=H_lab+c†c+g(b+b†)(c+c†), saltos √κ_c c (contados), √γ σ (cierra ciclo), √γφ n. ω_c=ω_b=1. **Factor exacto**: regla de oro con χ_c(ω)=2/(1-ω²-iκ_c ω) ⇒ κ=2g² Imχ(1)=4g²/κ_c (idéntico a RWA en la resonancia exacta; el término contrarrotante aporta solo corrección de orden κ_c² a la forma, no a κ) ⇒ g=√(κκ_c)/2. Condiciones: κ_c≫κ, γ, Ω d, desplazamientos (Markov) y κ_c≪ω_b (suprimir eventos virtuales y que J(ω)∝ω no sea Lorentziana plana; el Lorentziano plano en ω_b exige κ_c ≫ anchuras de líneas ~0.01). Compromiso: 0.03≲κ_c≲0.15. J(ω) efectivo ∝ ω/((1-ω²)²+κ_c²ω²) se anula en ω=0 ⇒ la componente DC -λ n **no** disipa a T=0.
Truncaciones: N_b=8,9,11, N_c=3,4: media por ciclo idéntica a 5 cifras (2.02842).
Resultado (n=2, base, δ_L resonante; media de eventos de c por ciclo radiativo y P(N)):

| κ | κ_c | local | vestida | micro | κ_c→0 (ajuste a+bκ_c², 3 puntos) |
|---|---|---|---|---|---|
| .003 | .03/.05/.10 | 2.3145 (P2=.610) | 2.0217 (P2=.805) | 2.0241/2.0284/2.0493 | **2.0215** (P2=.7975) |
| .01 | .05/.10/.15 | 2.9912 (P2=.452) | 2.0137 (P2=.909) | 2.043/2.123/2.259 | **2.0160** (P2=.907) |
| .03 | .10/.15/.20 | 4.905 (P2=.235) | 1.966 (P2=.899) | 2.46/3.07/3.94 | 1.963 (P2=.847) |

La maestra **vestida coincide** con la dinámica completa (en el límite κ_c→0: 2.0215 vs 2.0217; 2.0160 vs 2.0137; P2 0.7975 vs 0.805, 0.907 vs 0.909). La **local falla** por la nube κλ²/γ (0.29, 0.98, 2.9) que no aparece en el modelo microscópico. El exceso residual del micro crece ∝κ_c² (coef. 2.77, 10.8, 49 ≈ κ·T_ciclo/16 + estática) y es un artefacto de contar saltos de c: c está desplazado estáticamente (por <b+b†>=∓2λ en |x̃>) y fluctúa virtualmente (estado fundamental ultra-fuerte), lo que da saltos de c que no transportan energía al baño; desaparece con κ_c→0 pero entonces κ_c≳κ se pierde la validez de Markov. Para κ=.03 (>λ²) la vestida deja de ser rigurosa (P2 .847 vs .899; extrapolación con 3 puntos poco fiable). Conclusión de H4: **correcta, con las precisiones de abajo**.

## D. Comparación con scoutB/ y bundles/
- scoutB: mismo ε_n (fórmula idéntica), mismo J≈(n+λ²)/(1/Γ0+1/γ), mismos ciclos local [.032 .057 .610 .205 .071] y vestida 0.805, flujo local 38.3e-5 vs el mío 38.45e-5 (diferencia solo por el δ_L usado). Errores de H1 del scout (4e-6, 4e-5, 5e-4, 2.3e-3 a Ω=.05,.1,.2,.3) consistentes con los míos (Ω=.2: 5e-4-6e-4). Coinciden completamente.
- Discrepancia de definición: su "episodio" (cierre por salto 1→0 clasificando por número pre-salto, términos cruzados en evolución sin conteo) da P2=0.938 en la base; el mío (clustering) da 0.80 en la base (0.974 en el óptimo). No es contradicción numérica sino de definición; la de ellos no es un observable de detección y depende de la base de Fock en el marco vestido. Hay que declarar la definición y validar con MC de ventanas.
- bundles/: `bundles_core.py` declara los mismos conceptos; no comprobé sus tests numéricos más allá de leer el encabezado (C_log.txt existe).

## E. Veredicto
- **H1 correcta** (error ≤0.02 anchuras, Ω≤0.13; ≈0.2 anchuras en Ω=.2). Citar la fórmula con Σ exacta de d_{mk}, no solo 2Ω²/n (la corrección λ da 0.5-3e-4).
- **H2 correcta** con la vestida (≤6%, típico 1-2%); la derivación de (n+λ²) es el recuento n (cascada en |x̃>) + λ² (ring-down). Con la local no se cumple (+3 a +94%).
- **H3 requiere precisión**: P_n^cic ≤ e^{-λ²} (probado para N=N_x+Poisson(λ²), numérico: máx 0.9616<0.9806). **Falsa como "toda definición"**: por episodios aparte o por fracción de fonones se supera (0.985>0.9806; cota n/(n+λ²)).
- **H4 correcta cualitativamente**: la local genera una nube espuria κλ²/γ (confirmada por el modelo microscópico completo, que no la tiene); la vestida reproduce la dinámica microscópica en el límite de baño plano (κ_c→0) para κ=.003 y .01; matiz: validez cuando κ≪λ²ω_b ya no es necesaria para κ=.01 (κ≈λ²/2 funciona bien), y falla en rigor para κ≥λ². Es un test con un baño específico (modo c Lorentziano, T=0, acoplo en posición); el criterio relevante es que J(ω) se anule en ω→0.
- **Citable**: ε_n (con Σ), J con la vestida, la nube local κλ²/γ y su ausencia en el modelo microscópico, la cota P^cic≤e^{-λ²} con su definición. **No citar**: purezas de episodios sin definir ni validar; "cualquier definición"; P_n de n=3 en base (dominada por fondo).
