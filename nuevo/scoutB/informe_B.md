# Informe scout B: teoría analítica y de diseño de bundles de fonones (H1, H2, H3)

Etiquetas: [TEXTO] leído en PDF descargado; [RESUMEN] solo abstract; [CÁLCULO] mío, verificado con QuTiP; [NO VERIFICADO].
Unidades ω_b = 1. Parámetros base: λ=0.14, Ω=0.1, γ=2e-4, γφ=4e-4, κ=0.003, n=2 salvo indicación.

## 1. Resumen ejecutivo (veredicto)

1. **GO CONDICIONADO** para H1 (como sección analítica de un artículo mayor, no como artículo independiente); **NO-GO** para H2 (no explorado, sin evidencia) y **NO-GO como resultado central** para H3 (la fórmula es una renovación de dos etapas, trivial).
2. Hay una teoría semianalítica verificada: resonancia exacta a 2º orden (polarón + Stark), tasa de excitación Raman Γ_j, flujo J y pureza P_n en forma cerrada. P_n cerrada vs exacta (maestra completa, episodios): error máx 10 % (n=2) y 13 % (n=3), medio 1.7 % y 2.9 %; 92 %/97 % de la malla bajo 10 %. Flujo: 1–8 %. Advertencia: P≈0.9–0.98, así que el error de la *impureza* 1−P es mediano 15 %.
3. Resultado de diseño nuevo [CÁLCULO]: cota dura P_n ≤ e^{−λ²} (piso Franck-Condon, 0.981 con λ=0.14), κ* óptimo que equilibra la cola lorentziana de líneas n±1 (∝κ) con la fusión por decaimiento a mitad de cascada (∝γ/κ), y Ω_max(κ; X) analítico para pureza ≥ X.
4. **Hallazgo crítico para el grupo**: la pureza "por ciclos" validada en la memoria (P₂=0.61 a κ=0.003) reproduce EXACTAMENTE mi maestra local (jump b del laboratorio), pero es en buena parte un artefacto: el término DC (−λσ†σ) del jump emite una "nube" coherente κλ²/γ≈0.29 fonones por ciclo. Con maestra vestida (b sin componente DC, como en Bin) la pureza por ciclos sube a 0.805 y, agrupando por episodios de cavidad, a 0.938. La tendencia con κ se INVIERTE (local: P₂ cae con κ; vestida: sube). Hay que decidir y declarar el modelo de disipación.
5. Riesgo de novedad: Bin ya da Ω_eff^(n) y las resonancias analíticamente y la pureza numéricamente; no encontré pureza analítica en arXiv (búsqueda limitada; el PRResearch 2025 no se localizó: [NO VERIFICADO]). Los resultados son modestos y el núcleo (modelo de tasas) es físicamente simple; un árbitro de PRA lo vería incremental salvo que se combine con otro giro.

## 2. Novedad: referencias verificadas (descargadas y leídas, ver /pdf)

| # | arXiv / revista | Qué es analítico | Qué es numérico / abierto | Etiqueta |
|---|---|---|---|---|
| 1 | 1907.12714, Bin et al., PRL 124, 053601 | Ω_eff^(n)=Ω e^{−λ²/2}(λ)^n/√n! (régimen λ~ω_b), Ω_eff=Ω λ^n/n! (débil), resonancia Δ_n=−√(n²ω_b²−4Ω²) (Mollow); aproximante de super-Rabi | Pureza Π_n por Monte Carlo con ventanas ("calculate numerically the purity"); dice que el efecto fuera de resonancia crece con κ y con n; γφ "básicamente despreciable" | [TEXTO] |
| 2 | 1306.1578, Muñoz et al. | Definición π_N=λ_N/(λ_1+λ_N); λ_N=γ_a n_a/N | π_2, π_3 numéricos; sin cota analítica (grep de "analytic/Floquet/optimal" sin resultados relevantes) | [TEXTO] |
| 3 | 2108.03967, Cosacchi et al. | Menciona el corrimiento de polarón en las resonancias | Estadística de bundles con baño de fonones continuo (detrimental) y qubit superconductor; numérico | [TEXTO parcial] |
| 4 | 2012.10926, Bin et al., PRL 127, 073602 | Frecuencia super-Rabi analítica y maestra en imagen vestida (Suppl.) | Pureza Π_2 vs κ, θ numérica | [TEXTO] |
| 5 | 2603.27160, Zhang-Wang-Chen | Maestra vestida cuántica, modelo de Rabi | Correlaciones de bundles numéricas | [RESUMEN] |
| 6 | 2604.15604, Tang-Deng | Espectro vestido con interferencia, compromiso pureza/brillo | principalmente numérico | [RESUMEN] |
| 7 | 2604.15605, Tang-Deng | Intercambio de espín mediado por cavidad, φ geométrica | numérico | [RESUMEN] |
| 8 | 2609.07454, Chen et al. | Caracterización analítica de la población residual de estados de una excitación fuera de resonancia (2 emisores) | Pureza fotónica | [RESUMEN] |
| 9 | 2610.09971, Zhang-Zhao-Tang-Deng | Interacciones de orden superior en régimen Mollow | Selección por simetría de disipación; pares fotón-fonón | [RESUMEN] |
| 10 | 2011.03886, Photonics Res. 9, 1289 | Régimen resonante átomo-fonón | Bundles de fonones de átomo atrapado | [RESUMEN] |
| 11 | 2112.03679 | Excitón oscuro, Rabi gigantes | Modelo con excitones brillantes/oscuros | [RESUMEN] |
| 12 | 2406.12631 | Bundles no recíprocos | numérico | [RESUMEN] |

No localizado: PRResearch 2025 "two-photon decay enhanced even photon bundle" [NO VERIFICADO] (dos búsquedas arXiv sin resultado).

Veredicto de novedad: **H1 media** (no vi pureza ni flujo cerrados con competencia n±1, cola lorentziana y fusión, ni cota e^{−λ²}; búsqueda acotada); **H2 sin evidencia** (no probado); **H3 baja** (CV² de renovación es estándar).

## 3. Derivaciones clave

**Polarón exacto.** H = δ_L σ†σ + b†b + λσ†σ(b+b†) + Ω(σ+σ†) en el marco del láser, δ_L=ω_σ−ω_L. U=exp[−λσ†σ(b†−b)] da H_p = δ σ†σ + b†b + Ω(σ†D + σD†), δ=δ_L−λ², D=exp[−λ(b†−b)], con elementos exactos d_{mk}=⟨m|D|k⟩ (polinomios de Laguerre: d_{m0}=e^{−λ²/2}(−λ)^m/√m!). Verificado: ciclos en marco de polarón con jump b−λσ†σ coinciden con el marco del laboratorio (P=[0.032 0.057 0.610 0.205 0.071], flujo 38.3e-5 en ambos).

**Resonancia (condición exacta a 2º orden).** |g,0⟩↔|x̃,n⟩ cuando δ=−n+ε_n, i.e. δ_L=λ²−n+ε_n con
ε_n = −Ω² Σ_{m≠n} d_{m0}²/(m−n) + Ω² Σ_{k≥1} d_{nk}²/k  ≈ 2Ω²/n (m=0 y k=n dominan). Incluye corrimiento de polarón λ² y Stark dinámico en un solo resultado; reduce a Bin-III (√(n²−4Ω²)) para λ→0. Verificación contra el máximo del flujo exacto (n=2, κ=0.003): error 4e-6, 4e-5, 5e-4, 2.3e-3 para Ω=0.05, 0.1, 0.2, 0.3, es decir 0.00, 0.01, 0.14, 0.70 anchuras de línea; n=3: 0.00, 0.00, 0.03, 0.13. Válido para Ω ≲ 0.2 (n=2). A Ω=0.3 Bin-III es mejor (err 1.9e-3 vs 2.3e-3, parecidos). Rabi efectivo: Ω_n=Ω d_{n0} (Bin); no validado por separado (solo indirectamente vía flujo).

**Modelo de tasas (régimen sobreamortiguado, nκ ≳ Ω_n).** Estados (QD, m). Excitación Raman estimulada g,m→x,m+n (la escalera Raman conserva la resonancia para todo m) con
Γ_j = 2Ω² d_{n+j,j}² / γ_c(j), γ_c(j)=[(2j+n)κ+γ+γφ]/2.
Emisión j κ por fonón; decaimiento QD x_j→g_{j'} a γ|d_{jj'}|² (Franck-Condon); "desfase inducido por el bombeo" x↔g a γφ(Ω d/Δ)² fuera de resonancia. Este modelo de 18 estados reproduce la maestra completa (ciclos): P=[0.030 0.057 0.808 0.070 0.032] vs exacta [0.033 0.057 0.805 0.070 0.032], T=6032 vs 6011, flujo 3.36e-4 vs 3.36e-4; y con γ=2e-5 corrige P(4) solo si se incluye el desfase inducido (0.046 vs 0.046).

**Flujo cerrado.** J ≈ (n+λ²)/(1/Γ_0 + 1/γ), con Γ_0=4Ω²d_{n0}²/(nκ+γ+γφ). Error 1–8 % (tabla).

**Pureza cerrada (episodios de cavidad: un bundle = fonones entre dos retornos al vacío).**
P_n ≈ e^{−λ²} · Π_{j=1}^{n}[1 − p_j r_j] · (1 − ε_off), con
 - e^{−λ²}: tras el decaimiento radiativo el oscilador queda con Poisson(λ²) fonones: **cota dura** P_n ≤ e^{−λ²} independiente de κ, Ω, γ;
 - p_j=γ/(jκ+γ) (QD decae con j fonones aún en cavidad), r_j=Γ_j/(Γ_j+jκ) (reexcitación resonante g,j→x,j+n antes de emitir): fusión de bundles, crece ∝ γΩ²/κ³;
 - ε_off=R_off/(R_n+R_off), R_m=2Ω²d_{m0}²γ_c/((m−n)²+γ_c²): excitación en las colas lorentzianas de órdenes m≠n, crece ∝κ² (domina n=3 a κ ≥ 0.005).
Consecuencia (frente de Pareto analítico): fusión ∝ Ω²γ/κ³ y flujo ∝ Ω²/κ a Γ_0≪γ; mayor Ω compra flujo hasta saturar en nγ y cuesta pureza. Existe κ* por n: de la cerrada, n=2: 0.0043 (λ=0.14), 0.0028 (λ=0.1); n=3: ≈6e-4 (λ=0.14), la pureza máxima n=3 requiere κ pequeño.

## 4. Tablas analítico vs numérico (maestra completa)

n=2, episodios exactos (N=8), cerrada con ε_off (data/ y res_t8.txt, res_final.txt):

| Punto | P_exacta | P_cerrada* | flujo exacto | flujo cerrado |
|---|---|---|---|---|
| base | 0.938 | 0.949 | 3.36e-4 | 3.44e-4 |
| κ=0.002 | 0.902 | 0.909 | 3.48e-4 | 3.60e-4 |
| κ=0.001 | 0.807 | 0.786 | 3.48e-4 | 3.78e-4 |
| κ=0.006 | 0.967 | 0.973 | 2.99e-4 | 3.03e-4 |
| κ=0.01 | 0.970 | 0.974 | 2.59e-4 | 2.61e-4 |
| Ω=0.03 | 0.970 | 0.976 | 1.36e-4 | 1.37e-4 |
| Ω=0.2 | 0.916 | 0.918 | 3.75e-4 | 3.87e-4 |
| λ=0.2 | 0.899 | 0.900 | 3.80e-4 | 3.91e-4 |
| γ=1e-3 | 0.820 | 0.857 | 1.03e-3 | 1.02e-3 |
*P_cerrada = versión final de rate.py (con ε_off); el flujo cerrado es el de res_t8.txt. Error máx de esta tabla: 4.5 % en P (γ=1e-3).

Malla (Ω,κ) 8x8 (n=2) y 6x6 (n=3), P_exacta vs P_cerrada (res_final.txt): n=2 máx 10.1 %, medio 1.7 %, 92 % <5 %, 98 % <10 %; n=3 máx 12.8 %, medio 2.9 %, 78 % <5 %, 97 % <10 %. Peor caso: κ≲6e-4 y Ω ≳0.2 (régimen de Rabi coherente, fuera del supuesto de tasas).

Frente de Pareto (cerrada vs exacta, mismos (Ω,κ)): n=2: (J=3.2e-5, P=0.979/0.976), (1.07e-4, 0.978/0.974), (1.98e-4, 0.974/0.967), (3.6e-4, 0.967/0.958). n=3: (2.2e-5, 0.968/0.951), (9.5e-5, 0.958/0.934); fuera de validez a Ω=0.5.

Diseño (cerrada, γ=2e-4, γφ=4e-4): n=2, λ=0.14: P ≥ 0.95 exige Ω ≤ 0.050 (κ=0.002), 0.096 (0.003), 0.237 (0.005) con J=2.7e-4, 3.4e-4, 3.85e-4. n=3, λ=0.14: P≥0.95 solo a κ≈0.002 con Ω≤0.32 (J=1.65e-4); a κ≥0.003 no se alcanza 0.95. Nota: valores Ω≥0.3 están en el borde de validez de la cerrada.

**Sensibilidad al modelo de disipación** (n=2, Ω=0.1; P₂ por ciclos entre saltos radiativos, y episodios):

| κ | ciclos, vestida | ciclos, local (lab) | episodios, σD† (físico) | episodios, σ sin D† |
|---|---|---|---|---|
| 0.002 | 0.738 | 0.605 | 0.902 | 0.921 |
| 0.003 | 0.805 | 0.608 | 0.938 | 0.958 |
| 0.01 | 0.909 | 0.448 | 0.970 | 0.990 |

La nube de la maestra local aporta κλ²/γ ≈ 0.29 fonones/ciclo (Poisson), explicando P(3)=0.205. [CÁLCULO] La corrección del jump vestido es el estándar cuando κ ≪ λ² (aquí λ²=0.0196 > κ); no verifiqué qué hace exactamente Bin en su Suplemento (no accesible): [NO VERIFICADO]. Si los números del grupo se usan en un artículo, el modelo debe declararse y justificarse.

**H3 (estadística).** El tiempo entre bundles es hipoexponencial (etapas 1/Γ_0 y 1/γ): CV² = (Γ_0^{-2}+γ^{-2})/(Γ_0^{-1}+γ^{-1})², Fano asintótico F=CV² (<1, sub-poissoniano). Exacto vs fórmula (Ω,κ): (0.1,0.003) 0.714/0.746; (0.05,0.003) 0.494/0.515; (0.2,0.003) 0.870/0.920; (0.1,0.01) 0.538/0.543; (0.1,0.006) 0.611/0.624; T medio 6011 vs 5876. Útil como diagnóstico (F≈0.5–0.9), pero estándar (renovación): novedad baja. g^(n) de orden superior: no calculado.

## 5. Mapas (fig/)

- fig/mapa_P2_Om_kap.png y fig/mapa_P3_Om_kap.png: color = P_n exacta (episodios), rayas blancas = isocurvas P=0.8/0.9/0.95 de la fórmula cerrada, curva roja = Ω_max(κ) para P_n≥0.9 analítica.
- fig/pareto.png: frente pureza-flujo cerrado y puntos exactos.
- No hice el mapa (λ,Ω) de P_n; la resonancia sí se verificó en (λ,Ω) (tabla de §3).

## 6. Veredicto GO/NO-GO con criterios

| Criterio | Resultado |
|---|---|
| (a) fórmula cerrada no trivial, error <10 % en rango amplio | **Sí con reservas**: P_n y J dentro de ~10 % en 92–97 % de la malla; pero 1−P con error mediano 15 %, y la fórmula descansa en un modelo de tasas (no es Floquet/amplitudes cerrado) |
| (b) novedad aparente | **Parcial**: no vi pureza analítica ni cota e^{−λ²}; búsqueda limitada |
| (c) resultado de diseño útil | **Sí**: piso e^{−λ²}, κ*(n,λ), Ω_max(κ;X), frente de Pareto |

Recomendación: GO solo si el artículo combina esto con (i) la comparación de modelos de disipación (local vs vestida), que cambia las conclusiones numéricas previas del grupo, y (ii) otro giro propio (p. ej. molécula/interferencia). Como artículo único sería incremental.

## 7. Limitaciones y riesgos

- La métrica "episodio" (vuelta al vacío de la cavidad) no es la de Bin (ventanas Monte Carlo) ni la de Muñoz (π_N=λ_N/(λ_1+λ_N)); no reproduje las cifras de Bin (97–99 %). Las purezas no son directamente comparables.
- La pureza por ciclos entre saltos radiativos (memoria) mezcla fusión/partición artificial; P(3), P(4) son en su mayor parte reparto de fonones entre ciclos y la nube DC.
- Rabi efectivo y γφ vía desfase inducido solo validados indirectamente; Ω ≳ 0.3 y κ ≲ 6e-4 fuera del régimen de tasas (Rabi coherente).
- Solo λ ≤ 0.2, n=2,3; T=0; sin baño de fonones continuo ni ensanchamiento del QD.
- H2 (bombeo bicromático, modulación del acoplo) no se evaluó.
- El resultado del piso e^{−λ²} es consecuencia directa del decaimiento radiativo con σD†; si el emisor se reinicia por otro canal (p. ej. reset Raman), el piso cambia.

## 8. Código (carpeta /home/user/nphonons-bundles/nuevo/scoutB/)

- bl.py: maestra completa (lab y polarón, local/vestida), ciclos y episodios exactos (renovación + estadística de conteo por FFT), ε_n, d_{mk}, Ω_n.
- rate.py: modelo de tasas, P_closed, flux_closed, Om_max.
- t1.py..t8.py, variants.py, h3.py, maps.py, final.py: scripts de verificación y figuras; res_*.txt: salidas; data/: mallas exactas.
- Uso: `python3 final.py` (≈10-15 min, 1 proceso); `python3 variants.py`; `python3 h3.py`. Convergencia: P(k) idéntico a 4 decimales para N=7, 9, 12.
- PDFs en pdf/ (no versionar).
