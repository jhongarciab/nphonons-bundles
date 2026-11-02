# Dossier: consistencia de la maestra y pureza cerrada de bundles de N fonones

**Estado:** desarrollo completo; verificado de forma independiente en 5 rondas (reimplementación a ciegas, baño microscópico, Monte Carlo, auditoría de cifras con puntos nuevos y revisión adversaria de figuras). Falta escribir. Fecha: 2026-11-02.
**Título de trabajo (modesto a propósito):** *Master-equation consistency and closed-form purity of n-phonon bundles in a longitudinally coupled emitter–mode system.*
**Revista objetivo:** *Advanced Quantum Technologies* (primera opción) o *Phys. Rev. A* como estudio metodológico de consistencia de la maestra con prueba microscópica. **Estimación subjetiva del revisor adversario (no mía): AQT ≈ 45 %, PRA ≈ 25 %.** Tómese como orden de magnitud.

> **Etiquetas de verificación.**
> **[V-IND]** reimplementado de forma independiente por un agente que no vio mi código y coincide.
> **[V-MICRO]** contrastado con un modelo microscópico completo (emisor + modo + baño explícito).
> **[V-MC]** contrastado con Monte Carlo de trayectorias.
> **[V-LIT]** referencia leída en el PDF o comprobada contra el abstract de arXiv.
> **[NO-V]** no verificado de forma independiente: no citar como hecho.

---

## 0. Veredicto honesto (leer primero)

Este trabajo sale de una búsqueda con tres exploradores en paralelo (molécula de QDs como átomo Jaynes–Cummings de fonones; teoría analítica de bundles; minería de huecos en 30 artículos de 2025–2026). Los resultados fueron: la línea de la molécula de QDs es un **NO-GO** (efecto real pero moderado: g⁽²⁾(0) ≥ 0.07–0.2, Q ≥ −0.65); la idea que otro explorador recomendó (presupuesto de errores del qubit gato con Kerr y fonones térmicos) **ya está hecha en su propia maestría**; y la línea de bundles dio una teoría cerrada verificada. Es la que se desarrolla aquí.

**Lo que aporta, verificado:**
1. **Resonancia** con corrimiento Stark: δ_L = λ² − n + ε_n [V-IND].
2. **Flujo cerrado** J = (n+λ²)/(1/Γ₀ + 1/γ) y **pureza cerrada** con tres canales de pérdida; error ≤ 4.2 % en P en la región de validez [V-IND parcial, V-MC].
3. **Cota de Franck–Condon**: la pureza por ciclo radiativo cumple P ≤ e^(−λ²); por episodios el piso exacto de conteo es 1/(2−e^(−λ²)) [V-IND].
4. **La elección del disipador cambia los resultados.** La maestra *local* (salto √κ b del laboratorio) emite una nube espuria de κλ²/γ fonones por ciclo cuando κ ≪ λ²; la maestra *vestida* es la correcta, y esto se **comprobó con un baño microscópico explícito** (2.0215 microscópico frente a 2.0217 vestida y 2.3145 local; κ = 0.003) [V-MICRO]. Bin et al. (PRL 124, 053601) usan la maestra local con κ = 0.002 < λ² = 0.01 [V-LIT].
5. **La pureza depende de la definición.** Lo que aquí se llama *pureza por episodios* (0.94 en el punto base) es una variable de la cascada, **no observable**; la *pureza operativa* con agrupación por ventana τ_c es 0.70, 0.78 y 0.59 para τ_c = 2/κ, 5/κ, 10/κ en ese mismo punto (máximo 0.78) y solo coincide con la de episodios para κ/γ ≳ 500. Una fórmula operativa reproduce el Monte Carlo [V-MC]. Los números titulares no deben llamarse 'pureza' sin cualificar.

**Lo que NO es, y que el artículo debe decir:**
- **Ninguna plataforma medida cumple hoy el régimen.** Los QDs de GHz tienen λ/ω_b ≈ 0.002 (Spinnler) a 0.21 con banda lateral no resuelta (Kettler); el régimen de Bin (λ = 0.1, γ/ω_b = 2×10⁻⁴) está 50–300× por encima de lo medido. El trabajo es una **teoría corregida con requisitos**, no una propuesta de dispositivo realizable. La crítica de PRA ("no hay arquitectura realizable") sigue aplicando en parte.
- La novedad es **modesta**: la maestra vestida es estándar en QED ultrafuerte y **Bin et al. PRL 127, 073602 (2012.10926) ya la usan y citan a Beaudoin**; lo que no encontré es que alguien compare maestra local y vestida *para este problema*, ni la cuantifique. El artículo debe presentarse como **confirmación y cuantificación** del artefacto κλ²/γ del PRL 124, no como corrección novedosa. La contribución original está en (a) las fórmulas cerradas y la cota, (b) esa cuantificación con prueba microscópica, y (c) la definición operativa de pureza.
- No encontré un trabajo con las cuatro cosas a la vez (pureza cerrada, cota, κ\*, comparación local/vestida), pero la búsqueda cubre solo arXiv y no pude leer el Suplemento de Bin ni el artículo de PRResearch 2025 sobre decaimiento de dos fotones **[NO-V]**.

---

## 1. Modelo

Emisor de dos niveles σ con acoplo longitudinal a un modo bosónico b (ω_b = 1) y bombeo láser:

H = δ_L σ†σ + b†b + λ σ†σ (b+b†) + Ω(σ + σ†),  δ_L = ω_σ − ω_L.

Disipación: pérdida del modo κ, decaimiento radiativo γ, desfase γφ. **Parámetros base de este trabajo** (no son los de Bin; Bin et al. PRL 124 usan λ = 0.1, κ = 0.002, Ω = 0.2, γ = 2×10⁻⁴, γφ = 4×10⁻⁴): λ = 0.14, Ω = 0.1, κ = 0.003, γ = 2×10⁻⁴, γφ = 4×10⁻⁴. Transformación de polarón U = exp[−λσ†σ(b†−b)]: H_p = δσ†σ + b†b + Ω(σ†D + σD†), δ = δ_L − λ², D = exp[−λ(b†−b)].
**Plataformas con esta misma estructura:** QD con fonón de nanocavidad (Bin), ion atrapado con fuerza dependiente del espín, qubit con acoplo longitudinal a un resonador. Ver §5 para sus parámetros.

---

## 2. Resultados con verificación

### R1. Resonancia **[V-IND]**
δ_L = λ² − n + ε_n, con ε_n = −Ω² Σ_{m≠n} d_{m0}²/(m−n) + Ω² Σ_{k≥1} d_{nk}²/k ≈ 2Ω²/n y d_{mk} = ⟨m|D|k⟩ (polinomios de Laguerre).
Error frente al máximo del flujo exacto: **< 0.15 γ_c para Ω ≤ 0.2 (n = 2) y Ω ≤ 0.3 (n = 3)**; en n = 2, λ = 0.14: 0.001, 0.011, 0.14, 0.70, 2.2 γ_c para Ω = 0.05, 0.1, 0.2, 0.3, 0.4. Es ≈ 4× mejor que usar solo 2Ω²/n. Para λ → 0 reproduce el resultado de Bin (régimen III) hasta O(Ω⁴); **Bin-III es mejor solo para Ω ≥ 0.3**. Verificado en 22 puntos distintos de los de ajuste (n = 2, 3; Ω = 0.05–0.2; κ = 0.001–0.01; λ = 0.1–0.2). *Fig. 2.*

### R2. Flujo **[V-IND]**
J ≈ (n+λ²)/(1/Γ₀ + 1/γ), Γ₀ = 4Ω²d_{n0}²/(nκ+γ+γφ). El factor (n+λ²) es el recuento de n fonones de la cascada más λ² del ring-down tras la sacudida de Franck–Condon. Error con la maestra vestida **≤ 6 %, típico 1–2 %**; la maestra local lo excede entre +3 % y +94 %. Media exacta por ciclo 2.0217 frente a 2.0196. *Fig. 2(b).*

### R3. Pureza por episodios, fórmula cerrada **[V-IND parcial, V-MC]**
P_n ≈ e^(−λ²) · Π_j [1 − p_j (r_j + (1−r_j)(1−d_jj²))] · (1 − ε_off), con p_j = γ/(jκ+γ), r_j = Γ_j/(Γ_j+jκ), ε_off la fracción de excitación en las colas lorentzianas n±1.
- **En la región ξ = Ω_n/γ_c(0) < 0.5 y P ≥ 0.85:** error máximo de P ≤ 4.2 % en la malla de ajuste (100 % < 5 %); mediana del error de la **impureza** 3–10 %. **Verificación independiente con 56 puntos nuevos** (n = 2, 3; λ = 0.12, 0.16, 0.18; γ = 2×10⁻⁴ y 5×10⁻⁴; ninguno en las mallas): dentro de la región (40 puntos) error de P máx 1.3 %, mediana 0.5 %; **error de la impureza 1−P: mediana 6.1 %, máximo 17.7 %** (n = 2 con κ grande la subestima 10–16 %, n = 3 la sobreestima hasta 18 %). La métrica relevante para pureza ≥ 0.95 es la impureza.
- **Caveat de método:** definir la región con P exacta ≥ 0.85 usa la respuesta; en un uso real debe definirse con ξ y P_closed.
- **Fuera de ella falla:** hasta 22 % para κ ≲ 10⁻³ y Ω grande; la fórmula previa del explorador subestimaba la impureza 12–29 %.
- Para n = 3, P ≥ 0.95 es inalcanzable con los parámetros base (el máximo cerrado es ≈ 0.950 en κ ≈ 0.0017); n = 4 no es usable (P₄ = 0.185).
- Verificación: la fórmula se contrastó con la maestra exacta en una malla de 20×20 por (n, λ) por su propio autor y con una implementación independiente de H1–H2 (R1, R2) por otro agente; **no hubo una segunda implementación independiente de la fórmula de pureza v2**. *Fig. 3.*

### R4. Cota de Franck–Condon **[V-IND]**
Tras el decaimiento radiativo el oscilador queda en un estado de Poisson(λ²) (sacudida de Franck–Condon). Por tanto:
- **por ciclo radiativo: P_n^cyc ≤ e^(−λ²)** (cota comprobada numéricamente bajo la hipótesis de independencia entre el número de fonones previo y el Poisson(λ²) residual; máximo en una rejilla de 75 puntos 0.9616 frente a 0.9806 para n = 2, λ = 0.14, **no un máximo exacto**);
- **por episodios: el piso exacto de conteo es 1/(2−e^(−λ²)) ≥ e^(−λ²)** (diferencia O(λ⁴): 4×10⁻⁴ en λ = 0.14);
- **"para cualquier definición de bundle" es falso**: con la fracción de fonones en bundles se llega a n/(n+λ²) = 0.985 > 0.9806.
El máximo exacto sobre κ queda 0.65–1.3 % por debajo del piso en n = 2: **la cota no es ajustada**. *Fig. 5(b).*

### R5. Maestra local frente a vestida **[V-MICRO] [V-MC]**
- En el marco de polarón la maestra local de laboratorio se transforma en el salto **b + λσ†σ** (un error de signo del explorador, corregido; coincide con el laboratorio a 7×10⁻¹¹).
- Su componente DC emite una nube coherente de **κλ²/γ fonones por ciclo** (razón medida/predicha 0.996 para κ ≤ 5.5×10⁻³). Para n = 2, Ω = 0.1, κ = 0.003: **P_ciclo = 0.805 (vestida; MC 0.803 ± 0.004) frente a 0.609 (local; MC 0.615 ± 0.004)**; por episodios 0.94 frente a 0.72. La tendencia con κ se **invierte**: la vestida crece, la local cae después de κ ≈ 2×10⁻³.
- **Prueba desde primeros principios (WP2):** emisor + b + segundo modo c amortiguado (sin RWA, regla de oro para g_c, convergencia en truncaciones hasta 5 cifras), extrapolando κ_c → 0. Magnitudes: fonones medios emitidos por ciclo radiativo (izquierda) y P₂ por ciclo (derecha):

| κ | microscópico | vestida | local | P₂ micro | P₂ vestida | P₂ local |
|---:|---:|---:|---:|---:|---:|---:|
| 0.003 | 2.0215 | 2.0217 | 2.3145 | 0.7975 | 0.8050 | 0.6095 |
| 0.01 | 2.0160 | 2.0137 | 2.9912 | 0.9068 | 0.9088 | 0.4518 |
| 0.03 (> λ²) | 1.9626 | 1.9657 | 4.9049 | 0.8472 | 0.8994 | 0.2351 |

  **Cautelas:** (i) en κ = 0.003 la extrapolación usa κ_c = 0.03–0.1 (κ_c/κ ≥ 10, baño markoviano) y es convincente; en κ = 0.01 y 0.03 los κ_c usados (0.05–0.2) son **mayores que κ**, el coeficiente de κ_c² crece de 2.8 a 49 y la extrapolación (3 puntos, 2 parámetros) es poco fiable; (ii) para κ = 0.03 > λ² la vestida deja de ser rigurosa (P₂ micro extrapolado 0.847 frente a 0.899).
- **Precisión del enunciado:** la condición κ ≪ λ²ω_b es conservadora (la vestida funciona bien en κ = 0.01 ≈ λ²/2). El **parámetro de control real es κλ²/γ** (el tamaño del artefacto relativo a γ), y la hipótesis física es que la **densidad espectral del baño se anule en ω → 0** (baño de posición con J(0) = 0, T = 0). **Contraejemplo:** un baño plano con J(0) ≠ 0 acoplado a (b+b†) añade desfase λ²J(0) sobre σ†σ, que es justo lo que la local contiene; ahí la local sería correcta. El artículo debe discutir qué baño corresponde a cada plataforma (un fondo de fonones acústicos tiene J(0) = 0).
- **Disipador frente a operador de conteo:** parte de la discrepancia viene de que la local cuenta el operador b (que contiene la parte DC) y no el operador de salida de frecuencia positiva (Ridolfo 1206.0944). Conviene separar ambos efectos. *Fig. 4.*
- **Consecuencia para la literatura y para ustedes:** Bin et al. (PRL 124, 053601) usan la maestra local con κ = 0.002 < λ² = 0.01 [V-LIT, texto del PDF; el Suplemento no está en el PDF]; su PRL 127 posterior (2012.10926) usa la vestida. La pureza por ciclos de 0.61 de su documento `memoria/docs/IDEAS_VENTAJAS_MOLECULA.md` coincide exactamente con la maestra local (0.609) y es en buena parte artefacto de la nube κλ²/γ; con la vestida es 0.805.
- **Convención de signo:** con H_p = T†HT, T = exp[−λ n(b†−b)] resulta σ → σD(−λ); hay que fijar el signo de λ en D al publicar.

### R6. Pureza operativa **[V-MC]**
La pureza por episodios (retorno de la cavidad al vacío) **no es operativa**: con agrupación de eventos por hueco τ_c, P_MC depende fuertemente de τ_c y solo coincide con la de episodios para κ/γ ≳ 500. En el punto base: P_MC(2/κ, 5/κ, 10/κ) = 0.70, 0.78, 0.59 frente a 0.938 por episodios (partición ∝ e^(−κτ_c) y fusión de bundles ∝ γτ_c). La predicción operativa P_op(τ_c) = P_ep (1−q)(1−s)/(1+s) reproduce el MC con diferencia absoluta media 0.0075 (8 de 48 valores > 3σ). Con κ/γ = 500: 0.9641 ± 0.0020 (MC) frente a 0.9655. El MC (400 trayectorias por punto, 16 puntos; flujo coincide con el exacto en 0.9 % de media) valida la **estadística de conteo**, no la maestra. *Fig. 5(a).*

### R7. Diseño
- Ω_max(κ; P ≥ 0.95), n = 2, λ = 0.14 (recalculado por la revisión: 0.0416, 0.0852, 0.2201): 0.042 (κ = 0.002), 0.085 (0.003), 0.22 (0.005); sin límite para κ ≥ 0.01 hasta Ω = 0.3. Para P ≥ 0.90: 0.109 (0.002), 0.36 (0.003).
- κ\* depende de Ω: λ = 0.14, n = 2: 0.0071, 0.0086, 0.0107 para Ω = 0.05, 0.1, 0.2; el máximo **exacto** de P₂ en κ (Ω = 0.1) es 0.0089–0.0093 (λ = 0.14) y 0.0061–0.0063 (λ = 0.1); es un máximo plano, incertidumbre ±5–10 % según la rejilla. Los valores del explorador (0.0043, 0.0028) **no son reproducibles** y no se citan.
- El desfase γφ casi no cambia P (0.9378 → 0.9407 para γφ de 10⁻⁶ a 7×10⁻³) pero reduce el flujo; γ domina la pureza: P = 0.972 (γ = 2×10⁻⁵), 0.942 (1.8×10⁻⁴), 0.764 (1.6×10⁻³). *Fig. 5(c).*
- **Mapa de requisitos (Fig. 6, rejilla fina: 36×36 en (λ, γ), 50 valores de κ, 30 de Ω; n = 2, γφ = 2γ, ξ < 0.5, Ω ≤ 0.3, P₂^closed ≥ 0.9):** flujo máximo J ≥ 10⁻⁶ ω_b exige λ ≳ 0.005; **J ≥ 10⁻⁵ ω_b exige λ ≳ 0.011**; **J ≥ 10⁻⁴ ω_b exige λ ≳ 0.030**; en los tres casos γ/ω_b ≲ 6.6×10⁻³. La región no se resume con dos cotas independientes (a λ = 0.012 hace falta γ ≲ 1.5×10⁻⁴). Con una rejilla más gruesa el máximo de J cambia hasta 45 %, pero λ_min es estable. **Para λ < 0.1 la fórmula está extrapolada (no validada).**

---

## 3. Errores del trabajo previo (explorador B) hallados y corregidos

| Error | Efecto | Corrección |
|---|---|---|
| Signo de la maestra local en polarón (`bl.py:55`) | ΔP(k) hasta 1.6×10⁻³ | b + λσ†σ; coincide a 7×10⁻¹¹ |
| Fórmula cerrada sin la pérdida de FC de decaimientos a mitad de cascada (`rate.py:83`) | Impureza subestimada 12–29 % | Término añadido (v2): 3–10 % |
| Desfase inducido γφ(Ωd/Δ)² sin κ ni γ | Inconsistente con las colas | Tasa lorentziana única |
| Ω_max con tope silencioso 0.5 | Puntos fuera de validez en Pareto | Devuelve inf/nan; Ω ≤ 0.3 |
| κ\* no reproducible | Citas erróneas | Recalculado; depende de Ω |
| "Cota dura e^(−λ²)" sin definir | Falso por episodios | Cota por ciclo + piso por episodios |
| Mallas 8×8 | "Error máx 10 %" no extrapolable | Malla 20×20: máx 18–22 % |

---

## 4. Literatura y novedad **[V-LIT salvo indicación]**

Verificadas (PDF leído): Bin et al. 1907.12714 (PRL 124, 053601; **maestra local**; Suppl. no incluido; parámetros ω_b/2π = 1 THz, λ/ω_b = 0.1, κ/ω_b = 0.002, γ/ω_b = 2×10⁻⁴, γφ/ω_b = 4×10⁻⁴, Ω = 0.2 ω_b; pureza por Monte Carlo); Bin et al. 2012.10926 (PRL 127, 073602; maestra **vestida**, saltos solo m > n, cita a Beaudoin; sin comparar con la local; parámetros inconsistentes entre texto y pie de figura); Muñoz 1306.1578; Cosacchi 2108.03967; Deng 2011.03886. Maestra vestida: Beaudoin 1107.3990 (PRA 84, 043832; aproximación secular en la base de H_S; un baño σ_z da un término de frecuencia cero); Ridolfo 1206.0944 (operador de salida de frecuencia positiva X⁺, excluye la parte DC); Settineri 1807.06348.
**Ninguno** da fórmulas cerradas de pureza, cota, ni compara maestra local y vestida para este problema. Los trabajos recientes 2604.15604, 2604.15605, 2609.07454, 2603.27160, 2610.09971 tampoco (2609.07454 trata la fuga ∝ g²/Ω² para emisores finitos).
**No localizado:** el artículo de PRResearch 2025 "Two-photon decay enhanced even photon bundle emission" (8 búsquedas) **[NO-V]**.
Las referencias con BibTeX verificado están en `lit/refs_bundles.bib`.

---

## 5. Contraste con plataformas medidas (hallazgo honesto)

| Plataforma (fuente) | λ/ω_b | κ/ω_b | γ/ω_b | κ ≪ λ² | Observación |
|---|---:|---:|---:|:-:|---|
| Propuesta de Bin | 0.1 | 0.002 | 2×10⁻⁴ | sí (R = 5, marginal) | Parámetros de partida |
| QD, Spinnler 2311.05342 (1.466 GHz, g_ep = 2.9 MHz, Q = 2.1×10³, τ = 1.18 ns) | 0.002 | 4.8×10⁻⁴ | 0.092 | no (R ≈ 0.008) | Banda lateral no resuelta |
| QD en microhilo, Kettler 2105.04223 (330 kHz, g = 68 kHz, Q = 1650) | 0.21 | 6×10⁻⁴ | ≈ 480 | sí (R ≈ 70) | No resuelve bandas laterales |
| Ion Ca⁺, Lo 1412.7100 (2.1 MHz) | ≈ 0.003 | n/d | n/d | no | γφ ≈ 5.6×10³ s⁻¹ ≫ λ²ω ≈ 130 s⁻¹ |
| Qubits superconductores | n/d | n/d | n/d | n/d | Solo acoplo longitudinal paramétrico débil; sin demostración estática ultrafuerte |

Conclusión: **λ = 0.1–0.2 es 50–300× superior a lo medido en QDs de GHz.** Los puntos medidos caen en la zona sin solución de la Fig. 6 (el de Spinnler, γ/ω_b = 0.09, está en el régimen sin banda lateral resuelta). Fuente: WP3, números de abstract salvo los marcados [TEXTO] en `lit/lit_bundles.md`.

---

## 6. Lo que NO se debe afirmar

1. Que existe hoy una plataforma que cumpla el régimen (λ, κ, γ) de la propuesta de Bin.
2. Que la maestra local "es incorrecta" en general: solo cuando κ ≲ λ²ω_b y la densidad espectral del baño se anula en ω → 0. Para κ ≥ λ² la vestida tampoco es rigurosa.
3. Que la cota e^(−λ²) vale "para cualquier definición de bundle".
4. Purezas por episodios sin declarar la definición, ni P₃ en parámetros base (dominada por excitaciones fuera de resonancia).
5. Los valores κ\* y Ω_max del explorador original.
6. Que la maestra vestida ha sido derivada microscópicamente para el sistema completo: solo se comprobó con un baño de posición a T = 0.
7. Extrapolaciones a λ < 0.1 como validadas.
8. Que el trabajo "reproduce" la pureza de 97–99 % de Bin: la métrica es distinta y no se reprodujo.
9. Que la corrección de la maestra es novedosa respecto a Bin et al. PRL 127 (ya usa la vestida).
10. Los números de scoutA (g⁽²⁾(0) ≥ 0.07–0.2, Q ≥ −0.65): no se comprobaron en la revisión final.

---

## 7. Limitaciones

- T = 0, un solo modo, sin baño continuo de fonones ni inhomogeneidad.
- Fórmulas cerradas validadas en λ = 0.1–0.2, n = 2, 3; la región Ω ≳ 0.3 o κ ≲ 6×10⁻⁴ (Rabi coherente) queda fuera.
- La definición de episodio por clasificación pre-salto deja términos cruzados sin contar (pseudo-desenrollado); la dinámica total es la física y el MC la respalda.
- La prueba microscópica del disipador usa un baño de posición con extrapolación κ_c → 0; con κ = 0.03 > λ² la extrapolación de 3 puntos es poco fiable.
- Los datos de Fig. 4(c) se transcribieron de `verif/C2.log` (números exactos en el script).

---

## 8. Figuras (en `bundles/figs/`, con el kit de estilo `nuevo/kit/figstyle.py`)

| Figura | Contenido |
|---|---|
| `fig1.png` | (a) esquema del sistema, (b) diagrama de niveles en el marco de polarón con la resonancia n y los canales n±1, (c) registro de detecciones de fonones y fotones con agrupación en bundles |
| `fig2.png` | (a) espectro de flujo con las posiciones analíticas, (b) forma de línea analítica sobre maestra exacta (Ω = 0.05, 0.1, 0.2), (c) error de la posición de la resonancia: ε_n frente a Bin-III y frente a 2Ω²/n |
| `fig3.png` | (a, b) mapas P₂ y P₃ exactos con isocurvas cerradas y Ω_max; (c) flujo; (d) cortes en κ con κ\* |
| `fig4.png` | (a) pureza por ciclo local frente a vestida con MC; (b) fonones extra frente a κλ²/γ; (c, d) prueba con baño microscópico: fonones por ciclo y P₂ |
| `fig5.png` | (a) pureza operativa frente a τ_c; (b) piso de Franck–Condon; (c) efecto de γ |
| `fig6.png` | Mapa de requisitos en (λ, γ/ω_b): flujo máximo con P₂ ≥ 0.9; zona gris = sin solución; contornos 10⁻⁶ (sólido), 10⁻⁵ (discontinuo), 10⁻⁴ (punteado); plataformas medidas |

---

## 9. Esquema del artículo

**Mensaje:** *La emisión de bundles de N fonones tiene una teoría cerrada y verificada (resonancia con Stark, flujo, pureza de cascada, cota por ciclo); la maestra local produce una nube espuria de κλ²/γ fonones por ciclo, confirmada con un baño microscópico, y la vestida (ya usada en Bin et al. PRL 127) es la correcta; la pureza debe definirse operativamente; y los requisitos (J ≥ 10⁻⁵ ω_b exige λ ≳ 0.011 y γ/ω_b ≲ 7×10⁻³) están muy por encima de lo medido hoy.*

1. Introducción (bundles, Bin et al., QED ultrafuerte y maestras vestidas).
2. Modelo y transformación de polarón (Fig. 1).
3. Resonancia y flujo (R1, R2; Fig. 2).
4. Pureza: definiciones, fórmula cerrada, cota (R3, R4, R6; Figs. 3, 5).
5. Elección del disipador (R5; Fig. 4).
6. Diseño y requisitos frente a plataformas (R7, §5; Fig. 6).
7. Discusión y conclusiones.
Apéndices: derivación de ε_n y de la fórmula de pureza; método de renovación para P(k); modelo microscópico del baño; tabla de errores.

**Qué lo hace flojo para PRA:** novedad media; sin plataforma que cumpla; el efecto central (corrección de maestra) es técnico. **Qué lo defiende:** corrige un modelo usado en la literatura con una prueba microscópica, fórmulas cerradas con errores acotados y una cota con definición precisa. **Mitigación:** presentarlo como "teoría consistente y requisitos", no como propuesta de dispositivo.

---

## 10. Reproducibilidad

- `nuevo/bundles/src/bundles_core.py` (biblioteca), `bundles_mc.py` (Monte Carlo), `run_D1.py … run_D7.py`, `run_C.py`; `tests/run_tests.py` (14–15 pruebas, ≈ 8 s); datos en `bundles/data/*.npz` y `tablas.md`.
- Figuras: `bundles/figs/fig1.py … fig6.py` (≈ 1–2 min cada una; Fig. 6 ≈ 20 s).
- Verificación independiente: `nuevo/verif/informe_WP2.md` (y su código); literatura: `nuevo/lit/lit_bundles.md` y `refs_bundles.bib`; exploradores: `nuevo/scoutA/B/C/informe_*.md`; informe de datos: `bundles/informe_WP1.md`.
- Requisitos: Python 3.13, numpy, scipy, qutip 5.3, matplotlib. **Con BLAS multihilo cada solve es 40× más lento: fijar `OMP_NUM_THREADS=1`.**
- Los resultados de `memoria/` y de `doble_carril_fonon/` no se tocan.
