# Informe VF: revisión adversaria del dossier de bundles de N fonones

Scripts propios: `a.py`, `b.py`, `c.py`, `d.py` (y `R.npy`, 56 puntos de la tarea 2). OMP_NUM_THREADS=1, máximo 2 procesos.

## 1. Auditoría numérica

### 1.1 Cifras que coinciden con datos
- Tabla de R5 contra `C2.log`: los nueve valores coinciden (κ=0.003: 2.0215/2.0217/2.3145; κ=0.01: 2.0160/2.0137/2.9912; κ=0.03: 1.9626/1.9657/4.9049). Refit propio de κ=0.003 (cuadrático en κ_c², coef. 2.77): 2.0216. P₂ de κ=0.03 (0.847 frente a 0.899) correcto.
- Ω_max(κ; P≥0.95) = 0.0416, 0.0852, 0.2201 (el dossier redondea a 0.042, 0.085, 0.22); inf para κ≥0.01. Ω_max(P≥0.90) = 0.1086 y 0.360. Correcto.
- κ* = 0.00710, 0.00861, 0.01066 para Ω = 0.05, 0.1, 0.2. Correcto.
- Máximo exacto de P₂ en κ (Ω=0.1): rejilla fina propia 0.0093 (λ=0.14, P=0.9706) y 0.0063 (λ=0.1, P=0.9815). El dossier da 0.0089 y 0.0061. Es un máximo plano y depende de la rejilla; las dos cifras no son contradictorias, pero el dossier no declara la incertidumbre (±5–10 %).
- Efecto de γ y γφ (P exacta): 0.9378→0.9407 para γφ de 1e-6 a 7e-3; P = 0.972/0.942/0.762 para γ = 2e-5/1.8e-4/1.6e-3. El dossier dice 0.764 (0.762 aquí); el valor es muy sensible a la elección de la rejilla del γ en 1.6e-3. Sin importancia.
- P₃ máx cerrada = 0.9497 en κ=0.00167 (dossier "≈0.950, κ≈0.0017"); P₄ = 0.185 (exacto, correcto).
- Parámetros de §5 (λ, κ, γ de Spinnler y Kettler) se recalculan bien a partir de los datos citados (g/ω, ω/Q, 1/τω).

### 1.2 Discrepancias, redondeos engañosos y atribuciones erróneas
1. **(Crítica) Fig. 6 / R7: "un flujo ≥ 1e-5 ω_b exige λ ≳ 0.03 y γ/ω_b ≲ 3e-3" es incorrecto.** Con `R_requirements.npz`:
   - J ≥ 1e-5 tiene λ_min = 0.0106 y γ_max = 6.6e-3.
   - J ≥ 1e-4 tiene λ_min = 0.0308 y γ_max = 6.6e-3.
   - La frase mezcla los dos umbrales: λ ≳ 0.03 corresponde al contorno 1e-4, no a 1e-5.
   - La región J ≥ 1e-5 es un conjunto en dos parámetros (a λ=0.0124 hace falta γ ≲ 1.5e-4; a λ=0.03, γ ≲ 3e-3; a λ ≥ 0.08, γ ≲ 6.6e-3). No se resume con dos cotas independientes.
   - La malla gruesa es poco robusta: refinando κ (22→80) y Ω (12→40) el máximo J crece entre 0 % y 45 % (p. ej. λ=0.1, γ=6.6e-3: 1.6e-4→2.35e-4; λ=0.0124, γ=1.5e-4: 1.13e-5→1.19e-5). La frontera λ_min(1e-5) queda en 0.0106 (J=1.3e-5 en esa celda; fina 1.3e-5; en λ=0.0091, 7.0e-6) y es estable. La frontera del borde superior de γ cambia por una celda. Es decir, la conclusión cualitativa es robusta, la cifra concreta no.
   - Además `fig6.py` fija γφ=2γ (no 4e-4 como en la base) y exige ξ<0.5, Ω≤0.3, P_closed≥0.9: hay que declararlo en el pie de figura. Para λ<0.1 la propia fórmula está fuera de validez, así que el "λ ≳ 0.011" es una extrapolación.
2. **(Importante) §1 "Parámetros base (Bin et al.): λ=0.14, Ω=0.1, κ=0.003"** contradice §4 y §5: Bin PRL 124 usa λ=0.1, κ=0.002, Ω=0.2. Los parámetros base son del propio trabajo (o del explorador), no de Bin. Fig. 2(a) y todos los números de base (0.938, 0.805, 0.609…) son del punto propio.
3. **(Importante) §0 punto 5: "por ventana temporal τ_c de 0.59 a 0.95 en el mismo punto".** En el punto base P_MC(2/κ,5/κ,10/κ) = 0.70, 0.78, 0.59; el 0.95 es de otro punto (κ/γ=500 o la ventana de 5/κ con κ/γ=50). Al "mismo punto" la pureza operativa máxima es 0.78. La frase da una imagen más favorable de la operatividad de la que hay.
4. **Tabla de R5:** no dice que las cifras son fonones medios emitidos por ciclo; omite los P₂ microscópicos de κ=0.003 (0.7975 frente a 0.8050 vestida, −1 %) y κ=0.01 (0.9068 frente a 0.9088), que están en `C2.log` y son el test más estricto. Además, los tres valores de κ_c usados en κ=0.01 (0.05–0.15) y κ=0.03 (0.1–0.2) son mayores que κ: se extrapola desde un régimen en que la condición de baño markoviano (κ_c ≫ κ) se viola, y el coeficiente κ_c² crece de 2.8 a 49. La concordancia en κ=0.003 (κ_c≥0.03, κ_c/κ≥10) es convincente; en 0.01 y 0.03 la extrapolación está sobreajustada (3 puntos, 2 parámetros). Ya se admite parcialmente en §7.
5. **R4: "máximo numérico 0.9616 frente a 0.9806".** El 0.9616 es el máximo sobre una rejilla de 75 puntos (WP2, κ=0.01, Ω=0.03, γ=2e-5), no un máximo exacto; en WP1 el máximo por ciclos de n=2 sube solo a 0.78–0.92 para λ=0.05–0.3. La frase "la cota no es ajustada" es cierta, pero la cota "dura" solo está demostrada con la hipótesis de independencia N_x ⟂ Poisson(λ²) y comprobada numéricamente, no probada en general (el propio WP2 lo dice).
6. **§0 "error ≤4.2 % en P"**: es el máximo sobre la región ξ<0.5 de la malla de ajuste (λ=0.1, 0.14, 0.2). Ver 2: sobre puntos nuevos es 1.3 %, pero en la **impureza** 1−P el error llega a 18 % dentro de la región de validez. El dossier solo cita "mediana 3–10 %" y omite el máximo de la impureza (WP1: hasta 30 % en λ=0.1). La métrica relevante para un artículo de pureza ≥0.95 es la impureza.
7. R1: "Bin-III es mejor solo para Ω ≥ 0.3": WP1 confirma (0.56 frente a 0.70 γ_c); la comparación de Fig. 2(c) usa Bin-III+λ², no la resonancia exacta de Bin sin λ². No recalculado.
8. §0 cita g⁽²⁾ ≥ 0.07–0.2 y Q ≥ −0.65 (scoutA): no comprobados aquí.
9. §10 "tests/run_tests.py": no verificado que exista el directorio de tests.

## 2. Verificación independiente de P_closed (v2)
56 puntos nuevos: n∈{2,3}; λ∈{0.12, 0.16, 0.18}; Ω∈{0.04,0.07,0.13,0.17}; κ∈{0.0012,0.0022,0.0035,0.0075}; γ∈{2e-4, 5e-4}; γφ=4e-4; ninguno en las mallas D2. P exacta por episodios con `exact_point` (maestra vestida).

| Subconjunto | N | error P max | error P mediana | error impureza max | error impureza mediana |
|---|---:|---:|---:|---:|---:|
| Región ξ<0.5 y P_ex≥0.85 | 40 (71 %) | 1.32 % | 0.47 % | 17.7 % | 6.1 % |
| Fuera de la región | 16 (29 %) | 11.0 % | 3.2 % | 20.5 % | 11.9 % |

- El enunciado "error ≤4.2 % en P en la región" **se cumple** (máx 1.3 %), y la mediana de la impureza (6 %, rango del dossier 3–10 %) también.
- Dentro de la región la impureza llega a 15–18 % (n=2, Ω=0.17, κ=0.0075; n=3, κ=0.0012). Hay sesgo con signo: para n=2 con κ grande se subestima la impureza (−10 a −16 %); para n=3 se sobreestima (+5 a +18 %).
- Fuera de la región: todos los puntos con κ=0.0012 y n=2 (ξ≥0.8) fallan por 3–11 % en P (la fórmula sobrestima la impureza en 17–21 %: da P=0.58 donde el valor exacto es 0.65). Ahí la fórmula es solo cualitativa.
- Los puntos n=3 se comportan bien salvo κ=0.0012 con γ=5e-4 y λ=0.18 (+5.5 %, ξ=0.18: P_ex=0.7476 frente a 0.7888): el criterio ξ<0.5 por sí solo no basta; el criterio P≥0.85 oculta el fallo porque filtra con el valor exacto (selección con la respuesta; en un uso real no se conoce P_ex). Recomendable definir la región con cantidades de la fórmula (ξ y P_closed).
- Comparación de interpolación: λ∈{0.12,0.16,0.18} no se ajustó, así que la formula generaliza dentro de 0.1–0.2. Fuera de ese rango (Fig. 6, λ<0.1) no está validada.

## 3. Crítica física del mensaje central

**(a) "La maestra local es incorrecta para κ ≲ λ²".**
- La lógica es sólida en lo esencial. El salto local b se transforma en b+λσ†σ: la parte DC λσ†σ describe un baño que acopla al desplazamiento estático del oscilador, que en el cálculo cuenta κλ² fonones/tiempo en el estado excitado aunque no se emita ninguno real. Exceso medido κλ²/γ = 0.293 (2.3145−2.0217), coincide con la predicción.
- El test microscópico (b acoplado a c con (b+b†)(c+c†), c amortiguado, conteo de los saltos de c) **es el test relevante**: lo medible es el cuanto que sale al continuo, y lo que cuenta es lo que emite c, no la "cuenta" de b. Que dos modelos concuerden es sólido. Lo que NO prueba:
  1. El baño tiene J(ω) lorentziana centrada en ω_b con J(0)≈0 por construcción; justamente la condición bajo la cual la vestida se justifica por teoría estándar. El test confirma la implementación, no la generalidad.
  2. Es una prueba de un solo parámetro (κ, κ_c) en un punto base (λ=0.14, Ω=0.1), T=0, sin RWA en el acoplo bosónico.
- **Contraejemplo (baño con J(0)≠0):** un baño de espectro plano (ruido blanco) acoplado a (b+b†) añade desfase λ²·J(0) sobre σ†σ, es decir, exactamente lo que la local da (los saltos b+λσ†σ contienen el desfase DC). Para un baño óhmico (J∝ω^s, s≥1) o de cavidad, J(0)=0 y la vestida gana. Para un modo mecánico sobre un fondo de fonones acústicos (el caso de las QDs) J(0)=0, pero el artículo debe discutir explícitamente cuál es el baño de la plataforma; ahora solo lo hace §6 punto 2 y de pasada.
- El enunciado "incorrecta para κ ≲ λ²" parece una condición de κ, pero lo que importa es la densidad espectral en frecuencia cero y la jerarquía κ ≪ ω_b. De hecho en el microscópico la vestida funciona a κ=0.01 ≈ λ²/2, así que κ≲λ² no es el criterio; es un criterio sobre el tamaño del artefacto (κλ²/γ ≳ 1 relativo a la tasa de decaimiento radiativo γ). Sugerir "κλ²/γ ≳ 0.1" como parámetro de control.
- Cuidado con el uso de "incorrecta": la maestra local es la que Bin usa y lo que hace es contar un operador de salida (b) que no es el de frecuencia positiva (Ridolfo). Parte de la discrepancia es del operador de salida/detección y no de la maestra. Conviene separar ambos efectos (disipador frente a operador de conteo).

**(b) ¿Es "pureza por episodios" medible?** No. Un episodio se define por el retorno de la cavidad al vacío (variable oculta), con clasificación por número de fonones pre-salto y términos cruzados descartados (pseudo-desenrollado). El propio R6 muestra que la pureza operativa en el punto base es 0.59–0.78 y que coincide con la de episodios solo para κ/γ≳500. Los números titulares (0.94, 0.98) son "pureza intrínseca de la cascada", no una cantidad observable. Se puede llamar "fracción de episodios de n fonones" o "pureza de cascada", no "pureza" a secas. P_op de R6 sí es medible pero solo se valida con 48 valores (8 >3σ).

**(c) Honestidad sobre la novedad.** Bin et al. 2012.10926 ya usa la maestra vestida (saltos solo m>n) y cita a Beaudoin. El dossier lo dice (§4) pero el §0 punto 4 y el mensaje central presentan la "corrección" como aportación mientras que los propios autores la corrigieron en el trabajo siguiente. Lo que queda: cuantificación del artefacto (κλ²/γ), test microscópico, fórmulas cerradas y cota. Novedad modesta, y el artículo debe decir que "la corrección ya está implícita en Bin 2021".

**(d) ¿Sobre-promete?** El título "Polaron theory of n-phonon bundle emission: resonance, purity, the choice of master equation, and requirements" es amplio: (i) "teoría" cuando son aproximaciones numéricas validadas en n=2,3, λ=0.1–0.2, T=0, un modo; (ii) "requisitos" cuando el resultado es que ninguna plataforma cumple. El "NO-GO" de §0 (efecto real pero moderado) y la Fig. 6 sugieren un título más modesto: "Master-equation consistency and closed-form purity of n-phonon bundles in a longitudinally coupled emitter–mode system".

**(e) Objeciones de un revisor de PRA (por prioridad)**
1. Sin plataforma realizable: λ=0.1 es 50–300× mayor que lo medido, γ/ω_b en QDs es 0.09–480; la Fig. 6 lo admite. Falta un sistema donde el régimen sí exista (iones atrapados con λ_ef y γφ medidos, resonadores superconductores con acoplo longitudinal).
2. Novedad: técnica estándar (polarón, maestra vestida), el resultado central ya aplicado en Bin 2021; las fórmulas son extensiones de tasas de Bin.
3. La pureza titular no es observable (6b); la operativa depende de τ_c y es mucho menor.
4. La prueba microscópica es de un tipo de baño, extrapolación de 3 puntos a κ_c→0 y con κ_c>κ, solo T=0; sin baño térmico ni inhomogeneidad. Un baño de fonones acústicos real (J∝ω³, T≠0) cambia γφ (desfase por ruido de fonones térmicos), lo que se descarta en §7.
5. Validación de las fórmulas cerradas solo en n=2,3 y λ∈[0.1,0.2]; error de la impureza 6 % mediano, hasta 18 % en la región; la fórmula "no tiene derivación" en el dossier (apéndice).
6. Fig. 6 con rejilla gruesa y γφ=2γ ad hoc; extrapolada a λ<0.1.
7. El gran punto: sin comparación con experimentos o con el propio resultado de Bin (0.97–0.99) que dicen "no reproducimos" (§6 punto 8).

## 4. Revisión de figuras
Referencia mirada: arXiv:2609.18547 (PRA, pág. 3: esquema de niveles pequeño, mapas 2D g⁽²⁾ de doble columna con barra de color estrecha, tipografía 6–7 pt uniforme, paneles etiquetados (a),(b), sin recuadros negros). Las figuras del dossier son de tamaño y estilo comparables, con etiquetas de panel en recuadro negro (poco habitual en APS) y fuente serif con tamaño variable.

| Fig. | Problemas | Coherencia con el texto | Calificación |
|---|---|---|---|
| 1 | (b): flechas grises y rosa cruzadas, "Ω_n" y "γ (Franck-Condon)" apretados; (c): ejes de tiempo 0–20000 con 20 eventos (diagrama ilustrativo sin indicar que es una trayectoria simulada; falta el parámetro usado); etiquetas n=1,2,3 sin explicar que son filas del tipo de bundle; "cluster gap<5/κ" mezcla con colores de las barras verdes | Coincide con la descripción | Aceptable. Falta indicar parámetros; las flechas del (b) pueden limpiarse |
| 2 | (b): texto "lines: closed form, symbols: master eq. (n=2)" encima de las curvas azules; (c): leyenda tapa la línea gris "2Ω²/n only"; el eje y con tres órdenes de magnitud mezcla unidades γ_c; en (a) el rótulo de parámetros solapa la línea n=2 y el "γφ" | Coincide con R1; (c) muestra ε_n peor que Bin-III para Ω ≥ 0.3 y para n=2 en Ω=0.1 la línea azul está por encima de la naranja en 0.1 (no mencionado en el texto) | Aceptable-buena en contenido; edición pendiente (solapamientos) |
| 3 | (a) leyenda de la curva roja sobre fondo; contornos con etiquetas cortadas; (b) píxeles muy gruesos (cuadrícula 20×20 visible, bandas horizontales); (c) mapa de flujo casi uniforme, rango -5 a -3.25 con paleta que no discrimina, contorno "-4.0" cortado; (d) estrellas κ* y círculos ocultan datos; paneles de tamaños distintos; sin barra común | Coincide: Ω_max (roja) y κ*; (d) muestra que la cerrada se desvía bastante del exacto para λ=0.2 (la banda verde ≈ 0.01–0.02 por debajo), no cuantificado en el texto | Aceptable. Interpolar o rejilla más fina; (c) mejora con rango propio |
| 4 | (a) leyenda "P_dr e^{-κλ²/γ}" ilegible (el marcador azul tapa el texto); marcadores azul/cuadrado rojo sin leyenda; (b) correcto (línea sobre marcadores) pero la línea negra gruesa oculta los marcadores; (c) la figura clave es la más débil: marcadores κ_c=0 (×, +, ◇) superpuestos en el borde del eje, el eje x empieza en 0 con los símbolos cortados, sin barras de error, y mezcla extrapolación con datos; el texto "at κ_c=0: ◇ extrapolated…" sin separación | Coincide con R5; pero (c) muestra el valor medio, mientras que P₂ microscópico (más discriminante) no se muestra | (a),(b) aceptable; (c) insuficiente tal cual |
| 5 | (a) título "dashed: episode purity; lines: operational closed form" cortado por el borde del eje, ejes sin sentido en x (solo 3 puntos unidos por línea, la línea sugiere continuidad), eje y hasta 1.1; (b) leyenda desplazada fuera del marco (texto "e^{-λ²}" y "1/(2−e^{-λ²})" solapa las marcas del eje y, "master eq." tapa el recuadro (b)); (c) doble eje con naranja/azul: aceptable pero la leyenda tapa los datos | Fig. 5(a) coincide con R6; (c) la cerrada sobreestima P₂ en γ ≳ 5e-4 hasta 5 % no comentado; el texto afirma "γ domina la pureza" y la figura lo confirma | Insuficiente (editorial: leyendas rotas) |
| 6 | Región gris ambigua: mezcla "sin solución" (NaN) con "extrapolada (λ<0.1)" con el mismo tono, y la zona λ<0.1 válida tampoco se distingue de los píxeles; contornos 1e-4/1e-5/1e-6 con solo dos etiquetas: el contorno de la izquierda (λ≈0.005 a γ=1e-5) es el de 1e-6 y se confunde con el de 1e-5 (el de 1e-5 cruza γ=1e-5 en λ=0.0106); píxeles escalonados; la flecha de Kettler sale del panel hacia ningún marcador; el marcador de Spinnler se corta en el borde del eje; barra de color con etiqueta larga vertical | No coincide con el texto: "λ ≳ 0.03" corresponde al contorno 1e-4, no 1e-5 (ver 1.2.1); el texto dice que los puntos medidos caen en zona "sin solución" y el dibujo lo confirma | Insuficiente en claridad (el mensaje del texto no se puede leer de la figura) |

Estilo frente a PRA/AQT: todas las figuras necesitan (i) tamaños de fuente consistentes (6–8 pt), (ii) leyendas fuera del área de datos, (iii) paneles etiquetados sin recuadro negro, (iv) mapas 2D interpolados/rejilla más fina, (v) análisis (línea) sobre datos (marcadores) como en 2(b) y 4(b), y (vi) la Fig. 1 vectorial. Falta una figura con el esquema completo del experimento propuesto (QD + nanocavidad fonónica + láser + detector).

## 5. Veredicto

**Críticas**
1. Reformular el enunciado de la Fig. 6 (λ ≳ 0.011 para 1e-5, o λ ≳ 0.03 para 1e-4; γ ≲ 6.6e-3 en ambos casos) y repetir el cálculo con malla fina; declarar γφ=2γ; marcar claramente extrapolación frente a sin-solución; etiquetar todos los contornos.
2. Corregir la atribución de §1 (parámetros base no son de Bin) y la frase "0.59 a 0.95 en el mismo punto".
3. No llamar "pureza" sin cualificar a una variable oculta; poner la pureza operativa (0.59–0.78 en base) a la par que la de episodios en el resumen y el abstract.
4. Rehacer Fig. 5 (leyendas rotas) y Fig. 4(c) (extrapolación ilegible; añadir P₂ microscópico).
5. Presentar la corrección de la maestra como "confirma y cuantifica" respecto a Bin 2021; no como corrección novedosa; declarar el carácter condicional (J(0)=0, T=0).

**Importantes**
6. Reportar el error máximo de la impureza en la región (18 % aquí) y definir la región con cantidades de la fórmula, no con P exacta.
7. Añadir los P₂ microscópicos (κ=0.003, 0.01) a la tabla R5, etiquetar la magnitud; mencionar que κ_c>κ en κ=0.01 y 0.03.
8. Declarar que 0.9616 es máximo de rejilla; precisar la hipótesis de la cota.
9. Declarar incertidumbre de κ* exacto (0.0089 frente a 0.0093; 0.0061 frente a 0.0063).
10. Separar disipador frente a operador de salida (positivo) en R5; discutir un baño con J(0)≠0 y un baño acústico T≠0 de plataformas reales.

**Menores**
11. Fig. 1 limpiar (b), indicar parámetros de (c); Fig. 2 mover textos; Fig. 3 refinar rejilla y usar rango propio en (c); tamaño tipográfico homogéneo.
12. Verificar la existencia de `tests/run_tests.py`; citar g⁽²⁾, Q con su fuente.

**Sólido:** resonancia con corrección de Stark (R1; error < 0.15 γ_c, verificado por otro agente); flujo cerrado (R2); el artefacto κλ²/γ y su verificación con el baño microscópico en κ=0.003; la fórmula de pureza en el rango ξ<0.5, λ=0.1–0.2 (error en P ≤1.3 % en 40 puntos nuevos); honestidad del dossier sobre la falta de plataforma y las limitaciones (§6–7).
**Débil:** novedad, observabilidad de la pureza titular, la prueba microscópica de baño único con extrapolación, Fig. 6 (tanto numérica como visual), figuras 4(c), 5 y 6, y ausencia de una arquitectura realizable.

**Probabilidad subjetiva de aceptación** (artículo tal como lo plantea el dossier): PRA ≈ 25 % (rechazo probable por novedad y plataforma; podría pasar con revisión mayor si se enfoca como estudio metodológico de consistencia de la maestra con test microscópico); Advanced Quantum Technologies ≈ 45 % (más tolerante con teoría + requisitos y con "proposal-and-corrections"; el nivel de las figuras debe subir).
