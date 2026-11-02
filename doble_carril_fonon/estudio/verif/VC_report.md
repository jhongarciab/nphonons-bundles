# VC: revisión adversaria del DOSSIER (doble carril fonónico + chequeo de borrado)

Archivos de trabajo en `verif/vc/` (`dig.py`, `cmp.py`, `metrics.py`, `rerun.py`, PDFs/txt de los 4 papers). Sin commit/push; DOSSIER.md no editado.

## 0. Resumen ejecutivo
1. **Error crítico de convención de Tφ (factor 2).** Hitchcock define Tφ sobre |⟨a⟩|² (ajuste B0·exp(−κt−Γt−Δ²t²), Eq. 5; ΓTφ+(ΔTφ)²=1) y Tφ,I con f1+f2=1 (Eq. J10). Para modos iguales y ruido independiente, la coherencia relativa del doble carril es e^{−t/Tφ,I}, con Tφ,I=107 µs. El código (`erasure_sim.py`, gam=2/Tφ; VB lo confirma como e^{−Tγ(1−ρ)}) da e^{−2t/107} con ρ=0: **dephasing 2× demasiado rápido** (equivale a Tφ≈53 µs). Recalculado con Tφ=214 en la convención del código (T=100, g=3, FSR=3.3, T1q=300, Δ=20, σ=0.5): F_c 0.774→**0.855**, B2 sin heraldar 0.646→**0.714**, B2 heraldado 0.779→0.861, p_pass igual 0.781. Cambia el nivel absoluto de todas las tablas (la ganancia relativa G no cambia).
2. **Falta la línea base clave: qubit de riel simple {|0⟩,|1⟩} sin chequeo** (misma memoria, sin transmon). Calculado analíticamente con los parámetros del dossier (media sobre los 6 estados): T=100 µs, F=0.757 (convención del código) o 0.828 (convención corregida), frente a F_c=0.774 (p=0.78) o 0.855. En el promedio de fidelidad el chequeo **no supera** al riel simple; en Q=p(2F−1) el riel simple gana claramente (0.51/0.66 frente a 0.43/0.56). Ver §3.
3. **El escenario "cuasi-estático" (S2) no es cuasi-estático**: el código solo reescala una tasa exponencial (Tφ/(1−Φ)²). La física gaussiana exp(−(t(1−Φ)/Tφ)²) da a T=100 µs una coherencia relativa de 0.75, frente a 0.53 del código (a 25 µs: 0.98 frente a 0.86). Las cifras "cuasi-estático" (F_c 0.838, B2 0.700, "+0.10 B1→B2") no son citables.
4. Un dato citado mal: HBAR "mejor F_c = 0.58" (línea 82); el JSON y la tabla R-C dan 0.599.
5. La digitalización de Fig. 4(e) se **confirma** (error en Φ ≤ 0.003); 0.424±0.131 es correcto. La barra es la desviación estándar de las medidas individuales, no la incertidumbre de la media.

## 1. Auditoría numérica (dossier frente a datos)

| Línea | Afirmación | Comprobación | Estado |
|---|---|---|---|
| 20, 47 | ρ=0.42 / 0.66; G=0.86 / 1.22 | 1−0.58²=0.664; 1/(2·0.58)=0.862; (2·0.336)^−½=1.220 | OK |
| 21, 75, 124 | "−0.004 a −0.014" F_c−F_h | Φ=0.42 y diseño g=3/T1q=300: −0.006…−0.008 (T=25–200) y −0.013 (T=500). −0.004 es Φ=0.9 a T=25; −0.016 es Φ=0.9 a T=500. Sobre toda la rejilla de diseño (360 filas, Φ=0.42) el rango es −0.006…−0.17 | **Rango mezcla escenarios; solo vale para el diseño óptimo elegido** |
| 21, 75 | p=0.78 a 100 µs; ideal 0.83; cuesta ≈5 % | 0.781 y e^{−100/534}=0.829; diferencia 0.048 absoluta (5.8 % relativa) | OK (aclarar "puntos porcentuales") |
| 21, 76 | +0.07 a +0.13 frente a B2 (T=50–100) | Tabla: +0.070 y +0.127. Sobre las 20 configuraciones óptimas: +0.045…+0.070 (T=50), +0.093…+0.127 (T=100) | OK para el diseño g=3/T1q=300; el rango inferior de la rejilla es +0.05/+0.09 |
| 76 | B1→B2 +0.05 (blanco) y +0.10 (cuasi-estático) | 0.646−0.595=0.051; 0.700−0.595=0.105 | Números OK; el escenario cuasi-estático es inválido (§0.3) |
| 77 | F_c=0.759–0.774 (g≥3, T1q≥100); 0.75–0.76 (g=1.1) | g≥3: 0.759–0.774. g=1.1: 0.749–0.761 (0.751–0.761 con T1q≥100) | OK |
| 77 | "FSR 1 frente a 3.3 casi irrelevante" | Δ(F_c)≤0.014 (3/100: 0.773 frente a 0.759), pero con Δ_tm distinto | Aceptable |
| 78 | "28→100 µs: p_pass 0.69→0.77–0.81" | A 28 µs: 0.694–0.764 (g=1.1: 0.69–0.71; g=3: 0.76; g=10: 0.725–0.736). A 100 µs: 0.771–0.812 | Impreciso: 0.69 solo es g=1.1 |
| 78 | "100→300 µs: cambio < 0.02" | g=3, FSR=3.3: 0.812→0.781 (**−0.031**, otro Δ*/σ*); el resto +0.012…+0.017 | **Falso en un caso** (no monotonía del optimizador) |
| 79 | Φ=0.90, F_c=0.938 | 0.938 | OK |
| 82 | HBAR "mejor F_c 0.58 (T=40)" | JSON/tabla: 0.599 (0.5991–0.5995); F_unc 0.614 | **Discrepancia** |
| 60 | JC F_c=0.81, p=0.74, σ*=3; sin chequeo 0.67–0.69; +0.12 | VB: 0.812/0.740; 0.694 → +0.118 | OK |
| 53 | 0.424±0.131; 0.23 a 40 MHz; 0.11–0.20 a 56–59 | npy: 0.4240/0.1313; 0.2256 (39.7); 0.1138 (56.2) y 0.2031 (59.3) | OK |
| 53 | Δω primer bin = 3.4 MHz | Mi calibración con las marcas del eje: 3.29 MHz (el npy está desplazado +0.11…+0.17 MHz) | Menor |
| 53 | Espaciado 3.29 ≈ FSR | Paper: FSR=(3.3±0.2) MHz medido (3.7 MHz de diseño) | OK |
| 44 | Φ=1−(1−ρ)^{1/n} | Compatible con (J9)–(J11) | OK |
| tablas | R-A, R-B | Reproducidos desde `design_eval_si.json` y `design_opt.json`; `rerun.py` reproduce 0.774/0.781/0.646/0.779 | OK |

Observaciones sobre el barrido (R-B, `design_opt.json`): la rejilla es muy gruesa (Δ∈{4,6,10,20,40,80}, σ∈{0.5,0.7,1.2,2,3.5,6}). El óptimo cae **en el borde** en 14 de 24 filas (σ*=0.5, el mínimo) y Δ*=80 (el máximo) para g=10. Es cierto que F_c ya satura en F_h, pero p_pass y la afirmación "casi no depende de g" quedan sin explorar por debajo de σ=0.5 µs. El dossier (§10) solo lo admite para Δ pequeño.

## 2. Segunda digitalización independiente de Fig. 4(e)
Método propio (`dig.py`): PDF a 800 dpi, recorte de la página 5, detección de píxeles rojos (R>220, G,B<120), apertura morfológica con disco para aislar los puntos, y extremos de los topes de las barras como filas rojas horizontales anchas. Calibración con las **marcas del eje**: y=0, 0.5, 1 en las filas 743, 394.5, 46 (la rejilla coincide con el punto rojo grande en Φ=1, Δω=0); x=0, 20, 40, 60 en las columnas 249.5, 593.0, 936.5, 1280.5. Se detectaron 18 puntos más el punto en (0,1).

| Magnitud | máx. \|diferencia\| | rms | sesgo medio |
|---|---:|---:|---:|
| Φ media | 0.0027 | 0.0018 | +0.0018 (el autor es 0.002 menor) |
| +err | 0.0032 | 0.0018 | −0.0015 |
| −err | 0.0019 | 0.0010 | +0.0007 |
| Δω (MHz) | 0.165 | 0.123 | **−0.121** (el npy está desplazado +0.12) |

Primer bin: yo obtengo Φ=0.4247, +0.1306/−0.1306 en Δω=3.29 MHz. **0.424±0.131 es correcto**; el único error es la abscisa del npy (3.44 en lugar de 3.29), irrelevante para las conclusiones.

**Texto del paper sobre la barra (p. 5–6, Fig. 4(e))**: "averaging the cross-correlation strengths over detuning [red points], error bars indicate the standard deviation of all mode pair measurements [gray points] for that detuning". Fig. 4(d): cada Φ es "averaged over experimental occurrence and nine microwave mode frequencies" y se usan 19 modos. Consecuencias:
- La barra es una **desviación estándar poblacional** (±0.13 es la dispersión entre pares/mediciones, no el error de la media). La incertidumbre de la media es mucho menor. El dossier escribe "0.424 ± 0.131" como si fuera la incertidumbre del parámetro.
- Los puntos grises son medidas individuales de pares de modos (la nube del primer bin tiene cientos de puntos, más que los 18 pares adyacentes, así que incluyen frecuencia del modo de microondas y repeticiones). El texto no separa error de estimación de variabilidad intrínseca entre pares. Tu caveat de R4 ("incluye error de estimación") es plausible pero **no está respaldado por el paper**; el paper solo dice "large amount of variance". Reformular como "el paper no distingue".
- Dos cautelas que el dossier no menciona: (i) ⟨n⟩≈10⁴ y T1,in∝n^0.12, Tφ∝n^0.26: por la ley de potencias del propio paper, a 1 fonón Tφ sería ≈10 µs (107·10^(−4·0.26)) y T1≈180 µs; extrapolar 107/534 a un fonón es optimista (la línea 127 lo menciona, pero R6 usa 107 igualmente). (ii) la cavidad de microondas compartida (ωc≈2.85 GHz) puede inducir correlación espuria; el paper argumenta que es pequeña a 50 MHz de desintonía.

## 3. Literatura (§4)

| Paper | Dossier dice | Comprobado | Veredicto |
|---|---|---|---|
| Hitchcock 2502.16426 | T1,in=534 µs, Tφ=107 µs (un modo, ⟨n⟩≈10⁴); abstract Tφ≈60 µs; sin transmon | Confirmado el texto de la Fig. 3(a); abstract "T1,in≈500 µs, Tφ≈60 µs". Es un dispositivo con ringdown clásico y cavidad de microondas, sin qubit | OK; **añadir la convención de Tφ (|⟨a⟩|²) y la dependencia en n** |
| Fiaschi 2501.12056 (PRL 134, 243604) | "guía PnC, modos a ~5 MHz, adyacentes NO correlacionados" | El abstract dice: dos modos adyacentes (~5 MHz) "not correlated in time". Es una **cavidad optomecánica de cristal de silicio** (OMC), un único par de modos, medido ópticamente (difusión espectral) | Resultado OK; **"guía PnC" es incorrecto**: OMC, un par |
| Tubsrinuan 2208.13410 (PRB 109, L241301) | SAW, 14 modos, correlaciones que decaen con la desintonía; ráfagas anticorrelacionadas | Abstract: 14 modos SAW, 7 horas, "significant degrees of correlations that diminish with increased detuning", "slow, anti-correlated" bursts (tipo rayos cósmicos) | OK; añadir escala de horas (no es la de ringdown) |
| Franzen 2610.18778 | Γc<0.7 ms⁻¹ (95 %), W "decoherence-free under collective dephasing"; HBAR 12.7 MHz; W=doble carril | Γ̄c=−0.1(4) ms⁻¹, "Γc<0.7 ms⁻¹ at the 95 % confidence level (one-sided)" (N=2,3). "W states are decoherence-free under collective noise of any spectrum" (S-D). FSR=12.7 MHz. T2*≃2T1 (sin dephasing puro apreciable). Colectivo ≤Γc/2<0.3 ms⁻¹ | OK. Matiz: es una cota sobre ruido **puramente colectivo** en un HBAR sin dephasing puro, no una medida de Φ comparable con Hitchcock; la tabla la pone como "contradictoria", y es más bien "otro régimen". |

Referencias de revista de Fiaschi y Tubsrinuan: confirmadas con la API de arXiv (journal_ref). La tabla de §4 no atribuye a ningún paper algo que este no diga; los matices son la OMC de Fiaschi y las escalas temporales distintas.

## 4. Revisión lógica y de sobre-afirmación

### 4.1 Comparación F_c frente a B2: ¿es justa?
No del todo. F_c está **post-seleccionada**; B2 sin heraldar no. El "+0.07 a +0.13" mide en gran parte el efecto de descartar el 22 % de los intentos. Es la ganancia esperada de cualquier detección de pérdida, y la línea 21 la presenta como mejora de la memoria. Métricas combinadas con los datos del diseño (g=3, FSR=3.3, T1q=300, Φ=0.42 blanco; convención del código; la corregida va entre paréntesis para T=100):

| T (µs) | F_c | p | Q=p(2F_c−1) | 2F_unc−1 (B2) | F_ef=pF_c+(1−p)/2 | F_unc B2 | p·F_e* | F_e* B2 | Riel simple F | Riel simple 2F−1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 0.913 | 0.903 | 0.745 | 0.757 | 0.873 | 0.879 | 0.78 | 0.82 | 0.917 | 0.83 |
| 50 | 0.854 | 0.860 | 0.608 | 0.567 | 0.804 | 0.784 | 0.68 | 0.68 | 0.851 | 0.70 |
| 100 | 0.774 (0.855) | 0.781 | 0.427 (0.555) | 0.293 (0.428) | 0.714 (0.777) | 0.646 (0.714) | 0.52 | 0.47 | 0.757 (0.828) | 0.51 (0.66) |
| 200 | 0.699 | 0.645 | 0.257 | −0.031 | 0.628 | 0.485 | 0.34 | 0.23 | 0.657 | 0.31 |

(F_e* = (3F−1)/2, la fidelidad de entrelazamiento, ya que el octaedro de 6 estados es un 3-diseño.) Lectura:
- Con Q o con "fallo marcado = adivinar", el chequeo **sí gana** a B2 sin heraldar a T≥50 µs; a 25 µs empata o pierde (F_ef 0.873 frente a 0.879). La ventaja real es menor que +0.13 y no existe a T pequeño. Con el producto simple p·F_c (0.60 a 100 µs) el chequeo pierde frente a B2 (0.646).
- El **riel simple** (|0⟩/|1⟩ Fock, sin transmon, sin chequeo) iguala o supera en fidelidad media a todo lo anterior. Es la línea base que un revisor exigirá. El matiz honesto: el riel simple no tiene señal de borrado, y el dual-rail se justifica por el borrado convertible y por el esquema time-bin (que ya son dos modos). Pero entonces el artículo debe decir que **en promedio de fidelidad la ganancia sobre el riel simple es negativa o nula con las cifras de silicio**, y que el valor del dual-rail está en un régimen de otro tipo (código con corrección, T1≫Tφ, Φ alto).
- Para Φ=0.90 el dual-rail relativo sí mejora; ahí hay que comparar con el riel simple (no calculado aquí).

### 4.2 "El chequeo iguala al conteo ideal"
Es en gran parte tautológico: F_h (proyección a n=1) es el **techo** de cualquier esquema que solo detecte pérdida; que F_c esté a 0.006 de él valida que el pulso JPNS selectivo no añade error, pero no es un resultado nuevo. Además F_h≡F_c solo en el diseño optimizado (en la rejilla completa F_c−F_h llega a −0.2). El efecto real del chequeo es p_pass: 0.78 frente a 0.83 ideal. Tampoco se explora qué errores introduce el propio transmon sobre el qubit almacenado (dephasing por fotones térmicos, ya listado como limitación).

### 4.3 Sesgos del barrido
- El optimizador maximiza Q=p(2F−1) a T=100 µs y Φ=0.42 blanco y se reevalúa a otros T/Φ (in-sample). Favorece pasar el chequeo en p alto (σ pequeño, Δ grande), y el barrido se hace con el modelo que ya reproduce F_h.
- La rejilla es gruesa y el óptimo cae en el borde (ver §1). La no monotonía en T1q (100→300 µs: p 0.812→0.781) es artefacto de la rejilla.
- La Z lógica se optimiza sobre los mismos 6 estados de prueba (129 puntos); sesgo <10⁻⁴ según VB, aceptable.
- La conclusión "g=1.1 MHz basta" depende de usar g igual en ambos modos (§4.5).

### 4.4 Error del modelo de dephasing
Ver §0.1 y §0.3. Además el modelo es Markoviano; el ajuste de Hitchcock es mezcla exponencial+gaussiana (ΓTφ+(ΔTφ)²=1), y el peso gaussiano es desconocido: una parte de la decoherencia de los modos casi estáticos es en realidad reversible por eco/dynamical decoupling, y el compensador Z podría absorber derivas lentas. El dossier no lo discute.

### 4.5 Errores físicos o suposiciones frágiles en la arquitectura
1. **g igual para ambos modos.** En un resonador estacionario, g_k∝ solapamiento del modo con el acoplador; para modos adyacentes difiere O(1) y puede anularse en nodos. χ1≈χ2 exige g_1²/Δ_1≈g_2²/Δ_2; con g distintas la condición se puede imponer por diseño, y eso **invalida la afirmación R7** "la igualdad de χ excluye a los HBAR": con g_k ajustadas (o transmon colocado en el punto donde se compensan) el argumento cambia. R7 solo demuestra que no sirve con g iguales y el JPNS usado.
2. **Dephasing del transmon sobre fonones, fotones térmicos, Purcell**: el modelo incluye Purcell, pero no el desfase del modo por el qubit ni el efecto TLS cruzado transmon–superficie de silicio.
3. **Parámetros de transmon en silicio**: no existe un transmon acoplado a la guía de Hitchcock; g=1.1 MHz es transductor-cavidad. La conclusión "T1q≥100 µs" es una exigencia, no un dato.
4. **Ruido del propio chequeo en el sector n=1**: tras pasar, el transmon queda excitado y hay que resetearlo; el tiempo de reset y el error de lectura (ε=1 %) no se acumulan en p/F.
5. **Alpha positiva/negativa**: VB señaló α/(Δ+α) para modos sobre el transmon; el código usa la misma fórmula para ambos modos. Efecto ≤0.01 (VB), aceptable.
6. **Correlación medida a ⟨n⟩≈10⁴** (ver §2): Φ a un fonón es desconocida; Fiaschi (un fonón-escala OMC) apunta a 0.

### 4.6 Sobre-afirmaciones concretas en §0, §5, §7
- §0.1: "ρ=0.42 o 0.66, G=0.86–1.22": correcto como cálculo, pero la rama 1.22 usa el escenario inválido (§0.3).
- §0.2: "reproduce la fidelidad de un conteo ideal": es el techo, ver §4.2. "mejora +0.07 a +0.13 la fidelidad": es condicionada; hay que decir "frente a B2 sin heraldar y sin riel simple".
- §0.4: "el techo lo pone el desfase": con el error de factor 2, el techo es menos severo (F_c 0.855 a 100 µs).
- §5.3 ya corrige parte de esto ("reduce p ≈5 %"), pero §7 (mensaje) repite "iguala dentro de 0.014" mezclando Φ.
- §3 y R4: la nube gris y la barra se interpretan con más certeza de la que da el paper.
- §0 llama "robusto" al chequeo; la robustez es respecto a su implementación (VB), no a los parámetros físicos (Tφ, Φ).

## 5. Veredicto

### (a) Correcciones priorizadas
**Críticas**
1. Corregir la convención de Tφ (Hitchcock: |⟨a⟩|²; relativa independiente e^{−t/Tφ,I}); rehacer R6/R7 y todas las tablas (F_c 0.855 frente a 0.774, etc.) o justificar explícitamente usar Tφ,eff=53 µs como "pesimista".
2. Añadir la línea base de riel simple (sin transmon) y las métricas combinadas (Q, F_ef, entrelazamiento×p); reescribir el mensaje: el valor del chequeo no es "+0.13 de fidelidad".
3. Rehacer o eliminar el escenario "cuasi-estático" (implementar decaimiento gaussiano o llamarlo "exponencial lento").
4. Corregir R7 (0.58→0.599) y matizar que la inviabilidad en HBAR vale solo para g iguales y este JPNS.

**Importantes**
5. Barras de error: son desviaciones estándar poblacionales; no escribir "±0.13" como incertidumbre; retirar la afirmación de que incluye error de estimación o atribuirla como suposición.
6. Discutir dependencia en n (Tφ∝n^0.26, T1∝n^0.12): a un fonón Tφ≈10 µs extrapolado; barrido de sensibilidad en Tφ∈{10,30,60,107}.
7. Ampliar la rejilla (σ<0.5 µs, Δ>80), refinar el óptimo, y reportar rangos de F_c−F_h solo para el diseño elegido.
8. Corregir línea 78 (0.69→0.725–0.764 para g≥3; "<0.02" falso).
9. Fiaschi es una OMC de silicio (un par), no una guía; Franzen es cota colectiva en régimen sin dephasing, no Φ.
10. Incluir g_k distinta y mismatch de g en el barrido.

**Menores**
11. Δω del primer bin: 3.29 MHz (npy desplazado +0.12 MHz).
12. Aclarar "5 %" (puntos porcentuales).
13. Reset del transmon y coste temporal del chequeo; clarificar que la verificación VB confirma implementación, no los parámetros.
14. §9 menciona `tls_model.py` que no figura en el listado de `src/`.

### (b) Resultados sólidos
- R1–R3 (relación Φ–ρ, ley de ganancia, cota universal) analíticos y coherentes con el texto de Hitchcock.
- El código del chequeo coincide con una reimplementación independiente (VB) y con JC completo en g/Δ≤0.2; la regla de 2χ.
- Φ(Δω) digitalizado correcto (esta verificación); Φ≈0.42 en el primer bin con σ=0.13 dispersión.
- Reproducibilidad de tablas desde JSON; F_c≈F_h en el diseño óptimo (como validación del pulso).
- La conclusión negativa del modelo TLS y la honestidad sobre Φ≈0.42 y la contradicción en la literatura.

### (c) Probabilidad subjetiva de aceptación (si se escribe como propone el dossier, sin las correcciones 1–4)
| Revista | Probabilidad | Razón |
|---|---:|---|
| Advanced Quantum Technologies | ~35 % | Tolera estudios de requisitos; riesgo: modelo puramente teórico, sin riel simple |
| Phys. Rev. Applied | ~15 % | Exige efecto aplicable o datos; el efecto es modesto y la plataforma sin transmon |
| Phys. Rev. A | ~15–20 % | Un revisor pedirá el riel simple, métricas justas, Gaussiana en el cuasi-estático y la convención de Tφ; "iguala al conteo ideal" se leerá como trivial |

Con las correcciones 1–4 y una comparación honesta frente al riel simple (y, idealmente, un régimen donde el dual-rail con chequeo gane, p. ej. Φ alto o T1≪Tφ), estimaría AQT ~50 %, PRA ~30 %, PR Applied ~20 %.
