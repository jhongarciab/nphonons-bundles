# Dossier: memoria fonónica de doble carril con chequeo de borrado — requisitos y límites

**Estado:** desarrollo completo, verificación independiente hecha (3 rondas), falta escribir. Fecha: 2026-11-02.
**Título de trabajo:** *When does a dual-rail phononic memory with an erasure check beat single-rail storage? A criterion on noise correlation, and requirements from measured devices.*
**Revista objetivo:** *Advanced Quantum Technologies* (primera opción) o *Phys. Rev. Applied*; *Phys. Rev. A* es posible solo planteado como criterio cuantitativo + requisitos, no como propuesta de ventaja.

> **Etiquetas de verificación.**
> **[V-IND]** reimplementado de forma independiente por un agente que no vio mi código, coincide.
> **[V-JC]** contrastado con el modelo Jaynes–Cummings completo.
> **[V-ANA]** demostrado analíticamente y comprobado numéricamente por un agente independiente.
> **[V-DOBLE]** comprobado por dos códigos míos distintos (fórmula cerrada y QuTiP) pero sin agente independiente.
> **[V-LIT]** referencia comprobada contra el resumen de arXiv.
> **[NO-V]** no verificado de forma independiente: no citar como hecho.

---

## 0. Veredicto honesto (leer primero)

El resultado es un **criterio de requisitos y límites**, no una ventaja. Es publicable como tal; no lo es como "memoria mejor que las existentes".

1. **Criterio analítico central [V-ANA] (derivado a ciegas por VD y comprobado con su propio QuTiP en 12 puntos, acuerdo ≤ 10⁻⁵).** Para ruido de frecuencia gaussiano blanco, un doble carril con chequeo de borrado **ideal** supera a un qubit de carril simple {|0⟩,|1⟩} en fidelidad media (contando un fallo heraldado como adivinar al azar, F = ½) **si y solo si**

   **ρ > ½ (1 + Tφ/T₁)**,

   independiente del tiempo de espera T (solo para ruido blanco; para cuasi-estático ρ > ½(1+Tφ²/(T·T₁)), que sí depende de T; con amplitudes desiguales α₁ ≠ α₂: ρ > [Tφ/T₁ + α₁² + α₂² − 1]/(2α₁α₂), y C < 1 incluso con ρ = 1). Con Tφ = 107 µs y T₁ = 534 µs: ρ\* = 0.60. Corolario: **sin desfase (Tφ → ∞) el doble carril nunca gana**, porque (1+p)/2 < (3+p+2√p)/6 para todo p < 1. La conversión de pérdida en borrado no es gratuita en una memoria autónoma.
2. **La correlación medida no alcanza el umbral.** En los datos de Hitchcock et al., Φ = 0.42 ± 0.13 en modos adyacentes, equivalente a ρ = 0.42 (ruido blanco) o ρ = 0.66 (cuasi-estático). A T = 100 µs: F_ef = 0.781 frente a 0.828 del carril simple (blanco), y 0.826 frente a 0.834 (cuasi-estático, empate). Con Φ = 0.90 (hipotético): 0.871 frente a 0.828.
3. **Sí hay una ganancia en fidelidad *condicionada*:** F_c = 0.851 (blanco) y 0.907 (cuasi-estático) frente a 0.828 y 0.834 del carril simple, con probabilidad de paso 0.80 a T = 100 µs. Es una ganancia de +0.02 a +0.07 al precio de un 20 % de intentos descartados. Solo vale si el heraldo se aprovecha (reintentos, redes, corrección de errores).
4. **El chequeo en sí es de buena calidad:** con g = 3 MHz, Tφ = 107 µs y T ≤ 200 µs pierde ≤ 0.012 de fidelidad condicionada y ≈ 0.03 de probabilidad (0.80 frente a 0.83) respecto al techo de detección ideal (hasta 0.019 con Tφ = 10 µs o T = 500 µs; en F_ef la pérdida es 0.02–0.027). Es viable con FSR ≲ 3 MHz, T₁q ≳ 100 µs y g ≳ 3 MHz para acercarse al techo (con g = 1.1 MHz funciona pero es peor: F_c hasta 0.019 y p hasta 0.107 por debajo); es **inviable en HBAR** (FSR = 12.7 MHz, g = 0.27 MHz).
5. **Incertidumbre grande sobre Tφ a un fonón:** Hitchcock mide Tφ = 107 µs con ⟨n⟩ ≈ 10⁴ y Tφ ∝ n^0.26, lo que extrapola a ≈ 10 µs con un fonón **[NO-V]**. En ese caso solo ρ alto da ventaja (ρ\* = 0.51; con Φ = 0.90, F_ef = 0.88 frente a 0.75 del carril simple a T = 25 µs).

**Riesgo principal:** la correlación de ruido es contradictoria en la literatura (§4) y no existe aún un dispositivo de silicio con transmon. El artículo debe vender el **criterio y los requisitos**, no la ventaja.

---

## 1. Arquitectura (la parte concreta)

- **Codificación:** qubit time-bin fonónico en doble carril, un fonón compartido por dos modos longitudinales adyacentes (|10⟩, |01⟩) de un resonador piezoacústico multimodo. *Fig. 0.*
- **Transmon** acoplado a ambos modos con el mismo g (supuesto), a distancia Δ_tm ≫ FSR, con desplazamiento g–e por fonón 2χ_k ≈ 2(g²/Δ_k)·α/(Δ_k+α), α/2π = 185 MHz.
- **Chequeo JPNS:** pulso π gaussiano (duración 4σ) selectivo al sector de un fonón total; luego lectura del transmon (error 1 %). Si no se excita, hubo pérdida y se descarta. Fase determinista por χ₁ ≠ χ₂ compensada por una Z lógica común.
- **Carga/descarga (iSWAP parciales como en Franzen et al.):** **no se simulan**, se suponen ideales.
- **Restricción clave:** el JPNS necesita χ₁ ≈ χ₂ (si no, el desplazamiento revela en qué modo está el fonón), pero χ₁/χ₂ ≈ (Δ+FSR)/Δ. Hace falta Δ_tm ≫ FSR y, a la vez, 2χ suficiente para que el pulso discrimine n = 0 de n = 1.

---

## 2. Resultados con su verificación

### R1. Φ ↔ ρ **[V-ANA]**
Hitchcock et al. (App. J): Φ = 1 − T_φ,I/T_φ,xc con f(t) = −ln|⟨e^{−iφ}⟩|. Para ruido gaussiano con f ∝ tⁿ: **Φ = 1 − (1−ρ)^{1/n}**, es decir ρ = Φ (blanco) o ρ = 1−(1−Φ)² (cuasi-estático). Exacto solo en el caso gaussiano. *Fig. 1.*

### R2. Convención de Tφ **[corregido por VC; confirmado por mí en el texto de Hitchcock]**
Hitchcock define Tφ sobre |⟨a⟩|² (Eq. 5: |⟨a⟩|² = |a₀|² e^{−κt}|⟨e^{−iφ}⟩|²). Por tanto, por modo, |⟨e^{−iφ}⟩| = e^{−t/(2Tφ)}, y la coherencia relativa de dos modos independientes es e^{−t/Tφ}. **Mi primera versión desfasaba el doble**; todo lo de este dossier usa la convención corregida.

### R3. Fase relativa y cota universal **[V-ANA]**
Var(φ₁−φ₂) = α₁² + α₂² − 2ρα₁α₂. Para **cualquier** estadística: |⟨e^{iφ}⟩| ≥ 1 − Var(φ)/2 (varianza centrada); no hay cota superior universal.

### R4. Dato de correlación medido **[digitalización con segunda lectura independiente]**
Fig. 4(e) de Hitchcock et al. digitalizada dos veces por métodos distintos (500 y 800 dpi): máxima diferencia 0.003 en Φ y en las barras (rms 0.002). Φ en modos adyacentes (FSR = 3.29 MHz): **0.424 ± 0.131**; ≈ 0.23 a 40 MHz; 0.11–0.20 a 56–59 MHz. La barra es la desviación estándar de **todas las medidas de pares** (puntos grises); el artículo no separa error de estimación de variabilidad intrínseca. Φ individuales llegan a ≈ 0.9. Corrección: una lectura inicial mía (0.6–0.7) era errónea. Archivo: `phi_hitchcock_fig4e.npy` (abscisa corregida a n×3.29 MHz).

### R5. Chequeo de borrado: modelo, validación **[V-IND] [V-JC]**
- Modelo: transmon (2 niveles) × 2 modos (n ≤ 1), Lindblad con κ = 1/T₁ + (g/Δ)²/T₁q, desfase correlacionado, decoherencia del transmon (T₁q, T₂q), lectura ε = 1 %, fidelidad media sobre 6 estados lógicos maximizada sobre una Z lógica común.
- **VB** reimplementó el modelo a ciegas: coincide con el mío en 28 combinaciones (|ΔF_c|, |Δp| ≤ 10⁻⁴); límites analíticos reproducidos a 10⁻⁸.
- **Error mío corregido:** el desplazamiento por fonón es **2χ**, no χ. Con 2χ el modelo dispersivo reproduce el JC completo en Δ = 20 MHz (|ΔF_c| ≤ 0.005, |Δp| ≤ 0.02) y solo semicuantitativamente en Δ = 6 MHz (≤ 0.03). **Por eso el barrido se restringe a g/Δ ≤ 0.2.**
- JC completo (transmon 3 niveles, modos 3 niveles; truncación 3→4 cambia ≤ 2×10⁻⁴): Δ = 20 MHz, T = 100 µs, ρ = 0.65, T₁q = 100 µs → F_c = 0.81, p = 0.74; sin chequeo 0.67–0.69. Incertidumbre sistemática ≈ 0.01. Nota: estos valores del JC se obtuvieron con la convención antigua de Tφ; sirven para validar el modelo dispersivo, no como cifras del artículo.

### R6. Líneas base y métricas
- Definiciones: f_I(T) = (T/Tφ)ⁿ es el exponente de coherencia relativa de dos modos independientes (n = 1 blanco, n = 2 cuasi-estático) y f₁ = f_I/2 por modo; f_xc = (1−ρ) f_I con ρ = 1−(1−Φ)ⁿ.
- **B0:** carril simple {|0⟩,|1⟩}, sin chequeo. Fidelidad media cerrada F = (3 + p + 2√p·e^{−f_I/2})/6, con p = e^{−T/T₁}. **[V-DOBLE]** (coincide con QuTiP a 5 cifras).
- **B1:** doble carril en dos resonadores separados (ruido independiente), sin chequeo.
- **B2:** doble carril en el mismo resonador (ruido correlacionado), sin chequeo.
- **D:** doble carril + chequeo. **Dh:** heraldado ideal (techo).
- **Métricas:** F_c (condicionada al paso), p_pass y **F_ef = p·F_c + (1−p)/2** (fallo heraldado → adivinar). F_c y F_ef no son comparables con las no heraldadas sin decir cuál se usa; el dossier siempre da ambas.

### R7. Resultados: silicio, g = 3 MHz, FSR = 3.3 MHz, T₁q = 300 µs, Tφ = 107 µs, T₁ = 534 µs *(tabla completa en `results_tables.md`; figuras 3–5)*

| Escenario | T (µs) | F_c | p_pass | F_ef | B0 | B2 | F_h ideal | F_ef ideal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Φ=0.42 blanco (ρ=0.42) | 50 | 0.911 | 0.880 | 0.862 | 0.904 | 0.839 | 0.921 | 0.883 |
| Φ=0.42 blanco | 100 | 0.851 | 0.801 | 0.781 | 0.828 | 0.714 | 0.861 | 0.799 |
| Φ=0.42 blanco | 200 | 0.770 | 0.665 | 0.680 | 0.723 | 0.536 | 0.779 | 0.692 |
| Φ=0.42 cuasi-est. (ρ=0.66) | 100 | 0.907 | 0.801 | 0.826 | 0.834 | 0.759 | 0.915 | 0.844 |
| Φ=0.90 blanco (hipotético) | 100 | 0.963 | 0.801 | 0.871 | 0.828 | 0.805 | 0.970 | 0.890 |
| Φ=0.90 blanco | 200 | 0.934 | 0.665 | 0.789 | 0.723 | 0.649 | 0.943 | 0.805 |

Lecturas robustas: (i) el chequeo cuesta ≤ 0.012 en fidelidad y 0.028 en probabilidad frente al techo; (ii) el rendimiento casi no depende de g para g ≥ 3 MHz y de T₁q para T₁q ≥ 100 µs (con g = 1.1 MHz, F_ef = 0.75–0.76); (iii) FSR = 1 frente a 3.3 MHz es casi irrelevante; (iv) la ganancia respecto a B2 sin heraldar es grande pero **no es la comparación justa** (la justa es B0 con F_ef o con F_c y el coste en tasa).

### R8. HBAR: nada ayuda **[modelo validado]**
FSR = 12.7 MHz, g = 0.273 MHz, T₁ = 82 µs, sin desfase. El desajuste de 2χ entre modos es ≈ 10× (99 y 10 kHz a Δ = 1.5 MHz). Mejor F_c = 0.605 (T = 40 µs) frente a 0.614 sin chequear; el carril simple da 0.864 y **ni el heraldado ideal** supera al carril simple (F_ef ideal 0.807). Los 'sin chequeo' de esta tabla son B2 no heraldada; F_c se compara con ella solo para mostrar que el chequeo no aporta nada. Limitación: se supone el mismo g en ambos modos.

---

## 3. Resultado negativo: el modelo microscópico de TLS (no usar cuantitativamente)

Probé modos con frecuencia perturbada por TLS telegráficos, con corrimiento dispersivo A_jk = g_j² u_k²(x_j)/(ω_k − Ω_j) y perfiles viajero/estacionario.
- **El código es correcto** (VA: 9 dispositivos reproducidos con semillas fijas; función característica exacta confirmada con Monte Carlo a ≤ 0.0013).
- **Error mío de censura** (`first_cross` devolvía el tope de 3000 µs si no había cruce) sesgaba las medias de Φ (0.545 → 0.762 al excluir los topados). Corregido (malla a 30 ms, NaN).
- **Con la corrección el modelo no reproduce los datos:** ningún conjunto de parámetros ajusta a la vez la media de Φ, su dispersión (datos ≈ 0.13, modelo ≈ 0.35–0.48) y Tφ ≈ 107 µs. Los ajustes "buenos" iniciales eran un artefacto de la censura.
- **Consecuencia:** las predicciones previas (ondas viajeras > estacionarias; Φ ↑ con volumen) se **retiran** como cuantitativas. Solo permanece la hipótesis cualitativa ya presente en la literatura.

---

## 4. Literatura y novedad

**Verificadas por mí contra arXiv (título/autores) [V-LIT]:** Hitchcock 2502.16426; Franzen 2610.18778; von Lüpke 2110.00263 y 2303.00730; Lee 2304.13589; Chou 2307.03169; Koottandavida 2311.04423; de Graaf 2406.14621; Wang 2411.00719; Yang 2406.07360; Bozkurt 2412.08006; Garcia Belles 2602.22117; Kang 2505.12937; Tubsrinuan 2208.13410 (PRB 109, L241301); Maksymowych 2501.08289; Fiaschi 2501.12056 (PRL 134, 243604).

**Evidencia sobre correlación entre modos (contradictoria; VC confirmó lo marcado):**
| Trabajo | Qué hace / resultado |
|---|---|
| Hitchcock 2502.16426 | Guía de silicio + niobato de litio **sin qubit**; correlación positiva entre modos cercanos que decae con la desintonía (mi digitalización: Φ ≈ 0.42) |
| Tubsrinuan 2208.13410 | Resonador SAW, 14 modos: correlaciones que disminuyen con la desintonía; además ráfagas anticorrelacionadas |
| Fiaschi 2501.12056 | Cavidad optomecánica de silicio, un solo par de modos separados ≈ 5 MHz: posiciones en frecuencia **no correlacionadas** en el tiempo |
| Franzen 2610.18778 | HBAR: **cota** Γc < 0.7 ms⁻¹ (95 %, unilateral) sobre dephasing colectivo; coherencias consistentes con decoherencia local. No mide Φ |

**Novedad (búsqueda por palabras clave, no concluyente):** no encontré una propuesta de memoria estacionaria de doble carril fonónico con chequeo de borrado dispersivo, ni un criterio ρ\*. Vecinos cercanos: Wang 2411.00719 (dual-rail para fonones SAW itinerantes, QRAM); Kang 2505.12937 (dual-rail vibracional en iones); los dual-rail superconductores con borrado; los estados W de Franzen (que son un qubit de doble carril).

**Datos de literatura reportados por agentes, no verificados por mí [NO-V]:** P₀ de TLS, tasas de conmutación, números de revista (Nat. Phys. 21, 1469; Science 386, 783), T₁q = 1.7 µs en Bozkurt 2412.08006, χ = 147 kHz en von Lüpke 2110.00263 y datos de calibración de chequeos de Chou/Koottandavida/de Graaf. Verificar antes de citar.

---

## 5. Lo que NO se debe afirmar

1. Que el doble carril "protege del ruido común" en las plataformas actuales.
2. Que la arquitectura **mejora** la memoria en fidelidad media (F_ef) frente a un carril simple con la correlación medida.
3. Que el HBAR sirve para esta arquitectura con JPNS.
4. Que el chequeo "supera" al heraldado ideal: es el techo; solo se mide cuánto se acerca.
5. Predicciones de Φ para otras geometrías o volúmenes.
6. Cifras del modelo dispersivo con g/Δ > 0.2 como cuantitativas.
7. Tφ = 107 µs como valor a un fonón (es a ⟨n⟩ ≈ 10⁴; la extrapolación a ≈ 10 µs es [NO-V]).
8. Que Hitchcock tiene un qubit: es un dispositivo **sin transmon**. Todo lo del transmon con silicio usa parámetros supuestos.
9. Cifras del JC completo de VB como resultados finales: usan la convención antigua de Tφ y sirven solo para validar el modelo.

---

## 6. Limitaciones del modelo

- Dispersivo efectivo, validado con JC solo para g/Δ ≲ 0.2.
- Carga y descarga ideales; sin ruido durante los pulsos de escritura; sin reset del transmon ni acumulación de varios chequeos.
- Ruido gaussiano (blanco o cuasi-estático) con relación exacta solo en el caso gaussiano; la cota R3 cubre el caso general en alta coherencia.
- El acoplo de 1.1 MHz de Hitchcock es piezoeléctrico con un resonador de microondas, no con un transmon.
- Mismo g para ambos modos; sin desfase del transmon por fotones térmicos.
- Barrido de diseño grueso: optimizado con objetivo F_ef; para g = 10 MHz el óptimo cae en el borde (σ = 0.5 µs, Δ = 160 MHz); no cambia las conclusiones cualitativas.
- Φ medido tiene la incertidumbre de una digitalización de figura (≈ 0.003) y de la dispersión intrínseca no separada.

---

## 7. Esquema del artículo

**Mensaje:** un criterio sencillo y verificable (ρ > ½(1+Tφ/T₁)) dice cuándo un doble carril con chequeo de borrado vale la pena; las correlaciones medidas están por debajo; el chequeo es viable en resonadores de FSR ≲ 3 MHz con T₁q ≳ 100 µs y no en HBAR; con Φ ≳ 0.8 la ventaja sería clara.

1. **Introducción:** memorias fonónicas, dual-rail con borrado en cavidades, brecha: no hay criterio cuantitativo contra un carril simple con parámetros medidos.
2. **Arquitectura** (Fig. 0).
3. **Modelo y métricas** (R2, R3, R6).
4. **Ruido y correlación medida** (R1, R4; Fig. 1).
5. **Chequeo de borrado y restricción de χ** (R5; validación JC).
6. **Resultados:** criterio ρ\* (Fig. 5), mapa de diseño (Fig. 3–4), HBAR (R8).
7. **Discusión:** contradicción de la literatura sobre correlación; experimento decisivo (espectroscopía cruzada tipo W/GHZ en un resonador de silicio con transmon); uso del heraldo (redes, QEC).
8. **Conclusiones.**

**Figuras listas en `figs/`:** `fig0_arquitectura.png`, `fig1_phi_medido.png`, `fig3_diseno_g.png`, `fig4_vs_T.png`, `fig5_umbral_rho.png`.

**Probabilidad de aceptación (estimación subjetiva del revisor VC, no mía):** con las correcciones hechas y la comparación honesta con el carril simple, AQT ≈ 50 %, PRA ≈ 30 %, PR Applied ≈ 20 %. Tómese como orden de magnitud.

**Qué lo hace flojo para PRA:** efecto modesto, sin experimento propio, dato de correlación contradictorio, dispositivo de silicio sin transmon. **Mitigación:** titular y resumen centrados en el criterio y los requisitos.

---

## 8. Registro de errores encontrados y corregidos

| Error | Origen | Efecto | Corrección |
|---|---|---|---|
| Lectura Φ ≈ 0.6–0.7 de la Fig. 4e | Mía | Optimismo | Digitalización a 500 y 800 dpi: 0.42 |
| χ como desplazamiento por fonón | Mía (halló VB) | p_pass hasta 2.4× en Δ = 6 | 2χ, validado con JC |
| Convención de Tφ (factor 2) | Mía (halló VC) | Desfase ×2; F_c 0.774 → 0.851 | Tφ definido sobre \|⟨a⟩\|² |
| Falta de línea base de carril simple | Mía (halló VC) | Atribuía ventaja inexistente | B0 cerrada, verificada con QuTiP |
| Comparar F_c post-seleccionada con no heraldadas | Mía (halló VC) | Ventaja sobreestimada | F_ef y F_c siempre juntas |
| Escenario "cuasi-estático" como reescalado exponencial | Mía (halló VC) | Coherencia 0.53 frente a 0.75 | f ∝ tⁿ gaussiano |
| Censura a 3000 µs en `first_cross` | Mía (halló VA) | Ajustes TLS espurios | Malla a 30 ms |
| Desajuste de χ probado solo hasta 20 % | Mía | Subestimé la tensión de diseño | χ_k por modo + Z compensada |
| Fase determinista no compensada | Mía | F_c subestimada hasta 0.30 | Z lógica común |
| Mapa con g/Δ hasta 0.35 | Mía | Fuera del dominio validado | g/Δ ≤ 0.2 |
| Afirmación "iguala al conteo ideal" | Mía (VC: casi tautológica) | Engañosa | Reformulada como "cuánto se acerca al techo" |

---

## 9. Reproducibilidad

- Código y datos: `estudio/src/` (`erasure_sim.py`, `design_run2.py`, `breakeven.py`, `make_figs2.py`, `tls_model.py`, `recheck.py`, `fig_arch.py`), `phi_hitchcock_fig4e.npy`, `results_tables.md`, `figs/`.
- Verificación: `estudio/verif/VA_report.md`, `VB_report.md`, `VC_report.md`, con su código en `verif/va/`, `vb/`, `vc/`.
- Literatura: `estudio/lit/lit_report_TLS.md` y `lit_report_plataforma.md`.
- Ejecutar desde `estudio/src/`: `python3 design_run2.py && python3 breakeven.py && python3 make_figs2.py` (≈ 1–2 min). Requisitos: Python 3.13, numpy, scipy, qutip 5.3, matplotlib, pillow.
- Los scripts de `doble_carril_fonon/src/` usan χ en vez de 2χ y la convención antigua de Tφ: **quedan superados por `estudio/`**. `design_run.py`, `hbar_run.py` y `make_figs.py` de `estudio/src/` son versiones intermedias: usar los `*2`.

---

## 10. Pendientes opcionales

- Barrido de diseño más fino en σ y Δ_tm.
- Verificar las referencias marcadas [NO-V].
- Si se quiere reforzar para PRA: añadir la carga/descarga con iSWAP reales y la simulación con ruido 1/f no gaussiano.
