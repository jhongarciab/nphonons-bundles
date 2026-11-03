# Ruta P — Fuente de pares de fonones con transmon multinivel en cristal fonónico (simulación completa dependiente del tiempo)

Código: `codigo/` · Datos: `datos/` (npz/pkl/json) · Figuras: `datos/figuras_P.png`, `datos/figuras_P.npz`. Sin commit ni push. El PDF está en /tmp/rutaP_pdf (fuera del repo).

## 0. Resumen (cifras clave, todas con detección ideal en la guía)

| Escenario (caso B: qubit estacionado durante los pulsos, κ_m=10.8 MHz, ventana T_w=0.5 µs) | P(0) | P(1) | **P(2)** | P(≥3) | pureza P2/(P1+P2+P≥3) | flujo, reset activo 100 ns | flujo, reset pasivo 5·T1 |
|---|---|---|---|---|---|---|---|
| Ideal (|f,0⟩ perfecto, sin decoherencia) | 0 | 0.041 | 0.959 (0.991 a T_w=1; 1.000 a 3 µs) | 0 | 1.00 | 1.5 MHz | — |
| Cadena de bloques previa (E), solo T1=1.14 µs, T_w→∞, κ=8.3 | | 0.073 | 0.918 | | | | |
| **Publicado + supuestos** (T1=1.14, Tφ=0.30 µs, n_th=0.03, p_e=5 %, κ_int=0.37 MHz) | 0.100 | 0.156 | **0.718** | 0.026 | 0.80 | **1.1 MHz** | **0.115 MHz** |
| Publicado, κ_m=20 MHz (óptimo en el rango pedido) | 0.089 | 0.143 | 0.742 | 0.027 | 0.81 | 1.14 MHz | 0.119 MHz |
| Publicado, caso A (g encendido durante los pulsos), κ*=5.8 MHz | 0.174 | 0.191 | 0.609 | 0.026 | 0.74 | 0.94 MHz | 0.098 MHz |
| **Mejoras plausibles** (T1=10, Tφ=3 µs, n_th=0.005, p_e=1 %, κ_int=0.05 MHz) | 0.016 | 0.058 | **0.926** (0.954 a T_w=1) | 0.001 | 0.94 | 1.43 MHz | 0.018 MHz |
| Mejoras + T1=50 µs, Tφ=10 µs | 0.011 | 0.051 | 0.937 (0.967 a T_w=1) | 0.001 | 0.95 | 1.45 MHz | 0.004 MHz |

Veredicto corto: con los parámetros publicados, y **suponiendo un κ_m≈10 MHz hacia una guía que el dispositivo medido NO tiene** (su κ_m medido es 0.37 MHz), P(2)≈0.72±0.05, flujo ≈1 MHz solo con reset activo; el 0.918 del explorador previo es optimista en ≈0.2 absoluto. Con mejoras (T1≳10 µs) P(2)≈0.93–0.95. Detalles e incertidumbres en §6.

## 1. Parámetros del PDF arXiv:1902.04681 (Arrangoiz-Arriola et al., “Resolving the energy levels of a nanomechanical oscillator”, Nature 2019)

Páginas = páginas del PDF de arXiv (17 págs.).

| Parámetro | Valor medido | Dónde | Comentario / corrección |
|---|---|---|---|
| ω_ge^max/2π (sweet spot) | 2.417 GHz | p.3, Tabla I (p.7) | transmon sintonizable por flujo |
| Modos mecánicos ω_m/2π | 2.405, **2.257**, 2.153, 2.065, 2.002 GHz | Tabla I (p.7) | “ω_m≈2.3 GHz” es redondeo del modo 2 (2.257 GHz) |
| g_i/2π | 15.7, **15.2**, ~14, 14.2, ~13 MHz | Tabla I, Fig. 3b (p.3,7) | **g=15.2 MHz corresponde al modo de 2.257 GHz**; 15.7 al de 2.405 GHz |
| α/2π | 138 MHz (el paper define α=ω_ge−ω_ef>0; aquí α=−138 MHz) | p.3 y p.7 | medida por espectroscopía de dos tonos |
| T1 qubit | **1.14 µs** a ω_ge=2.301 GHz; rango 1.0–1.4 µs | p.3, p.10, Tabla I | solo se mide T1 de |e⟩; el decaimiento de |f⟩ NO está medido |
| Ancho total qubit γ/2π | ≈600 kHz | p.3, p.7, p.10 | γ=1/2T1+1/Tφ → **Tφ≈0.30 µs derivado por mí** (usa 1/2T1=0.44 µs⁻¹; γ=3.77 µs⁻¹). Es cota pesimista: γ incluye deriva lenta de flujo (la línea aparente es 2.8 MHz sin corregir y 1.1 MHz tras corregir deriva, p.10–11). **T2*/Ramsey no se mide** |
| κ_m/2π del modo | **370 kHz** (T_m=430 ns), modo de 2.405 GHz | p.3, p.10, Tabla I | medido en otro modo (2.405) que el de g=15.2 (2.257); sin guía acoplada |
| χ/2π | −1.56 MHz (2χ≈3 MHz), Δ/2π=−88 MHz | Tabla I, p.3-4 | |
| Resonador de lectura | ω_r/2π=3.026 GHz, κ_r/2π=1.3 MHz; pulso de lectura cuadrado 2 µs | Tabla I, p.10 | lectura dispersiva; **fidelidad de lectura no dada** |
| Pulsos de control | gaussianos σ_t=50 ns (calibración Rabi A1=2π·93.9 MHz/V, A3≈2A1); pulso de fonones coseno τ_mech=175 ns; espectroscopía 1.5 µs, Ω0≈700 kHz | p.9-10 | **tiempos de π no dados** |
| Temperatura | etapas del criostato 7 mK y 100 mK (Fig. de cableado) | p.8-9 | **ocupación térmica del modo/qubit NO medida**; n_th(2.257 GHz,T): 7 mK→2e-7; 10→2e-5; 15→7e-4; 20→4.5e-3; 30→0.028; 40→0.071; 50→0.13; 70 mK→0.27 |
| Tasa de repetición / tiempo de ciclo | no dados (promedios >1 h, retorno IQ cada 20 s) | p.11 | |
| Pulso físico | modo de simulación del paper: ME con QuTiP, γ1, γφ, κ (ec. 8, p.13) | p.13 | mismas decoherencias que uso aquí |

**Parámetros que NO da el paper (supuestos míos, marcados como tales):** T2*/Tφ real, ocupación térmica, p_e residual del qubit, T1 de |f⟩ y |3⟩, κ_int adicional con guía, fidelidad de lectura, tasa de repetición, duración de π.

## 2. Modelo y método

* Hamiltoniano (marco rotante a ω del modo; RWA en g): H = Σ_k E_k|k⟩⟨k| + g(s†a+s a†) + control, con E_k=kΔq+α k(k−1)/2, s=Σ√(k+1)|k⟩⟨k+1|, **transmon 4 niveles (|0..3⟩, 5 para N=3) × modo N=6** (truncado a n+k≤4; verificado frente al espacio completo 4×6 y 4×8: diferencia ≤2·10⁻⁵ en P(m)). **RWA verificada**: el modelo de Rabi completo (con N=14, g/ω=0.67 %) desplaza la resonancia |f,0⟩↔|g,2⟩ solo +4.9 kHz y el gap en 0.3 kHz (`datos/dressed_bs.json`).
* Resonancia |f,0⟩↔|g,2⟩ numérica: **Δq=ω_q−ω=+69.000 MHz** (el corrimiento Stark/Lamb de |f,0⟩ y |g,2⟩ se cancela a este orden), **V=5.74 MHz** (la fórmula 4g²/|α|=6.70 MHz sobrestima 14 %). El qubit queda por encima del modo (ω_q≈2.326 GHz para el modo de 2.257 GHz; alcanzable, <2.417 GHz).
* Control: 2 pulsos π gaussianos consecutivos (g–e a ω_q+δ1, e–f a ω_q+α+δ2), σ=T_p/5, T_p=24 ns cada uno (Ω_pico≈2π·43 MHz, ≈0.45 V con la A1 del paper; extrapolación mía), amplitudes y frecuencias calibradas numéricamente con g incluido (Nelder-Mead). Dos casos: **A** = g siempre encendido (qubit en ω_q); **B** = qubit estacionado lejos del modo (g_eff≈0) durante los pulsos y llevado a ω_q con un pulso de flujo (supuesto, no demostrado en el dispositivo).
* Disipación: salto contado √(κ_m(1+n_g))·a (clic en la guía), pérdida intrínseca no detectada √κ_int·a, calentamiento térmico √(κ n)a†, T1 en escalera (nivel k decae a kγ_q; f→e a 2γ_q), desfase √(2/Tφ)·n̂ (coherencia g–e decae a 1/Tφ, f–g a 4/Tφ), estado inicial térmico del modo (n_th) y del qubit (p_e en |e⟩).
* Conteo de clics **exacto** (no MC): función generadora con campo de conteo λ (FFT en λ), cadena de bloques RK4 durante los pulsos (dt=0.25 ns) y `expm` del Liouvilliano inclinado en la fase de emisión. Ventana de emisión T_w empieza tras los pulsos (los clics durante los pulsos se descartan; en el caso A son 0.2 clics/ciclo).
* **Validación**: (i) reproduce la cadena de bloques de E (P2=0.91749 vs 0.91750 de `transmon_f.pares`); (ii) frente a QuTiP `mesolve` con pulsos dependientes del tiempo (modelo 4×6 completo): clics medios 2.43553 vs 2.43545 (caso B) y 2.23748 vs 2.23746 (caso A); (iii) frente a QuTiP `mcsolve` (1600 trayectorias, escenario publicado, B): P(0,1,2,≥3)=0.090, 0.165, 0.716, 0.029 (±0.007, 0.009, 0.011, 0.004) frente a exacto 0.100, 0.156, 0.718, 0.026.
* **Disipador local frente a vestido**: comparé √κ·a (local) con saltos vestidos secularizados (36 canales) desde |f,0⟩ (T1=∞): diferencias en P2 ≤7·10⁻⁴ para κ_m∈[0.5,20] MHz (p. ej. κ=4 MHz: 0.99552 local vs 0.99606 vestido). Para g/ω=0.7 %, bano plano y κ_m≲V, la diferencia es irrelevante; el DC es irrelevante aquí. Se usa el local.

## 3. Resultados

### 3.1 Preparación |f⟩ y estados tras los pulsos (T_p=24 ns)
* Caso B, ideal: población de |f⟩=0.9999 (error de calibración 5·10⁻⁵, fuga a |3⟩ 4·10⁻⁵). **Los pulsos gaussianos sin DRAG bastan a 24 ns** (σ=4.8 ns, ancho espectral 33 MHz ≪ α=138 MHz).
* Con publicado (T1, Tφ, p_e=5 %): P(f)=0.892 (resto: 5 % en |e⟩→|g⟩ por p_e, ≈3 % decaimiento durante los 48 ns, 0.4 % desfase). Caso A: P(f)=0.64 y la conversión |f,0⟩→|g,2⟩ ya empieza durante el segundo pulso (la población g,2 cuenta como “g” del qubit).
* Duración del π (publicado, B, κ=8.3 MHz, T_w=0.5; `datos/pulsos_tp_drag.npy`): 10 ns → fuga a |3⟩ 6.0 %, P(f)=0.825, P2=0.674; 16 ns → fuga 0.3 %, 0.906, 0.711; 24 ns → 0.1 %, 0.892, 0.701; 40 ns → 0.1 %, 0.860, 0.676 (más decaimiento). DRAG (coeficiente 1, sin optimizar signo/escala) no mejoró: no lo recomiendo sin calibrarlo. Óptimo ≈16–24 ns.

### 3.2 P(2) óptimo sobre κ_m y desintonía
Referencia estática (|f,0⟩ ideal, solo T1 en escalera; `datos/mapa_estatico_N2.npz`): T1=1.14 µs → P2 máx 0.8956 (T_w=0.5 µs, κ=7.95 MHz, desintonía −1 MHz) y 0.9183 (T_w=3 µs, coincide con E: 0.918); T1=10 µs → 0.9685 (T_w=0.5, borde κ=20) y 0.989 (T_w=3, coincide con E).

Modelo completo, P(2) a T_w=0.5 µs (`datos/barrido_completo.npz`; κ_m en MHz):

| κ_m/2π | 0.5 | 1.26 | 2.3 | 4.3 | 7.95 | 10.8 | 20 | 50 | 80 |
|---|---|---|---|---|---|---|---|---|---|
| publicado, B | 0.171 | 0.404 | 0.542 | 0.638 | 0.698 | 0.716 | **0.738** | 0.771 | 0.786 |
| mejorado, B | 0.365 | 0.744 | 0.855 | 0.891 | 0.913 | 0.922 | **0.940** | 0.967 | 0.971 |
| publicado, A | 0.174 | 0.402 | 0.527 | 0.596 | 0.608 | 0.594 | 0.537 | | |
| mejorado, A | 0.377 | 0.746 | 0.832 | 0.829 | 0.789 | 0.756 | 0.674 | | |

* **El óptimo en κ_m no es 8.3 MHz con decoherencia real**: en el caso B sube monótonamente con κ_m (el óptimo está en el borde del rango pedido, 20 MHz) porque (i) la pérdida intrínseca κ_int cuesta ≈2κ_int/κ_m (6.8 % a 10.8 MHz, 1.2 % a 50 MHz) y (ii) el desfase hace la conversión parcialmente incoherente. En el caso A el óptimo está en κ_m≈3–6 MHz (a κ_m mayor el estado preparado, no puro, se degrada). Un κ_m de 20–80 MHz sería una guía muy fuertemente acoplada (Q≈30–100 del defecto): no demostrado.
* Desintonía (respecto a la resonancia numérica, a κ*): P2 publicado B es plano: 0.655 (−12 MHz), 0.713 (−4), 0.718 (−2), 0.716 (0), 0.688 (+4), 0.634 (+8), 0.572 (+12); el óptimo está en −2…−4 MHz (a κ=20: 0.7415 en −4 MHz). El promedio cuasi-estático gaussiano de la frecuencia del qubit con σ=0.5/1/2/3 MHz da 0.716/0.715/0.712/0.707 (publicado) y 0.922/0.921/0.919/0.915 (mejorado): **la fuente tolera deriva de flujo de varios MHz** (la sensibilidad del paper, ~1 MHz tras corrección, no limita).
* Ventana: P2(T_w)=0.683 (0.2 µs), 0.718 (0.5), 0.714 (1.0) para publicado B: el máximo está en 0.5–0.6 µs porque después el calentamiento térmico añade un tercer clic.

### 3.3 Flujo de pares real, P(2)/T_ciclo (T_ciclo = 48 ns pulsos + T_w + reset)

| Escenario | T_w (µs) | P(2) | activo (100 ns) | pasivo (5·T1) |
|---|---|---|---|---|
| Publicado B | 0.2 / 0.5 / 1.0 | 0.683 / 0.718 / 0.714 | 1.96 / 1.11 / 0.62 MHz | 0.115 MHz (T_w=0.5; 5·T1=5.7 µs) |
| Publicado A | 0.5 | 0.609 | 0.94 MHz | 0.098 MHz |
| Mejorado B (T1=10) | 0.2 / 0.5 / 1.0 | 0.867 / 0.926 / 0.954 | 2.49 / 1.43 / 0.83 MHz | 0.018 MHz (5·T1=50 µs) |
| Mejorado B, T1=50 | 0.5 | 0.937 | 1.45 MHz | 0.004 MHz |

La estimación previa de “1–2 MHz” solo es correcta con reset activo; con reset pasivo es **10–300 veces menor** (0.1 MHz publicado). Suponen eficiencia de detección 1 en la guía y que existe un reset activo de 100 ns (no demostrado en este dispositivo); con eficiencia η el flujo de pares detectados escala como η².

## 4. Presupuesto de error (caso B, κ_m=10.8 MHz, desintonía −2 MHz, T_w=0.5 µs)

### 4.1 Cascada acumulativa (publicado) y mejorado

| Paso | P(2) pub. | pérdida | P(2) mejor. | pérdida |
|---|---|---|---|---|
| Ideal |f,0⟩, solo κ_m, ventana 0.5 µs | 0.9591 | 4.1 % (ventana finita; **0.9 % a 1 µs y <10⁻⁴ a 3 µs**) | 0.9591 | 4.1 % |
| Fuga virtual |e,1⟩ / Zeno (T_w→∞, T1=∞) | ≈1·10⁻⁵ a κ≥8 MHz (6.6 % a κ=0.5 MHz) | | | |
| + pulsos reales (caso B) | 0.9591 | 0.0 % | 0.9591 | 0.0 % (caso A: −11 %: 0.943→0.848) |
| + T1 en escalera | 0.8609 | 9.8 % (11.9 % asintótico) | 0.9471 | 1.2 % |
| + Tφ | 0.8321 | 2.9 % | 0.9444 | 0.3 % |
| + térmico (n_th, p_e) | 0.7906 | 4.2 % | 0.9348 | 1.0 % |
| + κ_int | **0.7180** | 7.3 % | **0.9255** | 0.9 % |

Quitar un mecanismo del modelo completo publicado (B): sin T1 → 0.803; sin desfase → 0.742; sin térmico (n_th=p_e=0) → 0.778; sin p_e → 0.755; sin κ_int → 0.791; con |f,0⟩ inicial ideal → 0.805. Caso A: completo 0.609; sin pulsos imperfectos 0.756 (los errores de preparación con g encendido cuestan ≈15 %).

### 4.2 Escalado (T_w=0.5 µs, B, κ=10.8 MHz)

| T1 (µs) | 1.14 | 5 | 10 | 50 |
|---|---|---|---|---|
| solo T1 (resto ideal) | 0.861 | 0.935 | 0.947 | 0.957 |
| solo T1, T_w=1 µs | 0.878 | 0.963 | 0.977 | 0.988 |
| resto publicado | 0.718 | 0.782 | 0.792 | 0.801 |
| resto mejorado | 0.841 | 0.914 | 0.926 | 0.935 |

Ley empírica: ε_T1≈0.11 µs/T1 (tiempo efectivo ≈35–40 ns en estados con 1–2 excitaciones, con f→e a 2/T1). ε_φ≈0.0085 µs/Tφ (Tφ=0.1/0.3/1/3 µs → 8.1/2.9/0.9/0.3 %). Tφ (publicado, T1=1.14): 0.780/0.832/0.852/0.858/0.860/0.861 para Tφ=0.1/0.3/1/3/10/∞. A T1≥5 µs el techo lo ponen κ_int, p_e y el térmico (≈0.80).

Sensibilidad a la escalera (T1=1.14, resto ideal): escalera (f→e a 2γ_q) 0.861; sin escalera (a γ_q) 0.890; escalera + decaimiento directo f→g a γ_q 0.832; f→e apagado 0.920. La incertidumbre sobre el T1 de |f⟩ (no medido) vale ≈±3–4 % en P2.

### 4.3 Térmico
* **p_e (qubit)**: P2=0.755/0.748/0.733/0.718/0.681 para p_e=0/1/3/5/10 % (≈−0.74·p_e: el qubit en |e⟩ acaba en |g⟩ tras el primer π).
* **n_th del modo, guía fría / detector ideal** (el fonón térmico inicial sale en ~20 ns, antes de la ventana; el calentamiento entra por κ_int): P2=0.740/0.733/0.718/0.670/0.554 para n_th=0/0.01/0.03/0.1/0.3 (publicado, p_e=5 %); mejorado-T1=10: 0.875/0.865/0.846/0.783/0.632. Coste ≈ −7 %·(n_th/0.1).
* **n_th de la guía (cota pesimista, contando todos los clics térmicos)**: la tasa de clics de fondo es κ_m·n_g≈0.68 µs⁻¹ por cada 0.01: P2=0.740/0.563/0.333 para n_g=0/0.01/0.03 (los valores a n_g≥0.1 están dominados por alias de conteo, solo indican pérdida total). Requiere **n_g≲10⁻³ (T_guía≲15 mK)** para perder <2 %; la discriminación temporal (ventana corta ~0.2 µs, sincronizada con el disparo) mejora el factor ≈2.5.

## 5. Orden N=3: |3,0⟩→|g,3⟩

* Resonancia ω_q=ω−α → Δq=+138.0 MHz; acoplo efectivo numérico V_3=1.04 MHz (teoría 6g³/α²=1.11 MHz; 5.5× menor que V_2). Los tres π con g apagado dejan P(|3⟩)=0.9886.
* Estático (|3,0⟩ ideal, ventana 3 µs; `datos/mapa_estatico_N3.npz`): P3 máx = 0.949 (T1=∞, κ_m=0.47 MHz), 0.904 (T1=10 µs, κ=0.72), **0.653 (T1=1.14 µs, κ=0.72 MHz)**. El nivel |3⟩ decae a 3/T1 (380 ns): la conversión (≈1/V_3≈150 ns por fase, ventana 1–3 µs) compite con T1.
* Con pulsos reales (κ=0.72 MHz, B), P(3) a T_w=0.5/1/2/3 µs: ideal 0.46/0.86/0.93/0.94; solo T1: 1.14→0.32/0.55/0.59/0.59; 5→0.42/0.77/0.84/0.84; 10→0.44/0.82/0.88/0.89; 50→0.45/0.85/0.92/0.93; **publicado completo → 0.06/0.09/0.10/0.10** (Tφ=0.3 µs destruye la coherencia f_3–g: 9/Tφ); **mejorado (T1=10, Tφ=3…) → 0.34/0.64/0.69/0.69**; mejorado con T1=50, Tφ=10 → 0.74 (3 µs). Contaminación de P(2) ≈0.20–0.22 en publicado/mejorado.
* Flujo N=3 (reset activo 100 ns): mejorado 0.64/(0.072+1+0.1)=0.55 MHz (T_w=1 µs); publicado 0.10/3.2 µs=0.03 MHz. **No viable con los parámetros publicados; marginal con T1≥10 µs, Tφ≥3 µs y ventanas largas** (κ_m no reoptimizado para el caso mejorado).

## 6. Veredicto honesto

1. **Como dispositivo medido, no**: κ_m=0.37 MHz (modo cerrado sin guía), no hay guía fonónica ni detección, el T1 de |f⟩, la ocupación térmica y la deriva no están caracterizados, y no se demostró estacionar el qubit y volver a resonancia con pulsos de flujo.
2. **Con un rediseño (κ_m≈8–20 MHz hacia guía, detección ideal) y los demás parámetros publicados**: P(2)=0.72±0.05 (0.74 a 20 MHz), pureza 0.80, flujo ≈1.1 MHz con reset activo de 100 ns, ≈0.1 MHz con reset pasivo. El 0.918 de E (solo T1, T_w→∞) es optimista porque ignora: finita ventana (−2 %), p_e (−4 %), κ_int (−7 %), Tφ (−3 %) y térmico (−2 %).
3. **Con mejoras plausibles** (T1≈10 µs, Tφ≳3 µs, n_th≲5·10⁻³, p_e≲1 %, κ_int≲0.05 MHz): P(2)≈0.93 (T_w=0.5), 0.95 (T_w=1), pureza 0.94, flujo 1.4 MHz (activo) o 18 kHz (pasivo, T1=10 µs). Techo con T1=50 µs: 0.94–0.97.
4. **Qué bloquea, por orden** (publicado): T1 de |e⟩,|f⟩ (−10 %), κ_int (−7 %, corregible con mayor κ_m), p_e + térmico (−4 %), ventana (−4 %), Tφ (−3 %). Errores de pulso: despreciables **si** el qubit se estaciona (caso B); con g encendido (caso A) cuestan ≈11–15 % y reducen la P(2) a 0.61. La fuga virtual por |e,1⟩ es irrelevante (≲10⁻⁵ a κ_m≥8 MHz). Además, el reset es el cuello de botella del flujo.
5. **Incertidumbres sistemáticas** (no estadísticas, que son <0.01): Tφ markoviano de 0.30 µs (cota pesimista; con Tφ=∞ P2 sube a 0.74): ±0.03; T1 de |f⟩ (escalera frente a otras): ±0.03; p_e y n_th no medidos: ±0.04; κ_int con guía: ±0.03; efecto de los 4 modos espectadores (2.0–2.4 GHz, g≈13–16 MHz) y de ruido 1/f real, no modelados; T1 a la frecuencia de operación (1.0–1.4 µs). Combinadas: P(2)_publicado≈0.72 con rango razonable 0.60–0.80; mejorado≈0.93 con 0.88–0.96. El flujo supone η=1, que es el supuesto más optimista de todos.
6. Orden N=3: no viable con lo publicado (P3≈0.10), marginal con mejoras (≈0.65–0.7 y flujo ≲0.5 MHz).

## 7. Archivos

* `codigo/simP.py` (modelo, calibración de pulsos, conteo exacto), `run_P1.py` (mapas estáticos; la parte N=3 de este script quedó interrumpida y fue sustituida por `run_P4a.py`), `run_P2.py` (calibración, barridos κ y desintonía, duración de pulso/DRAG), `run_P3.py` (cascada, T1, n_th, p_e, Tφ, escalera, flujo), `run_P4a.py` (Bloch-Siegert, local frente a vestido, N=3), `run_P4b.py` (serie temporal QuTiP mesolve+mcsolve), `run_P5.py` (κ>20 MHz, cascadas T_w, promedio cuasi-estático, n_th(T)), `val_qutip.py`, `make_fig.py`.
* `datos/figuras_P.npz`: P2 vs κ_m y desintonía (`P_km_*`, `P_off_*`, `KM`, `OFF`; eje 1=T desde 0 a 3 µs en pasos de 0.1, última dim = [P0,P1,P2,P≥3]), mapas estáticos 2D, presupuesto vs T1 (`P2_vs_T1_*`) y vs n_th, serie temporal (`ts_*`: poblaciones P(k=0..3), ⟨n⟩ y clics acumulados del ensemble a 2 ns; una trayectoria con dos clics a 93 y 132 ns), mapas N=3. Otros: `barrido_completo.npz`, `mapa_estatico_N2/N3.npz`, `presupuesto.pkl`, `extras.pkl`, `N3_pulsos.pkl`, `serie_temporal.pkl` (100 trayectorias), `val_qutip.json`, `dressed_bs.json`, `pulsos_tp_drag.npy`, `figuras_P.png`.
* Reproducción: `OMP_NUM_THREADS=1 python3 codigo/run_P2.py && ... run_P3.py ...` (cada script ≲5 min).
