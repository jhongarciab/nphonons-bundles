# VA: verificación independiente (ruido de frecuencia por TLS, Φ de Hitchcock)

Código en `verif/va/`: `mc.py` (cf cerrada, cruce), `mc2.py` (MC telegráfico exacto, tiempos de espera exponenciales, integral exacta de s(t) por segmentos, vectorizado con searchsorted), `A_checks.py`, `synth.py` (B), `devC.py` (C), `gridbias.py`. Partes A y B se hicieron antes de leer `tls_model.py`. No se hizo commit ni push.

## A. Derivaciones

### A1. Función característica del telegráfico
s=±1 estacionario (P(+)=P(-)=1/2), salto con prob. γ dt. Sea u_±(t)=E[e^{-iδ∫s}; s(t)=±]. Entonces
u̇_+ = (−iδ−γ)u_+ + γu_-,  u̇_- = (+iδ−γ)u_- + γu_+, con u(0)=(1/2,1/2).
Matriz M = −γI + [[−iδ,γ],[γ,iδ]]; autovalores −γ ± μ, μ=√(γ²−δ²). La condición inicial simétrica y F=u_++u_- dan
**F(t)=e^{−γt}[cosh μt + (γ/μ) sinh μt]** (μ→iΩ: cos Ωt + (γ/Ω) sin Ωt; δ=γ: e^{−γt}(1+γt)).
Comprobación: F(0)=1, F'(0)=0, F''(0)=−δ² (=−δ²⟨s s⟩ integrada, correcto).
Límites: δ→0: F→1. γ→∞: μ≈γ−δ²/2γ ⇒ F→exp(−δ²t/(2γ)). γ→0: μ=iδ ⇒ cos δt. Coincide con la autocorrelación e^{−2γ|τ|} (σ_φ² a tiempos cortos δ²t²).
Verificación numérica (MC exacto 2·10⁵ trayectorias vs fórmula, δ,γ,t varios): diferencias ≤0.0013 (ruido estadístico ~0.002); límite gaussiano (γ=200) y estático (γ=1e-4) correctos. Nota: la forma `e^{-γt}cosh(...)` desborda (0·∞) para γt≳700; la forma del autor (exponenciales separadas) es estable.

### A2. Φ para ruido gaussiano
Fase gaussiana de media cero: ⟨e^{−iφ}⟩=e^{−σ_φ²/2} ⇒ f=σ_φ²/2. Modos con igual amplitud: f_I=2f=σ_φ². Fase relativa: Var(φ1−φ2)=2(1−ρ)σ_φ² ⇒ f_xc=(1−ρ)σ_φ²=(1−ρ)f_I (ρ = coeficiente de correlación de las fases; coincide con el de las frecuencias si los espectros son idénticos y la covarianza cruzada es ρ×espectro). Si f_I=c t^n: t_I=c^{−1/n}; (1−ρ)c t_xc^n=1 ⇒ t_xc=t_I(1−ρ)^{−1/n}.
**Φ=1−t_I/t_xc=1−(1−ρ)^{1/n}.** n=1 (blanco): Φ=ρ. n=2 (cuasi-estático): 1−√(1−ρ). General n: ρ=0.6 ⇒ 0.600, 0.368, 0.263 (n=1,2,3). ρ<0 da Φ<0 (anticorrelación empeora). Verificado numéricamente con gaussianas correlacionadas (var de la diferencia 0.7998 vs 0.8).
Límite: no es exacto para telegráficos (no gaussianos) ni cuando f no es una ley de potencias pura.

### A3. Doble carril vs modo simple
|10⟩+|01⟩ (fonón compartido): fase relativa φ1−φ2 (el fonón único pierde fase solo por la diferencia de energías). Var(φ1−φ2)=α1²σ²+α2²σ²−2ρα1α2σ² (σ² la varianza de fase unitaria); α1=α2=α: 2(1−ρ)σ_φ², σ_φ²=α²σ². Modo simple (|0⟩+|1⟩): fase φ, varianza σ_φ². Equilibrio: 2(1−ρ)=1 ⇒ **ρ*=1/2** (amplitudes iguales; con desiguales α1²+α2²−2ρα1α2=α_ref²). Ganancia en T_φ (f=1): ruido blanco G=T_dr/T_1=1/(2(1−ρ)) (ρ=0.9→5, ρ=0.5→1); cuasi-estático G=[2(1−ρ)]^{−1/2}; general [2(1−ρ)]^{−1/n}. Con α1≠α2 hay ruido residual (α1−α2)² aun con ρ=1.

### A4. Cota |⟨e^{iφ}⟩|≥1−⟨φ²⟩/2
**Es verdadera para cualquier φ real** (no hay contraejemplo): |z|≥Re z y cos x≥1−x²/2 ⇒ |⟨e^{iφ}⟩|≥⟨cos φ⟩≥1−⟨φ²⟩/2. La advertencia del enunciado (media de fase no nula) no la rompe, solo la hace floja: eligiendo α libre, |⟨e^{iφ}⟩|≥⟨cos(φ−α)⟩≥1−⟨(φ−α)²⟩/2 y con α=⟨φ⟩ queda la cota más fuerte **|⟨e^{iφ}⟩|≥1−Var(φ)/2** (Var≤⟨φ²⟩). Es vacua si Var≥2. Consecuencia: f=−ln|⟨e^{−iφ}⟩|≤−ln(1−Var/2), así que f=1 exige Var(φ)≥2(1−e^{−1})=1.264: T_φ ≥ tiempo en que Var φ=1.264 (cota inferior de T_φ). No existe cota superior universal (p. ej. fases bimodales dan |⟨⟩|=0 con Var finita). Check numérico con telegráfico (γ=1, A=1.5, t=3): |⟨⟩|=0.058≥1−Var/2=−1.81 (válido, vacuo).

## B. Monte Carlo en dominio del tiempo (40000 trayectorias, T en µs, rad/µs)
Dispositivos sintéticos; MC vs fórmula cerrada A1 (misma malla geométrica 70 pts, interpolación lineal) y vs cruce en malla fina (2·10⁵ pts):

| dispositivo | max|f_MC−f_cf| (f<1.5) | t_I MC / cf / fino | t_xc MC / cf / fino | Φ MC / cf / fino |
|---|---|---|---|---|
| D1 4 compartidos (A1=A2) | 0.002 | 0.809/0.809/0.810 | f_xc≡0 (nunca) | Φ→1 |
| D2 6 privados | 0.020 | 0.825/0.827/0.827 | 0.826/0.827/0.827 | 0.001/0/0 |
| D3 5 opuestos (A2=−A1) | 0.011 | 0.623/0.621/0.621 | 0.419/0.418/0.418 | −0.485/−0.485/−0.484 |
| D4 8 mixto (compart., privados, opuestos) | 0.010 | 0.604/0.604/0.605 | 1.367/1.371/1.371 | 0.558/0.559/0.559 |
| D5 3 cuasi-estáticos (γ~0.03) | 0.014 | 0.799/0.798/0.799 | 1.338/1.339/1.343 | 0.403/0.404/0.405 |

Acuerdo dentro del ruido (~1%). Casos de signo opuesto: Φ<0 (D3: f_xc=Σ f(2A)), correlación exacta (D1): f_xc≡0. Para D5 (n≈2) el valor Φ≈0.40 es coherente con una ley intermedia.

## C. Revisión de `src/tls_model.py`
Reproducción exacta de `sample_device`/`phi_curve` con `default_rng(semilla)` (mismas llamadas, parámetros nf=0.05, W=80, g∈[0.002,0.5] MHz, γ∈[1e-3,1e2], fsr=3.29, amp=1), MC exacto 20000 trayectorias hasta T≤450 µs:

| dispositivo | N | f1 / f2 / f_xc: max|ΔMC−código| (0.05<f<2) | t1 código/MC | t2 | t_I | t_xc | Φ código/MC |
|---|---|---|---|---|---|---|---|
| trav s3 | 6 | 0.02/0.01/0.01 | 358.6/360.9 | 32.8/32.8 | 32.1/32.1 | 24.7/24.5 | −0.301/−0.307 |
| stand s9 | 11 | 0.06/0.04/0.005 | 32.5/32.2 | 32.9/32.6 | 23.4/23.2 | 57.7/57.5 | 0.594/0.596 |
| trav s7, s9, s10, s7+2 glob., s9+2 glob. | 6–11 | ≤0.07 (f1,f2); f_xc coinciden | t1 98.8/100.1 (s10), 55.0/54.2 (s9), 36.1/36.3 (s9+glob) | | | t_xc>cap | Φ_código≈0.977–0.99 (cap) |
| stand s1, s3 | 9, 6 | ≤0.04 | | | | | f_xc(450 µs) 0.97/0.98 (s1); s3 f_xc≫1 |

Todas las diferencias son estadísticas (≲0.06 en f; ≲1.3% en tiempos). **No hay error de implementación en cf, logabs_prod ni en la estructura A1−A2 ni en los globales** (se cancelan en f_xc; entran igual en f1,f2): verificado con MC (glob_n=2). Unidades: `2π·amp·g²·u²/(k−Om)` con g,Δ en MHz (=g/2π, Δ/2π) da MHz→rad/µs correcto (chequeo numérico: g=0.1,Δ=10 ⇒ −6.283e-3). Perfil standing: u²=2 sin²(kmode·ph), ph∈[0,π], media 1 (misma normalización que travelling); correlación media de u²_1 y u²_2 es 0 (E cos2a=0), es decir añade solo variabilidad.
Tφ: la clave `t1` (2ª columna de `phi_curve`) = primer cruce de f_1=1 (incluyendo globales): sí es Tφ del modo 1 con criterio 1/e de |⟨e^{−iφ}⟩|.

### Hallazgos (por impacto)
1. **Tope de 3000 µs y Φ censurada** (`tls_model.py:14-16` `first_cross` devuelve `tgrid[-1]` si no hay cruce; el argumento `cap` nunca se usa; TG[-1]=3000 en l.21). En 300 dispositivos (nf=0.05, travelling): 25% ambos topados ⇒ Φ=0 espurio; 38% con t_xc topado y t_I no ⇒ Φ=1−t_I/3000≈0.9–1 (cota inferior, no valor); standing: 26% / 12%. La media de Φ "todos" (0.545 trav, 0.328 stand) difiere de la de dispositivos sin censura (0.762, 0.474). Φ estimada con topes no es citable como media; hay que tratar los datos como censurados (NaN/cota) o ampliar el tope. Impacto: factor ≈1.4–1.7 en ⟨Φ⟩.
2. **Cruces con f ya ≥1 en el primer punto** (l.18): `if i==0: return tgrid[0]` (0.5 µs) sin extrapolar; sesgo alto si f(0.5)≫1 (no ocurre en estos parámetros).
3. **Sesgo de la malla geométrica (70 pts) por interpolación lineal**: cuantificado contra mallas de 2000 y 6000 puntos (300 dispositivos): ΔΦ media 0.0000, rms 0.001, máx 0.011; Δt1 media −0.17 µs, rms 0.35 µs (mediana relativa −0.08%). **Despreciable**; 2000 vs 6000 pts: 3·10⁻⁴. Para F no monótona no se detecta (ver 4).
4. **Primer cruce y f no monótona** (l.15-19): con γ pequeño y δ>γ, |F|≈e^{−γt}|cos δt| tiene ceros/revivals; en 41 de 210 dispositivos con cruce, f1 vuelve por debajo de 0.8 tras el primer cruce (20%). T_φ es "primer paso por 1/e", no robusto frente a la malla de muestreo (aliasing de oscilaciones rápidas con TG de 70 pts); como definición es legítima pero conviene reportarla así y comparar con otros criterios (p. ej. último cruce).
5. `cf` (l.8-10): sustitución |w|<1e-12→1e-12 solo afecta a δ=γ exacto (error ~1e-4 relativo); inocuo.
6. `n=int(round(d/fsr))` (l.46, 60): el modo 2 usa índice kbase+n pero Δ=d; solo coherente si d es múltiplo de fsr. n=0 (d<fsr/2) da A1=A2 ⇒ f_xc≡0 (Φ→1−t_I/3000). Ocurre en `predict1.py` solo con d=fsr: sin problema ahí.
7. N=0 TLS (Poisson) ⇒ f=0, t=3000, Φ=0 (misma censura).
8. Sin regularización del denominador k−Om: |Δ|<g ocurre en 0.1% de los TLS; el sumando entonces es f≈γt (saturado) pero ya no es dispersivo (ver D).
9. `amp` (fit1.py etc.) es un parámetro libre de ajuste que multiplica χ: absorbe factores de definición; debe reportarse si ≠1.
No se hallaron errores en signos A1−A2 ni en unidades.

## D. Interpretación física
- **Factor 2**: JC dispersivo, H≈(ω+χσ_z)a†a con χ=g²/(ω−Ω) (g acoplo vacío, H_int=g(aσ_++h.c.)): el TLS en |e⟩ vs |g⟩ desplaza el modo ±χ, es decir diferencia total 2χ; el modelo usa A=χ y s=±1 ⇒ swing 2A, consistente. Falta factor solo si g está definida con otra convención (g/2 de Rabi, g=Ω_R/2, o σ_x acoplo con 2g), o si f y g son frecuencias vs angulares: ver qué es "g" en los fits. χ también tiene corrección ∝g⁴/Δ³ y, para multi-fotón, no lineal (Kerr TLS-modo). Para estado de baja ocupación el desplazamiento depende de n (a†a): A es la corrección a un fonón.
- **|Δ|<g**: la dispersiva falla; el espectro tiene división de Rabi (modos híbridos), el "ruido de frecuencia" no es ±A sino saltos entre ramas; con Δ→0 el resultado f≈γt es solo cualitativo. Fracción pequeña (0.1% de TLS) pero son los de mayor impacto individual; deben tratarse con diagonalización JC exacta o excluirse.
- **Difusión espectral vs población**: el modelo telegráfico representa tanto TLS que cambian de población (T1 del TLS ⇒ s cambia entre |e⟩,|g⟩, con rate asimétrico en equilibrio térmico, no ±1 simétrico a ℏΩ≫k_BT) como TLS fijos cuya frecuencia Ω_j salta (difusión espectral por TLS térmicos: efecto sobre A_jk vía Δ). Son dos mecanismos distintos: en el primero δω=±χ con s estocástico de población (la asimetría P(e)≠1/2 introduce un offset constante y reduce la varianza 4p(1−p)); en el segundo A depende de Ω_j(t) (modulación 1/Δ², no ±). Aquí se asume estacionario simétrico con P=1/2: válido para ℏΩ≲k_BT (kT a 20 mK≈0.4 GHz; los TLS ~GHz están principalmente en |g⟩, luego los fluctuantes son los de Ω≲kT) — la regla de selección térmica (peso 4p(1−p)) reduciría el número efectivo nf, no el signo o la estructura.
- **Robusto**: estructura Φ=1−(1−ρ)^{1/n} (A2), cf telegráfica exacta, correlación A1−A2 ⇒ TLS entre los modos y con signo opuesto empeoran Φ, TLS lejanos comunes (A1≈A2) se cancelan, signo de Φ y orden de magnitudes de T_φ relativo. Φ>0 requiere TLS dominantes compartidos; Φ<0 con TLS situados entre los modos.
- **Dependen de supuestos**: valor absoluto de T_φ (nf, g, γ, amp), perfil u² (standing vs travelling difieren en ⟨Φ⟩ 0.33 vs 0.55), asimetría térmica, cap/censura en Φ, y la validez dispersiva.

## Veredicto
**Citables**: (i) F(t)=e^{−γt}[cosh μt+(γ/μ)sinh μt] y sus límites (verificada por MC y por el código); (ii) Φ=1−(1−ρ)^{1/n} (gaussiano), ρ*=1/2 y G=1/(2(1−ρ)) (blanco); (iii) cota |⟨e^{iφ}⟩|≥1−Var/2 (verdadera, con precisión de Var centrada); (iv) implementación de cf, amplitudes, A1−A2, globales, perfiles y unidades en `tls_model.py` (reproducida por MC independiente, diferencias estadísticas); (v) sesgo de malla TG: ≲0.1% (despreciable).
**No citables sin corrección**: medias/distribuciones de Φ de `phi_curve` por la censura a 3000 µs (25–38% de dispositivos afectados; Φ=0 o ≈1 espurios); Tφ como primer cruce cuando f1 no es monótona (20% de dispositivos); valores absolutos dependientes del modelo físico (factor de g, `amp`, estadística térmica, validez dispersiva).
