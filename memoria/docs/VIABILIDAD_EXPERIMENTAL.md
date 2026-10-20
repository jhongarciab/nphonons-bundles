# Viabilidad experimental de la memoria de fonones con estados oscuros (16 oct 2026)

Búsqueda dirigida a cada requisito del esquema (`tex/memoria_teoria.tex`). **Estado de cada fuente:** `[TEXTO]` leído en texto completo (PMC); `[FICHA]` solo resumen/ficha o resultado de búsqueda; `[INFERIDO]` cálculo propio a partir de cifras de otra fuente.
Nada de lo de abajo está contrastado por lectura del artículo completo salvo lo marcado `[TEXTO]`; las cifras de las fichas hay que revisarlas antes de citarlas.

## 1. Resumen: requisito frente a estado del arte

| # | Requisito del esquema (punto de trabajo 100 GHz) | Lo mejor encontrado | Semáforo |
|---|---|---|:-:|
| 1 | Acoplo fonón–excitón $\lambda_i/\omega_b\sim0.1$–$0.2$ ($g_s\simeq0.06\,\omega_b$) | QD–resonador fonónico a 1.466 GHz: $g_{ep}/2\pi=2.9$ MHz, $g_{ep}/\Omega_m\simeq2\times10^{-3}$, $Q_m=2.1\times10^{3}$, $x_{zpf}=2.7$ fm, $V=4\times10^{-3}\lambda^3$ `[TEXTO]` (Spinnler 2024). Escalado propio a 100 GHz: ver §2 | rojo/amarillo |
| 2 | Cavidad acústica a $\sim100$ GHz con $Q\gtrsim10^3$ ($f\!\cdot\!Q\simeq10^{14}$) y un QD dentro | $f\!\cdot\!Q$ hasta $10^{14}$ en membranas 2D a 50–600 GHz **sin QD**, temperatura ambiente `[FICHA]` (Zalalutdinov 2021); $Q>1000$ a 10–100 GHz en VCSEL GaAs/AlAs, modulación hasta 40 GHz `[FICHA]` (Czerniuk 2014); cavidad de superred a $\sim350$ GHz `[FICHA]` (Lamberti 2017); micropilar optofonónico a 300 GHz compatible con QD `[FICHA]` (Esmann 2019). Resonador con QD: $f\!\cdot\!Q=3\times10^{12}$ Hz `[TEXTO]` | amarillo |
| 3 | $T\lesssim0.5$–$1$ K ($n_{\rm th}\ll1$ a 100 GHz) | El mismo experimento opera a 4.2 K con $\langle n_m\rangle=58$; a 100 mK, $\langle n_m\rangle=0.98$ `[TEXTO]` (a 1.47 GHz; a 100 GHz $n_{\rm th}=0.008$ a 1 K `[INFERIDO]`) | amarillo |
| 4 | Pareja/triple de QD con Förster $J\sim0.1\,\omega_b\simeq0.04$–$0.08$ meV, **sin túnel** | Pareja apilada: tunelamiento suprimido entre dots no idénticos, acoplo dipolar y transferencia asistida por fonones `[FICHA]` (Gerardot 2005); anticruce de $4.5\to1.5$ meV para separaciones de $10\to12$ nm `[FICHA]` (Krenner 2005); desdoblamiento $>30$ meV a 4 nm `[FICHA]` (Bayer 2001). Triples apilados: **no encontré** | amarillo/rojo |
| 5 | Control **independiente** de $\varepsilon_i$ (Stark) a $\sim10$ µeV con separaciones de 7–10 nm; $\varepsilon_1-\varepsilon_2\simeq0.05$ meV | Un solo campo vertical cambia la alineación de la pareja (Krenner); control local a escala nanométrica: **no encontré** | rojo |
| 6 | Biexcitón con $|\Delta_{xx}|\gtrsim6\,\omega_b\simeq2.5$ meV | InAs/GaAs: ligadura de pocos meV, el **signo** cambia con el tamaño (binding en dots grandes, antibinding en pequeños); $+2.1$ meV en un dot pequeño `[FICHA]`. Mi simulación admite ambos signos | amarillo |
| 7 | Ruido de carga: $\sigma_1\lesssim0.3\,\mu$eV, volumen sin trampas $r_0\gtrsim12$–$34$ nm | Líneas casi limitadas por transformada en dots en diodo (Kuhlmann 2013 `[FICHA]`; Zhai 2020 GaAs `[FICHA]`); en el resonador de Spinnler, $\Gamma_{\rm inh}/2\pi=550$ MHz $=4\,\Gamma_R$ ($\sigma\simeq1\,\mu$eV) `[TEXTO]`; **superficies grabadas a $\lesssim200$ nm ensanchan hasta varios GHz** `[FICHA]` (Liu 2018) | rojo/amarillo |
| 8 | Ruido **correlacionado** entre los puntos ($\chi\to1$) | Correlaciones de ruido de carga medidas en dots de silicio: coherencia de $>0.5$ a $<0.1$ al pasar de 75 a 300 nm `[FICHA]` (PRApplied 23, 064058, 2025); modelo de TLS con decaimiento exponencial con la distancia `[FICHA]`; propuesta teórica de sondear fluctuadores con parejas de QD ópticos `[FICHA]` (Purohit 2015). Medición en una pareja apilada de 7–10 nm: **no encontrada** | amarillo |
| 9 | Acoplador **conmutable** $\kappa_{\rm in}/\kappa\simeq10$ a 100 GHz | Acopladores sintonizables de fonones solo a pocos GHz: SAW–transmon con acoplo $+7\to-20.6$ MHz `[FICHA]`; propuesta de circuitos fonónicos a 5.14 GHz `[FICHA]`; acoplador direccional de un solo fonón `[FICHA]`. **Nada a 100 GHz** | rojo |
| 10 | Pulsos Raman de Rabi $\Omega\simeq0.2\,\omega_b$ (20 GHz) y duración $\sim170/\omega_b\simeq0.27$ ns | Rabi con pulsos de ps habituales; el límite es el desfase inducido por excitación (acoplo al baño de fonones durante Rabi) `[FICHA]` (arXiv 1304.4756 y Ramsay 2010 en `REFERENCIAS.bib`) | amarillo |
| 11 | Detección/preparación de un fonón de 100 GHz | Estados de Fock de fonones sólo a GHz con qubits superconductores (Chu 2017, 2018; PRX 9, 021056 (2019), autores no verificados `[FICHA]`); **ninguno a 100 GHz en semiconductores** | rojo |

## 2. Lo que dice el experimento más cercano (Spinnler et al., Nat. Commun. 15, 9509, 2024) `[TEXTO]`
- QD de InAs (trión negativo) en un haz de GaAs de $960\times180$ nm$^2$ con escudo fonónico; $\Omega_m/2\pi=1.466$ GHz, $Q_m=2.1\times10^3$, $m_{\rm eff}=7.4\times10^{-16}$ kg, $x_{zpf}=2.7$ fm, $V=4\times10^{-3}\lambda_{\rm ac}^3$ con $\lambda_{\rm ac}=3.25\,\mu$m.
- $g_{ep}/2\pi=2.9$ MHz (simulado 3.2 MHz), $g_{ep}/\Omega_m\simeq2\times10^{-3}$; $\Gamma_R/2\pi=135$ MHz; ancho inhomogéneo 550 MHz ($4\Gamma_R$).
- $T=4.2$ K con $\langle n_m\rangle=58$; el artículo estima que con 100 mK, $5Q_m$, $2g_{ep}$ y línea limitada por transformada se llega a $\langle n_m\rangle=0.56$. Propone: refrigeración, mejor $Q_m$ (pasivar superficies, soft-clamping), más $g_{ep}$ (haz más estrecho, $g_{ep}\propto m_{\rm eff}^{-1/2}$) y **varios QD en un resonador** para acoplo excitón–excitón.
- No discute el ruido de carga como limitación.

**Escalado propio a 100 GHz `[INFERIDO]`.** $\lambda=g_{ep}/\Omega\propto D\,\varepsilon_{zpf}/\Omega$ con $\varepsilon_{zpf}\propto\sqrt{\Omega/(\rho V c^2)}$. Si el volumen modal escala como $\lambda_{\rm ac}^3\propto\Omega^{-3}$ (misma geometría reducida), $\lambda\propto\Omega$ y el factor 68 entre 1.47 y 100 GHz lleva $2\times10^{-3}\to0.14$, justo el valor del modelo. **Pero** eso exige $V=4\times10^{-3}(47\,{\rm nm})^3\simeq(7.5\,{\rm nm})^3$, por debajo de lo fabricable.
Con $V\sim(0.1$–$1)\lambda_{\rm ac}^3$ ($25$–$250$ veces mayor) $\lambda$ baja un factor 5–16, es decir $\lambda\sim0.01$–$0.03$, coherente con mi estimación 1D anterior ($0.02$–$0.07$, `GEOMETRIA_DELTA_LAMBDA.md`). **Es el riesgo central:** el modelo supone $\lambda=0.14$–$0.20$.

**Sensibilidad del esquema a $\lambda$ menor** (de las fórmulas de `memoria_teoria.tex`): $\Omega_{\rm eff}\propto g\propto\lambda$, de modo que $t_\pi\propto1/\lambda$ a $\Omega$ fija. Con $\lambda/3$: $t_\pi\to\sim550\,\omega_b^{-1}$ (0.9 ns), la pérdida del fonón durante la escritura $\kappa t_\pi/2$ se triplica, y subir $\Omega$ para compensar exige $\Omega\simeq0.6\,\omega_b$ (Rabi 60 GHz) con mayor población virtual del brillante $(\Omega/\Delta_I)^2$. No lo he simulado.

## 3. Otras observaciones de viabilidad

**Geometría de la molécula.** El esquema pide tres puntos con $J_{12}=0.2$, $J_{23}=0.075$, $\varepsilon_1-\varepsilon_2=-0.125$ (en $\omega_b$, es decir $0.083$, $0.031$ y $-0.052$ meV a 100 GHz), con distancias $r\simeq7$–$11$ nm y $r'/r=1.39$.
- A 10–12 nm de separación vertical el desdoblamiento por tunelamiento es de 1.5–4.5 meV (Krenner `[FICHA]`), muy superior a $J$. Hace falta que el túnel esté suprimido por **desafinación de los niveles de un solo portador** (pareja no idéntica, Gerardot `[FICHA]`) o por una barrera gruesa; eso choca con que las energías de transición de los excitones deban coincidir a decenas de µeV.
- Los dots autoensamblados tienen dispersión de energía de meV; ajustar $\varepsilon_i$ a $\sim10\,\mu$eV con separaciones de nm necesita tuning local (no encontré técnica) o selección de dots por espectroscopía.

**Orden de niveles.** Con dipolos ópticos en el plano en pila vertical, $J>0$ y el brillante queda arriba: es el orden que usa el esquema (D por debajo de B). No es una dificultad adicional.

**Ruido y superficies.** Liu 2018 `[FICHA]`: superficies grabadas a menos de unos 200 nm ensanchan la línea hasta varios GHz. Un resonador fonónico de haz suspendido a 100 GHz tiene dimensiones $\sim\lambda_{\rm ac}=47$ nm, con superficies a decenas de nm del dot: el requisito $r_0\gtrsim12$–$34$ nm sin trampas y $\sigma_1\lesssim0.3\,\mu$eV es incompatible con esa geometría. Una **cavidad plana de superred GaAs/AlAs** o un micropilar de micras, sin superficies a 50 nm, es más compatible con el ruido (cavidades de 300–350 GHz existen `[FICHA]`).

**Temperatura.** El requisito es $T\lesssim0.5$–1 K; el experimento de referencia estima que 100 mK es el camino y que a 4.2 K el modo está lleno de fonones térmicos (58 a 1.47 GHz; a 100 GHz son 0.43 a 4 K `[INFERIDO]`).

**Acoplador conmutable.** El punto más débil: el esquema pide $\kappa_{\rm in}\simeq10\kappa$ durante la carga y $\kappa$ al almacenar, a 100 GHz. Solo hay acopladores de fonones a pocos GHz (SAW–transmon, propuestas piezoacústicas).
Alternativa a explorar: carga y lectura con un puerto de acoplo fijo y alineación en frecuencia (resonador sintonizado dentro/fuera de la banda del puerto por deformación) en lugar de variar $\kappa$.

**Fuente y detección.** El fonón de 100 GHz tendría que proceder de un esquema de emisión (tema de la línea anterior) o de un generador externo; no hay detección de un solo fonón a esa frecuencia en semiconductores.

## 4. Camino experimental razonable (sin la memoria completa)
1. **Medir $\lambda$ a 100–350 GHz.** QD en cavidad plana GaAs/AlAs (Esmann 2019, Lamberti 2017): bandas laterales de fonón (Franck–Condon) del excitón; el factor de Huang–Rhys da $\lambda^2$.
2. **Pareja apilada con Förster aislado.** Espectroscopía de la pareja (Gerardot) para separar Förster de túnel y medir $J$ y $\Delta_{xx}$.
3. **Espectro conjunto de difusión espectral** de los dos excitones de una pareja para medir la correlación del ruido ($\chi$) frente a la distancia (el análogo en silicio está en `[FICHA]`).
4. **Escritura/lectura Raman a 1–2 GHz** como prueba de principio con el resonador de Spinnler (un QD con transferencia fonón–exciton asistida por láser; no la molécula).
5. Solo después, el acoplador conmutable y la molécula de tres puntos.

## 5. Fuentes (todas `[FICHA]` salvo Spinnler `[TEXTO]`)
- Spinnler et al., Nat. Commun. 15, 9509 (2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11535015
- Liu et al., Phys. Rev. Applied 9, 064019 (2018): efecto de la nanofabricación. https://link.aps.org/doi/10.1103/PhysRevApplied.9.064019
- Kuhlmann et al., Nat. Phys. 9, 570 (2013); arXiv:1301.6381.
- Zhai et al., arXiv:2003.00023 (GaAs de bajo ruido).
- Gerardot et al., Phys. Rev. Lett. 95, 137403 (2005); arXiv:cond-mat/0507589.
- Bayer et al., Science 291, 451 (2001). Krenner et al., Phys. Rev. Lett. 94, 057402 (2005); arXiv:cond-mat/0410206.
- Scheibner et al., arXiv:0705.1205 (biexcitón molecular). Stinaff et al., Science 311, 636 (2006).
- Esmann et al., Optica 6, 854 (2019). Lamberti et al., arXiv:1708.05676. Czerniuk et al., Nat. Commun. 5, 4038 (2014). Zalalutdinov et al., Nat. Commun. 12, 3267 (2021).
- Brüggemann et al., Nature Photon. 6, 30 (2012); Czerniuk et al., PRL 118, 133901 (2017).
- Legrand et al., Phys. Rev. B 93, 184304 (2016) (atenuación de fonones en GaAs a 0.2–1 THz).
- Purohit, Braunecker, Lovett, Phys. Rev. B 91, 245301 (2015); ruido de carga correlacionado en silicio, Phys. Rev. Applied 23, 064058 (2025) y arXiv:2405.03763.
- Waks et al. Nano Lett. 18, 4734 (2018) (superradiancia de dos QD en guía) y Phys. Rev. Applied (control eléctrico).
- Ripin et al., arXiv:2302.13484 (acoplo fonónico sintonizable, WSe$_2$, fonones de THz).
- Tunable couplers de fonones: arXiv:2404.18540, Nature npj Quantum Inf. (reconfigurable quantum phononic circuits, 2022), arXiv:2312.04414.

## 6. Propuestas para cubrir los puntos rojos y amarillos (16 oct 2026)
Verificado numéricamente en `src/check_propuestas.py`: el diseño de 4 puntos, la tolerancia de oscuridad y la tabla de frecuencias. El resto son propuestas **aún no simuladas**.

### 6.1 Cambios de arquitectura (cada uno elimina un requisito)
| # | Propuesta | Requisito que elimina o relaja | Estado |
|---|---|---|---|
| A1 | **Doble carril en frecuencia** en una cavidad de **dos modos** ($\omega_1,\omega_2$), con dos ranuras oscuras separadas por $\omega_2-\omega_1$ y un solo láser: las dos ramas escriben y leen a la vez | Elimina el intervalo no protegido $\tau_u$ (la limitación dominante por ruido de carga: $\sigma_1\lesssim0.3\,\mu$eV) y la separación entre time-bins | sin simular; extensión directa del modelo reducido con 2 modos |
| A2 | **Rectángulo de 4 puntos** (lados $a\ne a'$, diagonales $b$, $\varepsilon_i$ iguales): los 4 autoestados tienen peso $1/4$ en cada punto; los 3 oscuros están por debajo de $B$ (a $0.34$, $0.30$ y $0.16$ en las unidades del ejemplo numérico) | La diferencia de peso $w=|D_i|^2-|D_j|^2$ es **cero**: el desfase puro entre ranuras desaparece para cualquier ruido diagonal (correlado o no); la oscuridad la fija la simetría, no un ajuste fino de $\varepsilon_1-\varepsilon_2$ | verificado: pesos $0.25$, energías $-0.14,-0.10,+0.04$ frente a $B=+0.20$ ($a{=}0.12$, $a'{=}0.05$, $b{=}0.03$); con $\lambda=(0.2,0,0,0)$ los tres acoplos $|g|=\lambda/4=0.05$ |
| A3 | **Fuente y memoria en la misma cavidad, lectura óptica**: un emisor tipo Bin crea el fonón en la cavidad, la molécula lo escribe; la salida se hace convirtiendo $D$ en brillante (pulso Stark) y detectando el fotón | Sin acoplador conmutable (#9), sin detección de fonones (#11): la memoria actúa como transductor fonón$\to$fotón | sin simular |
| A4 | **Absorción directa con pulso Raman conformado** mientras llega el paquete (tipo memoria Raman en cavidad), con $\kappa$ fijo | Evita conmutar $\kappa$ si se necesita puerto externo; pide $4\Omega_{\rm eff}^2/\kappa_{\rm in}\gg\kappa_{\rm in}$ | sin simular; condición estimada |
| A5 | **Curva de ventaja frente a $\lambda$** con pulsos reoptimizados ($\lambda$ a $1,\tfrac12,\tfrac13,\tfrac15$ del valor del modelo, $\Omega$ libre hasta $0.6\,\omega_b$) | Fija el $\lambda$ mínimo con el que la memoria supera al pasivo: da el objetivo para la plataforma (#1) | sin simular; cómputo moderado |

**Tolerancia de la oscuridad** (triple, $g_1,g_2=0.35,0.6$; verificado): un error $\delta\varepsilon_1$ en la suma de fila hace brillante al oscuro con peso $\sim(\delta\varepsilon/g)^2$:
$\delta=0.05\,\omega_b$ ($21\,\mu$eV a 100 GHz) $\to$ peso $1.6\times10^{-3}$, pérdida radiativa $0.2\,\%$ en $T=3000$; $\delta=0.10\,\omega_b$ ($41\,\mu$eV) $\to$ $6.8\times10^{-3}$, $0.8\,\%$.
La tolerancia es de $\pm40\,\mu$eV, no de $10\,\mu$eV, así que basta una perilla global (campo vertical) más preselección del dot.

### 6.2 Elección de frecuencia (compromiso)
| $f_b$ (GHz) | $\hbar\omega_b$ (meV) | $J_{12}$ ($\mu$eV) | $r_{12}$, $r_{23}$ (nm) | $|\Delta_{xx}|\ge6\omega_b$ (meV) | $\sigma_1\le\hbar\omega_b/1300$ ($\mu$eV) | $Q$ ($f\!Q=10^{14}$) | $n_{\rm th}$ 1 K / 0.3 K |
|---:|---:|---:|---|---:|---:|---:|---|
| 30 | 0.124 | 25 | 10.4, 14.5 | 0.74 | 0.10 | 3300 | 0.31 / $8\times10^{-3}$ |
| 50 | 0.207 | 41 | 8.8, 12.2 | 1.24 | 0.16 | 2000 | 0.10 / $3\times10^{-4}$ |
| **70** | 0.289 | 58 | 7.9, 10.9 | 1.74 | 0.22 | 1400 | 0.036 / $1\times10^{-5}$ |
| 100 | 0.414 | 83 | 7.0, 9.7 | 2.48 | 0.32 | 1000 | 0.008 / $\sim10^{-7}$ |
| 150 | 0.620 | 124 | 6.1, 8.5 | 3.72 | 0.48 | 670 | $7\times10^{-4}$ / $\sim10^{-11}$ |
| 200 | 0.827 | 165 | 5.5, 7.7 | 4.96 | 0.64 | 500 | $7\times10^{-5}$ / 0 |
Bajar la frecuencia **relaja** $J$ (separaciones mayores, menos túnel), $\Delta_{xx}$ (1.2--1.7 meV ya está en el rango de InAs/GaAs) y $\kappa$ (con $f\!Q$ fijo), y **sube $\lambda$** a volumen fijo ($\propto f^{-1/2}$); pero **endurece el ruido** ($\sigma_1\lesssim\hbar\omega_b/1300$) y la temperatura. **50--70 GHz con $T\lesssim0.3$ K** parece el mejor compromiso.

### 6.3 Qué haría con cada punto
- **Rojo 1 ($\lambda$):** cavidad plana de superred GaAs/AlAs (sin superficies grabadas), frecuencia 50--70 GHz, volumen modal pequeño; correr A5 para saber cuánto $\lambda$ se necesita. Alternativa de plataforma con acoplo fonónico fuerte: emisores excitónicos en WSe$_2$ (Ripin 2023) con fonones de THz, que tienen réplicas de varios fonones.
- **Rojo 2 (control de $\varepsilon_i$):** A2 (simetría en lugar de ajuste fino) + preselección + perilla global (campo/deformación); tolerancia $\pm40\,\mu$eV.
- **Rojo 3 (acoplador):** A3 (sin puertos) o A4 (sin conmutar). Solo si hace falta un puerto externo, alineación en frecuencia del resonador con la banda de un puerto de banda estrecha.
- **Rojo 4 (detección):** lectura óptica de A3 (el fonón se mide como fotón).
- **Rojo 5 (ruido):** A1 (sin $\tau_u$) + A2 ($w=0$) + cavidad plana, diodo n-i-p y pasivación ALD.
- **Amarillo $Q$:** cavidad plana/micropilar GaAs/AlAs; $f\!Q=10^{14}$ pide $Q\simeq1400$ a 70 GHz (Czerniuk: $Q>1000$ a 10--100 GHz).
- **Amarillo $T$:** $T\lesssim0.3$ K (3He/dilución) para $n_{\rm th}\ll1$ a 50--70 GHz.
- **Amarillo pareja Förster:** dots no idénticos con barrera de 9--12 nm; a 50--70 GHz $J$ de 41--58 $\mu$eV con $r_{12}$ de 8--9 nm.
- **Amarillo $\Delta_{xx}$:** a 50--70 GHz se necesitan 1.2--1.7 meV (el rango de InAs/GaAs, con el signo libre).
- **Amarillo correlación ($\chi$):** medirla (§4 punto 3); A2 la hace irrelevante para el desfase puro entre ranuras.
- **Amarillo Rabi 20 GHz:** pulsos conformados (rampa adiabática) para eliminar el rizado $(\Omega/\Delta_I)^2\sim2\,\%$ y reducir el desfase inducido por excitación.

## 7. Búsqueda dirigida a elevar λ (geometrías y materiales; 16 oct 2026)
Pregunta: ¿qué existe experimentalmente para acoplar un exciton de QD (o un emisor) a un modo acústico de 50--100 GHz con $\lambda\gtrsim0.03$--$0.07$? Estado: `[FICHA]` salvo indicación.

**Lo que encontré**
| Plataforma | Dato | Utilidad para λ |
|---|---|---|
| Cavidades acústicas GaAs/AlAs (planares y micropilares) | modos confinados de decenas a centenares de GHz; micropilares con $f\!\cdot\!Q\to10^{14}$ **a temperatura ambiente**; patronado lateral sub-100 nm por litografía de haz de electrones (revisión arXiv:2606.30510, texto casi completo); transductores a ~100 y 200 GHz (Huynh 2008, picosegundos, sin Q) | Existe el confinamiento axial y lateral a la escala de mi modelo; **la revisión no da cifras de $g$ para QD** y el caso cuantitativo con un dot dentro no apareció |
| Nanovigas y nanomembranas de GaAs con QD | modo flexural en nanoviga fotónica con QD (arXiv:2407.09456); membranas con QD y actuador piezoeléctrico (arXiv:1905.07738); disco con modo de respiración de 1.4 GHz y $Q=19\,000$ | Todo $\le$ pocos GHz; nada a sub-THz |
| QD en nanohilo | modulación acústica por SAW (Nano Lett. 2024), amortiguamiento por fonones; **no encontré** cavidades de hilo con superred axial GaAs/AlAs | Hipótesis (nanohilo con superred axial) sin respaldo en lo encontrado |
| Cristal fonónico de diamante con SiV (Kuruma, Nat. Phys. 2024) | banda completa de 50--70 GHz, detalles de ~20 nm, reducción $\times18$ de la relajación inducida por fonones | Prueba de principio de **ingeniería fonónica a 50--70 GHz con un emisor único**; no mide $g$ coherente ni es un QD |
| Polaritones de microcavidad | $g_0$ de decenas de MHz a GHz (Sesin 2022); protección del desfase por el gap de Rabi | Ruta de acoplo grande (resonante con el exciton); es otro sistema (polaritones, no QD) |
| WSe$_2$ con excitones interlayer | réplicas de fonones de THz, «acoplo excepcionalmente fuerte», sintonizable por campo (Ripin 2023) | Acoplo grande pero a THz y con emisores de TMD |
| Nitruros GaN/AlN | factor de Huang--Rhys de fonones LO entre 0.01 y 0.5 (0.02--0.3 según dot); término piezoeléctrico grande para fonones acústicos | **Sin** medida de acoplo a cavidades acústicas; no hay evidencia de $\lambda$ mayor |
| Sensibilidad a la deformación | dots en nanomembrana con actuador piezoeléctrico: ~20 meV con deformación biaxial de $\pm0.2\,\%$; una estimación a partir de un barrido de ~10 meV con $0.15\,\%$ da ~70 meV por ciento `[INFERIDO]` | Consistente con $\Delta D\sim$ unos pocos eV a 10 eV por unidad de deformación (el $\Delta D=6$--$10.5$ eV supuesto) |

**Lectura**
1. No encontré ninguna medida de $g$ (o $\lambda$) de un QD dentro de una cavidad acústica de 50--100 GHz. La referencia cuantitativa más cercana sigue siendo Spinnler (1.47 GHz, $g/\Omega=2\times10^{-3}$).
2. Hay **tres hechos de fabricación a favor de mi modelo de pilar:** cavidades GaAs/AlAs de 20--1000 GHz con $f\!\cdot\!Q$ de $10^{12}$--$10^{14}$, pilares con lado sub-100 nm, y un cristal fonónico de diamante con detalles de ~20 nm a 50--70 GHz.
3. **El acoplo grande no tiene un camino ya demostrado:** los polaritones y los TMD lo ofrecen pero en otros sistemas; los nitruros no tienen datos acústicos.
4. **Elastic**: el cálculo vectorial 3D (FEM) sigue siendo la prueba decisiva y no está disponible en `pcjhon`.
5. Si no hay un camino a $\lambda\gtrsim0.05$ con QD de GaAs, la plataforma para la memoria Raman de tres puntos queda en el régimen de F18 con $S\lesssim0.2$ ($F_h\lesssim0.6$ y Rabi de 90 GHz).
