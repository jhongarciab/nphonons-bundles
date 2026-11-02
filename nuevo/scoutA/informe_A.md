# Informe A — QED de cavidad de fonones con una molécula de QDs (átomo JC híbrido)

## 1. Resumen ejecutivo (veredicto)
**NO-GO como artículo principal.** El acoplo transversal g=δλ·J/S existe y es el esperado (verificado numéricamente, g=0.070 con δλ=0.14), y marca una diferencia cualitativa frente al emisor único (g²(0)<1 y Q<0 donde el único da g²≈1.8, Q>0). Pero el efecto es **moderado**, lejos de los umbrales de GO:
- Bombeo óptico, parámetros de Bin (γ=2e-4, κ=0.003): n̄≤0.11, g²(0)≥0.62, |Q|≤0.04 (el modo casi no se llena: ganancia ≤γ, pérdida κ).
- Mejor caso plausible con γ=1e-2 (requiere Purcell óptico, no es el γ de Bin) y κ=0.003: g²(0)=0.20, Q=−0.38, P(1)=0.49. Con Q_mec=1000 (κ=0.001): g²=0.145, Q=−0.56, P(1)=0.62. Caso optimista λ₂=−λ₁, κ=0.001, γ=1e-2: g²=0.071, Q=−0.65, P(1)=0.69. Ninguno cumple g²≲0.05, Q≲−0.7 ni P>0.8.
- Bloqueo convencional (bombeo mecánico del modo, átomo repuesto ópticamente): g²(0)≈0.10–0.13 (n̄≈0.03–0.06); limitado porque el átomo está excitado como máximo ~50 % del tiempo; además exige excitar mecánicamente un modo de ~THz (no es factible por IDT/SAW).
- Novedad: no encontré la combinación exacta en arXiv (búsqueda por API, ver §2), pero cada ingrediente tiene antecedentes (JC átomo–fonón, bloqueo con qubit, láser de fonones con QDs, micromáser de fonones). Con un efecto moderado, el giro nuevo no basta para PRA/PRB.
- Condición analítica: sí (g, resonancia d=±g−λ̄², modelo de escalera oscura CPT con acuerdo semicuantitativo), pero la propia analítica prueba que Q≥~−0.5 en el esquema de bombeo óptico (Poisson tipo x^n/(n!)²).

## 2. Novedad (búsqueda arXiv 2012–2026; API export.arxiv.org)
Nivel de lectura: [TEXTO]=intro/conclusión leídas del PDF; [RESUMEN]=solo abstract (API); [TÍTULO]=solo título/ref. de la API.
| id | Título | Autores/año/revista | Lectura | Relación |
|---|---|---|---|---|
| 1706.07777 | Collective enhancements in many-emitter phonon lasing | Droenner, Naumann, Kabuss, Carmele; PRA 96, 043805 (2017) | [TEXTO] | QDs excitados ópticamente en nanocavidad acústica; acoplo electrón-fonón **diagonal**, mapeo a Tavis–Cummings con corrimiento colectivo; distingue emisores no idénticos; NO usa moléculas ni acoplo transversal/Stark. Lo más cercano. |
| 1302.1855 | Nonlinear Quantum Optomechanics via Individual Intrinsic Two-Level Defects | Ramos, Sudhir, Stannigel, Zoller, Kippenberg; PRL 110, 193602 (2013) | [RESUMEN]+inicio | JC TLS–fonón en strong coupling, señales en espectro optomecánico; el átomo JC de fonones ya existe (defectos), no con excitones híbridos de QDM. |
| 1608.01548 | Phonon blockade in a nanomechanical resonator resonantly coupled to a qubit | PRA 94, 063853 (2016) | [RESUMEN] | Dos mecanismos de bloqueo (anarmonicidad e interferencia) en NAMR–qubit; frágil a la temperatura. |
| 0910.3066 | Qubit-induced phonon blockade as a signature of quantum behavior in NAMRs | PRA 82, 032101 (2010) (anterior a 2012, referencia histórica) | [RESUMEN] | Bloqueo de fonón inducido por qubit superconductor. |
| 1603.07546 | Method for observing robust and tunable phonon blockade in a NAMR coupled to a charge qubit | PRA 93, 063861 (2016) | [TÍTULO] | Qubit de carga (≈ DQD) con fonón: antecedente directo del átomo híbrido, aunque de electrón, no excitón. |
| 1506.08622 | Tunable multiphonon blockade in coupled nanomechanical resonators | PRA 93, 013808 (2016) | [TÍTULO] | Bloqueo multifonón (no atómico). |
| 1308.4213 | Nonclassical mechanical states in an optomechanical micromaser analogue | PRA 88, 053828 (2013) | [RESUMEN] | Límites sub-Poissonianos de osciladores mecánicos, ciclos «micromáser». |
| 2204.01605 | Phonon trapping states as a witness for ... phonon blockade in a hybrid micromaser system | EPJ Plus 137, 981 (2022) | [RESUMEN] | Estados de atrapamiento de fonones en micromáser optomecánico. |
| 1306.1786 | A phonon laser utilizing quantum-dot spin states | Khaetskii, Golovach, Hu, Žutić; PRL 111, 186601 (2013) | [RESUMEN]+inicio | Láser de fonones con espines de QD (no excitones). |
| 2205.01277 | Large Single-Phonon Optomechanical Coupling between QDs and Tightly Confined SAWs in the Quantum Regime | DeCrescent, Wang, ..., Silverman (NIST), 2022 | [RESUMEN]+inicio | Experimental: g0≈2π·1.2 MHz con QD único en SAW de 3.6 GHz; señala el régimen de transducción, no de JC fuerte. |
| 2011.03886 | Motional N-phonon bundle states of a trapped atom with clock transitions | Photonics Res. 9, 1289 (2021) | [TÍTULO] | Bundles de fonones en iones (línea Bin). |
| 2505.09906 | Resonant and Anti-resonant Exciton-Phonon Coupling in Quantum Dot Molecules | 2025 | [RESUMEN] | Relajación orbital asistida por fonones en QDM (fonones propagantes, no de cavidad). |
| 1910.08675 | Light emission properties in a DQD molecule immersed in a cavity: phonon-assisted tunneling | 2019 | [TÍTULO] | QDM en cavidad óptica; fonones como baño. |
| 1803.02000 | Indirect strong coupling regime between a QD and a nanocavity mediated by a mechanical resonator | 2018 | [TÍTULO] | QD–resonador mecánico en cadena optomecánica. |
| 2009.09871 | Simultaneous blockade of a photon, phonon and magnon induced by a two-level atom | PRA 101, 063838 (2020) | [TÍTULO] | Bloqueo con átomo de dos niveles. |
**Juicio:** no hallé en ~25 consultas (incluyendo 'unconventional phonon blockade', 'quantum dot molecule phonon', 'double quantum dot Jaynes-Cummings phonon', 'phonon bundle') un trabajo que combine QDM excitónica + acoplo transversal por δλ + sintonía Stark + estadística de fonones. La búsqueda por API no es exhaustiva (los títulos [TÍTULO] no se leyeron; Kabuss et al. PRL 2012 no apareció en arXiv). Veredicto de novedad: combinación probablemente nueva, ingredientes conocidos.

## 3. Modelo y prototipo [CÁLCULO]
H/ħ (marco rotante del láser, ω_b=1): H=Σ_i δ_i n_i + J(σ₁†σ₂+h.c.) + b†b + Σ λ_i n_i (b+b†) + Ω Σ(σ_i+σ_i†); δ_{1,2}=±ε/2−wL; estados {|0>,|X₁>,|X₂>}. **Biexcitón excluido**: energía de enlace de unos meV ≫ ω_b y Ω, el láser es resonante con la banda de un excitón. Decaimiento radiativo γ por punto, desfase γφ=4e-4, pérdida κ, T=0. Truncación Nb=16–24: g²=0.2936, n̄=0.1688 idénticos a 4 cifras (Nb=10/16/24). Conversión: con γ=2e-4 y τ_rad=1 ns, ω_b≈5e12 s⁻¹ (ω_b/2π≈0.8 THz, ħω≈3.3 meV) → S=ω_b ↔ J=0.5ω_b≈1.6 meV (rango 0.5–3 meV OK); Q=1/κ=200–333. Con ω_b/2π=100 GHz sería γ≈1.6e-3 y J≈0.2 meV.
**Comprobaciones:** (a) desdoblamiento de vacío: semi-ancho numérico 0.0699 / 0.0611 / 0.0429 para ε=0/0.5/0.8 (δλ=0.14) frente a g=δλJ/S=0.0700/0.0606/0.0420 (correcciones ~2 % por el desplazamiento de polarón λ̄²=0.005, centro del doblete en S/2−λ̄²). (b) Límites (γ=2e-3, Ω=0.002, d=0.06): con δλ=0 (λ₁=λ₂=0.14 o 0.07), con J=0.01 (dos emisores independientes) y con emisor único, n̄≲1e-4 y g²≫1 (el efecto desaparece: es genuinamente del JC híbrido). (c) Comparación del mejor sobre wL, γ=2e-3, κ=0.003: molécula Ω=0.004 g²=0.245, n̄=0.25, Q=−0.19; Ω=0.05 g²=0.82, n̄=0.41, Q=−0.07; emisor único Ω=0.05 g²=1.84, n̄=0.12, Q=+0.10; dos independientes Ω=0.05 g²=1.72, Q=+0.10.
**Mecanismo:** a ε=0 solo |+>=(|X₁>+|X₂>)/√2 es brillante, |−> oscuro; el láser resonante con |+>, JC |+,n>↔|−,n+1> y |0,n> forman un Λ con estado oscuro D_n∝g√(n+1)|0,n>−Ω_e|−,n+1> (Ω_e=√2Ω). Ganancia limitada por γ (el átomo debe decaer desde |−>), de ahí n̄≈(γ/κ)·O(1).

### Tablas (S=1, ε=0, J=0.5, λ=(0.14,0) salvo indicación, γφ=4e-4, Nb=16–18; barrido Ω∈{0.002..0.04}, d=wL−S/2∈[−0.3,0.3], criterio n̄>0.05)
| caso | mín g²(0) | (Ω,d,n̄) | mín Q | P(1) | máx P(2) |
|---|---|---|---|---|---|
| A γ=2e-4, κ=.003 | 0.62 | (0.005,−0.08,0.10) | −0.04 | 0.10 | 0.00 |
| B γ=1e-2, κ=.003 | 0.22 | (0.002,−0.08,0.17) | −0.38 (Ω=.01, n̄=.64) | 0.49 | 0.24 |
| C γ=1e-2, κ=.001 | 0.145 | (0.002,−0.08,0.38) | −0.56 (n̄=.76) | 0.62 | 0.36 |
| D λ=(.14,−.14), γ=2e-4, κ=.003 | 0.55 | (0.005,−0.15,0.10) | −0.05 | 0.10 | 0.00 |
| E λ=(.14,−.14), γ=1e-2, κ=.003 | 0.097 | (0.005,−0.15,0.44) | −0.47 | 0.52 | 0.18 |
| F λ=(.14,−.14), γ=1e-2, κ=.001 | 0.071 | (0.002,−0.15,0.40) | −0.65 (n̄=.88) | 0.69 | 0.37 |
Otros: γ=0.03, κ=.003 (Ω=0.02,d=−0.06): Q=−0.375, n̄=1.30, P(1)=0.45, P(2)=0.28 (no hay estado de Fock: P máx(1)≈0.5–0.7). Los mínimos de g² caen en d≈+g−λ̄² o −g−λ̄² (picos del doblete polaritónico: 0.06 / −0.075–−0.08 observados vs 0.065 / −0.075 predichos).
**Bloqueo convencional** (átomo reducido RWA {|0>,|−>,|+>}, repuesto óptico 0↔−, bombeo mecánico F=0.002 al polaritón inferior, κ=.003, g=0.07): g²(0)=0.112 (γ=2e-4, Ω_r=.003, n̄=.058), 0.130 (γ=2e-4, Ω_r=.02), 0.098 (γ=2e-3, Ω_r=.02), 0.197 (γ=1e-2, Ω_r=.05); población excitada del átomo ≈0.5–0.6. No se midió el espectro de emisión del modo; solo se verificó el doblete por diagonalización (sub-idea (i) espectro: no comprobado por espectro).

## 4. Analítica [CÁLCULO]
1. g=δλ·sinθ/2=δλJ/S, sinθ=2J/S, (n₁−n₂)=cosθτ_z+sinθτ_x. Verificado (arriba). Parte longitudinal común λ̄ → polarón, corrimiento −λ̄².
2. Resonancias de la excitación óptica: d=wL−S/2=±g−λ̄² (polaritones |±>≡(|+,0>±|−,1>)/√2).
3. Escalera oscura (rate_model.py): D_n; b|D_n>∝|D_{n−1}> exactamente; peso w_n=Ω_e²/(Ω_e²+g²(n+1)); trampa T=|−,0>. Tasas: D_n→D_{n+1} γw_n; D_n→D_{n−1} κ(n+w_n); D_0→T κw_0; T→D_0 γ. Comparación (d=0): (Ω,γ,κ) num n̄/g²/Q vs modelo: (.07,.01,.003) 1.40/.98/−.03 vs 1.72/.84/−.28; (.05,.01,.005) .885/.835/−.146 vs .948/.735/−.251; (.05,.002,.003) .407/.82/−.073 vs .444/.64/−.16; (.1,.002,.005) .166/1.25/.04 vs .367/.83/−.06; (.1,2e-4,.003) .070/1.01/.001 vs .066/.81/−.013; (.03,.03,.003) 1.555/.794/−.321 vs 1.555/.769/−.359. Acuerdo cuantitativo bueno solo para Ω pequeño; el modelo sobreestima la sub-Poissonianidad y falla a Ω mayor (ignora excursiones a estados brillantes). Límite del modelo: P(n)∝x^n/(n!)² → Q→−1/2. No hay una condición cerrada de «Fock» porque no existe.

## 5. Limitaciones y riesgos
- Sin biexcitón, colisión de decaimientos colectivos (sub/superradiancia) ni acoplo óptico distinto por punto (dipolos iguales); decaimientos independientes. T=0 (a ω_b/2π≳0.1 THz razonable). Acoplos λ del documento del grupo, no de un cálculo microscópico; δλ=0.14 (λ₂=0) y λ₂=−λ₁ son extremos optimistas.
- γ=1e-2 requiere Purcell óptico (τ≈20 ps para ω_b≈5e12); κ=0.001 ↔ Q=1000 a THz es dudoso. Q de nanocavidades de fonones THz (Q~10⁴–10⁵ citado en 1706.07777 intro para superredes, [TEXTO] de ese artículo).
- Solo escaneo a Nb≤24; grids gruesos (Ω×d) con mínimos en el borde inferior de Ω en algunos mapas (Ω≥0.002); ninguna optimización global ni ε≠0 explorada a fondo (solo ε=0 en barridos; ε=0.5,0.8 solo en g).
- Sub-ideas (ii) estados de Fock/micromáser y (iii) correlaciones cruzadas: (ii) evaluada, sin Fock; (iii) no explorada.
- Un posible artículo menor (PRA) «átomo JC de fonones con excitones híbridos» podría defender g²≈0.1–0.2 y el patrón de dobletes, pero no cumple tu criterio de GO.

## 6. Código (carpeta /home/user/nphonons-bundles/nuevo/scoutA/)
`qdm.py` (modelo; `ss(**kw)`), `scan3.py/scan4.py` (mapas Ω×d; guardan .npz), `scan5.py` (fino, convergencia, límites), `scan6.py` (tabla A–F), `cmp1.py` (comparación emisor único/independientes), `chk_split.py` (g), `rate_model.py`+`cmp_rate.py` (analítica), `blockade.py` (bloqueo convencional), `q.py/ids.py` (arXiv). Uso: `python3 scan6.py > scan6.out` (≈6 min, 1 proceso). Salidas: scan2.out, scan4.out, scan6.out, cmp_rate.out. PDFs en pdf/.
