# VB: verificacion independiente del modelo dispersivo (chequeo de borrado JPNS, doble carril)

Codigo propio en `verif/vb/` (no se leyo el codigo del autor hasta terminar `mydisp.py`). Unidades: us, rad/us. Sin commit/push.

## 0. Ejecucion
```
cd verif/vb
python t1.py          # validaciones analiticas (dispersivo propio)
python scan_my.py     # dispersivo propio, 4 puntos x 7 sigma  -> my_scan.json
python scan_au.py     # codigo del autor, mismos puntos + comparacion -> au_scan.json
python scan_disp2.py  # dispersivo propio con chi corregido (2*chi, ver 3) -> disp2_scan.json
python jc_spec.py     # espectro JC (desplazamientos por foton)
python scan_jc.py; python scan_jc_s1.py   # JC completo (~8 min, 4 nucleos) -> jc_scan.json, jc_scan_s1.json
```
Archivos: `mydisp.py` (modelo propio), `myjc.py` (JC: transmon 3 niveles, modos 3 niveles, `nm=4` verificado), `scan_*.py`.

## 1. Validaciones analiticas (mi implementacion)
| Prueba | Resultado |
|---|---|
| Sin ruido ni transmon, T=100: fidelidad | 1.0000000 |
| Solo perdida, poblacion de \|10> | 0.829223 vs e^{-100/534}=0.829223 |
| Coherencia \|<10\|rho\|01>\| = 0.5 e^{-T/T1} e^{-T gamma (1-rho)} (rho=0, .65, .95; T=100) | 0.063955/0.215538/0.377618 vs analitico 0.063955/0.215538/0.377618 (acuerdo 1e-8) |
| Selectividad pulso pi (sin ruido, Delta=20, FSR=3.3), sigma=1/3/8 | p_e(\|00>)=0.91/0.43/0.002; p_e(\|10>)=0.994/0.989/0.956 -> selectivo solo para sigma>~3 us |
| Delta=6, sigma=3: p_e(\|00>)=0.001, p_e(\|10>)=p_e(\|01>)=0.66 | el desajuste (chi1-chi2)/2 reduce el pase |

La fidelidad se maximiza sobre phi analiticamente (F(phi)=a0+a1 cos+a2 sin, comprobado contra malla).

## 2. Mi implementacion vs codigo del autor (`src/erasure_sim.py`, run con chis=(chi1,chi2) segun spec)
28 combinaciones (4 puntos x 7 sigma): **|dF_c| <= 1e-4, |dp_pass| <= 1e-4** (tambien F_c sin Z-comp identico a 4 cifras). El codigo del autor implementa correctamente la especificacion: orden tensorial (transmon,m1,m2), indices de Rz (idx 1 = \|01>), lectura con epsilon, renormalizacion, gq, operadores de desfase correlacionado. No hay error de codigo en el sentido de la especificacion.

Resultados (spec, chi_k=2pi g^2/Delta_k alpha/(Delta_k+alpha)); optimo sobre sigma en {1,1.5,2,3,4,6,8}:
| Punto (Delta,FSR,T,rho) | sigma_opt | F_c (mio=autor) | p_pass | F_c sin chequeo |
|---|---|---|---|---|
| 6, 3.3, 100, .65 | 2 | 0.818 | 0.637 | 0.674 |
| 20, 3.3, 100, .65 | 6 | 0.790 | 0.705 | 0.694 |
| 20, 1.0, 100, .65 | 6 | 0.792 | 0.717 | 0.694 |
| 6, 3.3, 50, .95 | 2 | 0.969 | 0.709 | 0.882 |

Observaciones criticas sobre el codigo (sin afectar los numeros): `erasure_sim.py:67` malla de phi de 129 puntos (error <~1e-4, cuantificado); `:78` el `SI` por defecto (T1q=28, T2q=15, Delta=10) NO es el de silicio de esta comparacion (solo afecta a `__main__`); `:34` firma `f(t,args)` deprecada en qutip>=5.5; `:49-50`, `:47` funciones muertas (`zfid`, `fid`); `design_run.py:23` llama `E.run(T,0.0,...)` (rho=0, ruido "blanco") a diferencia de rho=0.65/0.95 aqui; `design_run.py:7`, `design_map.py:10` definen chik con la misma formula (ver 3).

## 3. Hallazgo conceptual (no de codigo): chi por foton subestimado x2, y signo de alpha
Diagonalizando el JC (g/2pi=1.1, alpha=-185 MHz, `jc_spec.py`), el desplazamiento de la transicion g-e por foton del modo k es **2*chi_k**, con chi_k = g^2/Delta (alpha/(alpha-Delta)) [modos por debajo del transmon], no chi_k. Delta=6: JC split(10)-(00)=0.380 MHz, split(01)-(00)=0.255; 2chi=0.417/0.274 (modos por debajo) o 0.391/0.248 (por encima); chi solo = 0.195/0.124. Delta=20: JC 0.135/0.117 vs 2chi(debajo)=0.136/0.119. El H dispersivo del spec/autor, `(chi_bar - chi1 n1 - chi2 n2)|e><e|`, usa chi como desplazamiento TOTAL por foton, por lo que **las detunings (vacio y entre \|10>/\|01>) quedan a la mitad de las reales**. Ademas la formula alpha/(Delta+alpha) con alpha=+185 corresponde a modos por ENCIMA del transmon; para modos por debajo (spec del JC: w_k=w_q-Delta) es alpha/(alpha-Delta) (+24 % a Delta=20, +3 % a Delta=6).
Efecto: el modelo del autor SUBESTIMA el pase espurio de vacio y la selectividad requerida; sobreestima el sigma optimo en Delta=20.

## 4. JC completo vs dispersivo
JC: transmon 3 niveles, modos 3 niveles (nm=4 en Delta=6,sigma=3: F_c 0.7509 vs 0.7507, p 0.2021 en ambos -> convergido), acople g, anarmonicidad -185 MHz, pulso gaussiano de barrido, mismas disipaciones (1/T1q q, 2gq q+q como desfase, kappa, desfase correlacionado), lectura bare con error eps (f cuenta como g). Frecuencia del laser: maximiza min(p_e|10>, p_e|01>) sin ruido alrededor del punto medio de las dos transiciones (maximizar solo \|10> sesga hacia una de ellas; el laser optimo queda a <=0.03 rad/us del punto medio, ver `off`). Se corrieron ambas orientaciones (modos debajo/arriba); difieren <=0.01 en F_c.
Tabla (JC debajo | autor spec | dispersivo propio con 2chi_debajo), F_c / p_pass:

| Punto | sigma | JC | Autor (spec) | Disp. 2chi |
|---|---|---|---|---|
| 6,3.3,T100 | 1 | 0.809 / 0.627 | 0.752 / 0.817 | 0.828 / 0.649 |
| | 1.5 | 0.793 / 0.530 | 0.803 / 0.713 | 0.822 / 0.513 |
| | 2 | 0.793 / 0.410 | 0.818 / 0.637 | 0.815 / 0.368 |
| | 3 | 0.751 / 0.202 | 0.809 / 0.498 | 0.788 / 0.136 |
| 20,3.3,T100 | 1 | 0.740 / 0.882 | 0.696 / 0.953 | n/d |
| | 2 | 0.807 / 0.779 | 0.715 / 0.898 | 0.811 / 0.786 |
| | 3 | 0.812 / 0.740 | 0.743 / 0.835 | 0.815 / 0.748 |
| | 4 | 0.805 / 0.710 | 0.770 / 0.778 | 0.808 / 0.720 |
| | 6 | 0.790 / 0.645 | 0.790 / 0.705 | 0.794 / 0.657 |
| 20,1.0,T100 | 1 | 0.738 / 0.875 | 0.697 / 0.952 | n/d |
| | 3 | 0.805 / 0.750 | 0.750 / 0.831 | 0.815 / 0.763 |
| | 4 | 0.799 / 0.730 | 0.777 / 0.778 | 0.809 / 0.745 |
| | 6 | 0.786 / 0.688 | 0.792 / 0.717 | 0.795 / 0.709 |
| 6,3.3,T50,rho.95 | 1 | 0.954 / 0.699 | 0.931 / 0.867 | 0.976 / 0.724 |
| | 1.5 | 0.936 / 0.590 | 0.961 / 0.784 | 0.971 / 0.572 |
| | 2 | 0.939 / 0.457 | 0.969 / 0.709 | 0.965 / 0.409 |
| | 3 | 0.894 / 0.225 | 0.962 / 0.555 | 0.940 / 0.150 |

(n/d: ver `disp2_scan.json`; los sigma=4,6,8 completos estan en los JSON.) Con sigma=8 no se corrio el JC (p_pass<0.04 en Delta=6).

**sigma optimo (F_c maximo en la malla)**
| Punto | JC | Autor (spec) | Disp. 2chi |
|---|---|---|---|
| 6,3.3,T100 | 1 (F_c .809, p .63; borde de malla) | 2 (.818, .64) | 1 (.828, .65) |
| 20,3.3,T100 | 3 (.812, .74) | 6 (.790, .70) | 3 (.815, .75) |
| 20,1.0,T100 | 3 (.805, .75) | 6 (.792, .72) | 3 (.815, .76) |
| 6,3.3,T50,rho.95 | 1 (.954, .70; borde) | 2 (.969, .71) | 1 (.976, .72) |

Nota: en Delta=6 el optimo esta en el borde sigma=1 (no se exploro <1 us; ahi el pulso es corto y el JC real incluye fuga al nivel f).

## 5. Discrepancias y validez de la aproximacion dispersiva
- Autor vs mio: <=1e-4 (acuerdo exacto). Autor vs JC: F_c en el optimo de cada modelo: Delta=6: +0.009 (T100), +0.015 (T50); Delta=20: -0.020 (autor subestima, sigma_opt erroneo 6 vs 3). **p_pass del autor en Delta=6 a igual sigma esta sobreestimado hasta x2.4** (sigma=3: 0.50 vs JC 0.20; sigma=2: 0.64 vs 0.41); el acuerdo en el optimo (0.637 vs 0.627) es casual porque los sigma optimos difieren. A sigma fijo y Delta=20, F_c del autor queda 0.03-0.07 por debajo del JC (sigma<=4) y p_pass 0.05-0.12 por encima.
- Dispersivo con 2chi (misma estructura, chi corregido): vs JC, Delta=20: |dF_c|<=0.005, |dp|<=0.02 (sigma 2-6) -> valido. Delta=6 (g/Delta=0.18, FSR=3.3): |dF_c|<=0.03, dp_pass <=0.06 hasta sigma=2, ~0.07 relativo a 0.2 en sigma=3; hay efectos no dispersivos (acople efectivo entre modos via el transmon J~g^2/Delta ~0.2 MHz, no-linealidades de orden superior, desplazamiento de vacio ~ sum g^2/Delta_k distinto de chi_bar: JC vacio 0.317 MHz) -> solo cualitativo/semicuantitativo.
- JC debajo vs arriba: F_c igual <=0.01 (Delta=20) y <=0.004 (Delta=6).
- Truncacion de modos nm=3 vs 4: <=2e-4.
- No incluido: doble conteo menor de Purcell en JC (kp ya esta en kappa y el JC lo genera de forma natural durante el pulso; sesgo <~0.4 % en p, 3-12 us de pulso).

## 6. Veredicto: que se puede citar
1. **No citar** como cuantitativos los numeros del modelo dispersivo del autor (Delta=20: F_c=0.79, sigma*=6; Delta=6: p_pass=0.64, sigma*=2) con chi del spec: el H tiene chi/foton a la mitad de la fisica (factor 2 en detunings) y alpha/(Delta+alpha) corresponde a modos sobre el transmon.
2. Citables (JC completo, robustos a orientacion y truncacion): Delta=20, T=100, rho=.65: F_c=0.81 (FSR 3.3) / 0.805 (FSR 1.0), p_pass~0.74-0.75, sigma*=3 us; Delta=6 FSR 3.3: F_c~0.80-0.81, p_pass~0.63 a sigma~1 us (cota a sigma<=1; a sigma=2: 0.79/0.41); T=50, rho=.95, Delta=6: F_c~0.94-0.95, p_pass 0.46-0.70 segun sigma (0.954/0.70 a sigma=1). Incertidumbre sistematica estimada ~0.01 en F_c.
3. Mejora de F_c por el chequeo frente a no chequear (0.67-0.69 a T=100; 0.88 a T=50 rho=.95) se mantiene: +0.12 en Delta=20 y +0.13-0.14 en Delta=6 (T=100); +0.07 (T=50) segun JC.
4. Las cifras de la matriz de disenio (design_run/design_map) heredan el sesgo de chi; rehacerlas con 2chi y signo correcto, o contrastarlas con JC, antes de citarlas. La conclusion cualitativa (FSR=1.0 vs 3.3 casi irrelevante a Delta=20; Delta grande da p_pass mayor) es robusta.
