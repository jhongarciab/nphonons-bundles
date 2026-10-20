# Ventana de parámetros adimensionales y plataformas (16 oct 2026)

Idea (del autor): como Bin et al. (PRL 124, 053601), fijar la teoría en **parámetros adimensionales** ($\lambda/\omega_b$, $g/\omega_b$, $J/\omega_b$, $\kappa/\omega_b$, $\gamma_\varphi/\omega_b$, $\Omega/\omega_b$) y mostrar que **existen plataformas** con esos valores, sin atar el resultado a una sola.

## 1. Qué hace Bin (texto de arXiv:1907.12714, leído)
- Modelo: QD de dos niveles con acoplo longitudinal $\lambda\,\sigma^\dagger\sigma(b+b^\dagger)$ a un modo acústico $\omega_b$, láser a la banda lateral de orden $n$.
- Parámetros de sus figuras: $\lambda/\omega_b=0.03$ y $0.1$ con $\Omega/\omega_b=0.003$; $\lambda/\omega_b=0.03$ con $\Omega/\omega_b=0.8$ (régimen Mollow).
- Posicionamiento: «hemos considerado un sistema semiconductor con un QD acoplado a una nanocavidad acústica de THz; **nuestra propuesta no está limitada a esta arquitectura y podría implementarse o adaptarse en una variedad de plataformas**. Para nuestro diseño específico, el modelo teórico predice que la tecnología actual [12--15] ya debería...» (citas a cavidades fonónicas GHz--THz ya fabricadas y a acoplos QD--fonón medidos). Además observa que el Hamiltoniano reducido es análogo al de un **ion atrapado** [44].
- Es decir, **Bin justifica un rango de $\lambda/\omega_b$ por plataformas, no un diseño concreto**: el enfoque que propones.

## 2. Nuestra ventana en las mismas unidades
Del barrido F18/F13 (punto de trabajo $S\simeq0.5$, $F_h\approx0.84$ con E/S reales):
| parámetro | valor | comparación con Bin |
|---|---|---|
| acoplo transversal brillante--oscuro $g_s/\omega_b$ | $\gtrsim0.03$ (0.02 → $F_h\approx0.73$) | no existe en Bin |
| acoplo longitudinal máximo $\lambda_{\max}/\omega_b$ ($g\approx0.4$--$0.6\,\lambda_{\max}$) | $\gtrsim0.07$ | dentro del rango de Bin ($0.03$--$0.1$), en su extremo alto |
| intercambio $J/\omega_b$ | $0.1$--$0.2$ (brechas 0.35/0.6) | no existe en Bin |
| bombeo $\Omega/\omega_b$ | $\lesssim0.4$ (hasta 0.9 si $S\lesssim0.2$) | dentro del rango de Bin ($\le0.8$) |
| $\kappa/\omega_b$ | $10^{-3}$ (E/S con acoplador de $10\kappa$) | Bin: $\omega_b\gg\kappa\gg\gamma$ |
| $\gamma_\varphi/\omega_b$ | $\lesssim4\times10^{-4}$ (con $\chi\to1$; ver `memoria_teoria.tex`) | — |

## 3. Estado de cada plataforma en esos parámetros
`[FICHA]` = solo resumen/ficha/resultado de búsqueda; `[INFERIDO]` = aritmética propia.
| plataforma | $\lambda/\omega_b$ o $g/\omega_b$ demostrado | $\kappa/\omega_b$, coherencia | nota |
|---|---|---|---|
| **QD + cavidad fonónica** (Spinnler 2024, 1.47 GHz) | $g/\Omega=2\times10^{-3}$ | $Q=2.1\times10^3$ | `[TEXTO]`. Extrapolación a 50--100 GHz: $\lambda\approx0.006$--0.09 según área lateral/geometría (`RESULTADOS_A1_CONTROLES.md` §7--10) |
| **Circuito superconductor, ultrafuerte** (qubit de flujo + resonador) | $g/\omega\simeq0.12$ (Niemczyk 2010) y $0.72$--$1.34$ (Yoshihara) `[FICHA]` | $\kappa/\omega\sim10^{-3}$--$10^{-4}$ (de memoria) | Alcanza $\lambda/\omega\ge0.07$ **hoy**, pero el modo es un fotón de microondas, no un fonón. Un qubit de flujo se acopla con $\sigma_z$-like y la parte no uniforme viene de las distintas inductancias. El intercambio entre qubits $J$: **no verificado** |
| **Ion atrapado** (Rabi cuántico simulado) | acoplo efectivo tunable hasta el régimen ultrafuerte y profundo (Pedernales 2015, Lv 2018) `[FICHA]` | modo fonónico de coherencia larga, enfriamiento al estado fundamental (de memoria) | Es el análogo exacto que cita Bin; el modo es un fonón. Intercambio entre iones: **no verificado** |
| **Transmon + HBAR/SAW** (Yale 2017) | $g\approx2\pi\times300$ kHz a 6.65 GHz: $g/\omega\simeq4.5\times10^{-5}$ `[INFERIDO]` | cooperatividad 260, coherencia del fonón $>10\,\mu$s | acoplo transversal; lejos de $\lambda/\omega\ge0.07$ |
| **SiV en diamante** (Meesala 2018; Kuruma 2024) | susceptibilidad $\sim1$ PHz/deformación (orbital), $\sim100$ THz/deformación (espín); cristal fonónico con banda 50--70 GHz `[FICHA]` | $Q\sim10^5$ a GHz en cristales optomecánicos de diamante `[FICHA]` | sin cifra de $g/\omega$ |
| **Polaritones de microcavidad** (Sesin 2022) | $g_0$ de decenas de MHz a GHz `[FICHA]` | — | no son emisores de dos niveles |
| **Excitones interlayer WSe$_2$** (Ripin 2023) | réplicas de fonones de THz, acoplo «excepcionalmente fuerte» `[FICHA]` | — | acoplo grande; emisores de TMD |

## 4. Consecuencias
1. **El argumento al estilo de Bin es legítimo y honesto**, pero hay que decir cuáles plataformas tienen hoy $\lambda/\omega\ge0.07$: los circuitos superconductores y los iones atrapados (con un modo bosónico que no es acústico en el primer caso); los QD solo con confinamiento lateral de 20--30 nm o extrapolaciones.
2. El **modelo es genérico** (tres emisores con intercambio $J$ y acoplos longitudinales $\lambda_i$ desiguales a un modo bosónico): el título y el resumen pueden hablar de «un modo bosónico» y dejar al QD como un caso.
3. Lo que hace falta confirmar para circuitos e iones, antes de proponerlos: que pueden darse **(i) tres emisores casi idénticos con intercambio $J\sim0.1$--$0.2\,\omega_b$ y (ii) acoplos $\lambda_i$ no uniformes**, con $\kappa/\omega_b\sim10^{-3}$ y $\gamma_\varphi/\omega_b\lesssim4\times10^{-4}$. En circuitos, $J$ entre qubits de flujo y las inductancias; en iones, interacciones espín--espín mediadas por otros modos.
4. **Figura genérica propuesta:** $F$ y $F_h$ frente a $g_s/\omega_b$ (escala log) con marcadores de las plataformas y la banda de viabilidad (datos de F18; los marcadores de otras plataformas requieren re-simular con su $\kappa/\omega_b$ y $\gamma_\varphi/\omega_b$).
