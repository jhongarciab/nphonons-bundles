# WP1: consolidación de la teoría de bundles de N fonones (emisor de dos niveles, acoplo longitudinal)

Directorio: `/home/user/nphonons-bundles/nuevo/bundles/` (src/, tests/, data/). Unidades ω_b = 1. Parámetros base: λ=0.14, Ω=0.1, κ=0.003, γ=2e-4, γφ=4e-4, n=2, maestra vestida, marco de polarón, δ_L=δ_res(n,λ,Ω). "Episodio" = fonones entre dos retornos de la cavidad al vacío. Pureza P_n = fracción de episodios con exactamente n fonones. Nada se ha dado por bueno del trabajo previo: todo se recalculó con código nuevo (que reproduce las cifras del explorador donde estas eran correctas: P_ep=0.93814, flujo 3.3633e-4, ciclos [0.033 0.057 0.805 0.070 0.032]).

## 1. Qué se hizo

| Entregable | Archivo |
|---|---|
| A. Biblioteca (H, maestras lab/polarón vestida/local, d_mk, ε_n, δ_res, Γ_j, flujo y pureza cerrados, cota, Ω_max, κ*, distribuciones exactas por ciclo/episodio, flujo exacto, modelo de tasas) | `src/bundles_core.py` |
| MC propio (H_eff constante, evolución exacta entre saltos, 400 trayectorias a la vez) y agrupación por τ_c | `src/bundles_mc.py` |
| B. Pruebas (14, todas OK; `python3 tests/run_tests.py`, ~8 s) | `tests/run_tests.py` |
| C. Validación MC | `src/run_C.py` → `data/C_mc.npz`; `src/run_C2.py` → `data/C2_cycles_local.json` |
| D1–D7 | `src/run_D1.py … run_D7.py`, `run_D2b.py` → `data/D1_spectrum.npz, D2_n{2,3}_lam{0.1,0.14,0.2}.npz, D2b_rate_*.npz, D3_pareto.npz, D4_local_vs_dressed.npz, D5_floor.npz, D6_trajectories.npz, D7_dephasing.npz` (metadatos en el campo `meta`, JSON) |
| Convergencia / resonancia / tablas | `data/convergence.txt`, `data/resonance.txt`, `data/analysis.txt`, `data/tablas.md` (`src/analyze.py`, `src/mk_tables.py`) |

Mallas: D2/D3 20×20 por (n, λ) (Ω∈[0.02,0.3], κ∈[3e-4,1e-2], log), 6 mallas exactas = 2400 puntos + 120×120 cerrada; D1 1126 puntos en δ_L; MC con semillas fijas (1000+i, 5000+i, 77+n). Todo el cómputo cabe en ~10 min con 2 procesos (BLAS a 1 hilo; el exacto cuesta ~15 ms/punto).

Método exacto: Liouvilliano denso (dim 4N²), estado estacionario, y distribución de conteo por renovación: P(k)=Tr[E (A⁻¹C)^(k−1) A⁻¹ r_s] con un único LU (sin FFT). Convergencia en N (`data/convergence.txt`): con N=n+6, |ΔP|<1e-6 y |ΔJ|/J<2e-5 en los 6 casos probados (n=2,3,4; incluidos Ω=0.4 y κ=3e-4); con N=n+4, 1e-3..1e-2. Σ_k P(k)=1 a <3e-6 (K=2n+6) salvo en la esquina coherente (Ω=0.3, κ=3e-4: la cola k>K pesa 2.5 %; P_n no se ve afectada, es un coeficiente exacto).

## 2. Errores y discrepancias en el código/informe previo (scoutB)

1. **Signo de la maestra "local" en polarón** (`bl.py:55`, `bj=b if dressed else b-lam*nx`). Con el H_p del propio código (σ†D+σD†, D=exp[−λ(b†−b)]) el salto local de laboratorio se transforma en **b + λσ†σ**. Con el signo previo la maestra "local" NO coincide con el laboratorio: ΔP(k) hasta 1.6e-3, Δflujo 0.19 % (P_loc(2)=0.6080 vs 0.6095); con el signo corregido la coincidencia es 7e-11 (test `test_lab_equals_polaron_local` y `test_local_sign_regression`). Era un error pequeño (el informe decía "coinciden"), pero tras corregirlo la nube κλ²/γ se confirma con 0.4 % (ver §5.6). Además, MC en el marco de laboratorio con la maestra local (independiente del polarón) da P_cyc(2)=0.615±0.004 vs exacta 0.6095.
2. **Fórmula cerrada subestima la impureza 10–25 % (sesgo sistemático)** (`rate.py:83`, `P_closed = e^{−λ²}(1−mer)(1−off)`). Falta la pérdida de Franck-Condon de los decaimientos a mitad de cascada que no se re-excitan, Σ_j p_j(1−r_j)(1−d_jj²) (p. ej. n=2: ≈0.0066 de la impureza 0.062). Con el término añadido (`P_closed` v2; la v1 queda disponible con `fcmid=False`) la mediana del error de la impureza baja de 12–29 % a 2–10 % en la región de validez (tabla §3). El informe previo daba "error mediano de la impureza 15 %"; en mi malla más amplia la v1 da 13–23 %.
3. **Modelo de tasas con desfase inducido** (`rate.py:22-25`): usa γφ(Ωd/Δ)² fuera de resonancia, que no incluye κ ni γ y no coincide con las colas lorentzianas de la fórmula cerrada (ε_off). Sustituido por una única tasa lorentziana Raman 2Ω²d²γ_c/(Δ²+γ_c²) para TODOS los pares (`rate_model`): error de P ≤0.4 % (medio) y mediana de impureza 2–6 % en la región de validez, frente a 0.5–0.8 % de error en P del modelo previo (base: exacta 0.9381, lorentz 0.9385, previo 0.9434).
4. **Ω_max con tope silencioso** (`rate.py:91`, `else 0.5`): devuelve 0.5 cuando P≥X hasta Ω=0.5, fuera del dominio de validez (Ω≲0.3). Ahora devuelve `inf` (más allá de Omcap) o `nan` (inalcanzable) y exige Ω≤0.3 al graficar. El frente de Pareto n=3 previo incluía un punto a Ω=0.5 con "cerrada 0.747 / exacta 0.844" (`final.py:30` rejilla hasta 0.5): se restringió a Ω≤0.3.
5. **κ\* no reproducible**: el informe da κ\*(n=2)=0.0043 (λ=0.14), 0.0028 (λ=0.1) y n=3 ≈6e-4. Con la fórmula previa (v1) a Ω=0.1 obtengo 0.0081 (λ=0.14) y 0.0054 (λ=0.1); n=3: 0.0013/0.0006; con v2: 0.0086 / 0.0057 (n=2). El máximo EXACTO de P_2 en κ (Ω=0.1) es 0.0089 (λ=0.14, interpolado de D5) y 0.0061 (λ=0.1). κ\* depende de Ω (crece con Ω: λ=0.14, n=2: 0.0071, 0.0086, 0.0107 para Ω=0.05, 0.1, 0.2). Los valores del informe previo no se pueden citar.
6. **"Cota dura e^{−λ²}"**: es cota dura para la pureza *por ciclos* (comprobado en `test_floor_bounds`), pero por episodios el piso exacto de conteo es 1/(2−e^{−λ²}) ≥ e^{−λ²} (diferencia O(λ⁴): 4e-4 en λ=0.14, 7e-3 en λ=0.3). Ningún punto exacto viola ninguno de los dos. Más importante: el máximo exacto sobre κ queda por DEBAJO del piso (ver §5.5), el piso no es ajustado.
7. **Definición de episodio**: los saltos "1→0" y "≥2" se separan con proyectores sobre el ket y el bra (`bl.py:81-83`, `episodes_pol`), dejando los términos cruzados κ(b_e ρ b_2†+h.c.) en la evolución sin conteo. La dinámica total es la física y la traza se conserva (el término cruzado tiene traza nula), pero no es un desenrollado estándar. La comprobación independiente es el MC (§5.7): tras corregir el efecto de la ventana, el MC reproduce la pureza por episodios. Además, un episodio solo termina con un salto 1→0 del modo, de modo que un retorno "silencioso" al vacío por un decaimiento Franck-Condon (x,1→g,0) no cierra el episodio (efecto O(p_1λ²)).
8. Menores: `bl.py:27` (`np.fft.fft` sobrante, inocuo); `final.py:41` usa N=8 (n=2) y 9 (n=3), es decir n+6 y n+6, correcto (mi convergencia: N=n+6 basta); `maps.py:41-45` (expresión basura en `rows`, inocua). Con BLAS multihilo (por defecto) el LU de 256×256 tarda 0.67 s frente a 16 ms con 1 hilo (por solve): es la causa de los "43 s por punto" de `res_t8.txt`.
9. Mallas del informe previo: 8×8 (n=2) y 6×6 (n=3) con Ω≤0.25/0.4 y κ≥6e-4. Su "error máx 10 %/13 %" no es extrapolable: en mallas 20×20 con κ desde 3e-4 y Ω hasta 0.3 el error máximo en P es 18–22 % (n=2) y 6–21 % (n=3), concentrado en κ≲1e-3 (§4).

## 3. Tablas de comparación analítico–exacto (episodios; malla 20×20 por fila)

Región de tasas: ξ=Ω_n/γ_c(0)<0.5 (Ω_n=Ω|d_n0|, γ_c(0)=(nκ+γ+γφ)/2). "v1" = fórmula previa.

| n | lam | region | pts | P cerrada: err rel max / medio [%] | % malla <5 / <10 % | impureza 1-P cerrada: mediana / media [%] | idem formula previa (v1) mediana [%] | flujo cerrado: medio / max [%] | modelo de tasas: P err medio [%] | impureza mediana [%] | flujo medio [%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.1 | toda | 400 | 22.4 / 2.36 | 86 / 92 | 11 / 11 | 19 | 5.5 / 51 | 1.17 | 5 | 3.7 |
| 2 | 0.1 | xi<0.5 | 282 | 4.2 / 0.74 | 100 / 100 | 10 / 10 | 21 | 1.6 / 11 | 0.57 | 5 | 2.5 |
| 2 | 0.1 | P_ex>=0.85 & xi<0.5 | 252 | 2.6 / 0.52 | 100 / 100 | 10 / 10 | 22 | 1.3 / 6 | 0.34 | 4 | 2.0 |
| 2 | 0.14 | toda | 400 | 20.1 / 3.18 | 78 / 88 | 10 / 11 | 17 | 6.8 / 41 | 1.27 | 4 | 3.3 |
| 2 | 0.14 | xi<0.5 | 203 | 2.4 / 0.54 | 100 / 100 | 7 / 8 | 18 | 1.0 / 3 | 0.48 | 4 | 1.9 |
| 2 | 0.14 | P_ex>=0.85 & xi<0.5 | 186 | 1.9 / 0.46 | 100 / 100 | 7 / 8 | 18 | 1.0 / 3 | 0.33 | 3 | 1.6 |
| 2 | 0.2 | toda | 400 | 18.1 / 4.04 | 70 / 84 | 9 / 10 | 13 | 8.7 / 39 | 1.25 | 3 | 3.0 |
| 2 | 0.2 | xi<0.5 | 116 | 1.0 / 0.27 | 100 / 100 | 4 / 4 | 12 | 1.0 / 3 | 0.22 | 2 | 0.7 |
| 2 | 0.2 | P_ex>=0.85 & xi<0.5 | 114 | 1.0 / 0.27 | 100 / 100 | 4 / 4 | 12 | 1.0 / 3 | 0.20 | 2 | 0.7 |
| 3 | 0.1 | toda | 400 | 6.0 / 0.65 | 99 / 100 | 3 / 6 | 11 | 6.8 / 47 | 0.65 | 4 | 2.5 |
| 3 | 0.1 | xi<0.5 | 400 | 6.0 / 0.65 | 99 / 100 | 3 / 6 | 11 | 6.8 / 47 | 0.65 | 4 | 2.5 |
| 3 | 0.1 | P_ex>=0.85 & xi<0.5 | 262 | 4.1 / 0.53 | 100 / 100 | 6 / 8 | 26 | 2.7 / 36 | 0.43 | 6 | 2.1 |
| 3 | 0.14 | toda | 400 | 15.2 / 1.22 | 93 / 100 | 6 / 8 | 23 | 4.2 / 26 | 0.79 | 3 | 2.5 |
| 3 | 0.14 | xi<0.5 | 400 | 15.2 / 1.22 | 93 / 100 | 6 / 8 | 23 | 4.2 / 26 | 0.79 | 3 | 2.5 |
| 3 | 0.14 | P_ex>=0.85 & xi<0.5 | 286 | 3.3 / 0.59 | 100 / 100 | 6 / 7 | 29 | 2.1 / 10 | 0.35 | 3 | 1.5 |
| 3 | 0.2 | toda | 400 | 20.6 / 1.87 | 89 / 97 | 7 / 7 | 22 | 4.4 / 31 | 1.21 | 3 | 2.8 |
| 3 | 0.2 | xi<0.5 | 368 | 7.9 / 1.27 | 95 / 100 | 6 / 7 | 23 | 3.8 / 26 | 1.05 | 3 | 2.7 |
| 3 | 0.2 | P_ex>=0.85 & xi<0.5 | 267 | 2.1 / 0.57 | 100 / 100 | 6 / 6 | 22 | 1.6 / 6 | 0.33 | 2 | 1.5 |

Lectura: en la región (ξ<0.5 y P_exacta≥0.85) la pureza cerrada v2 tiene error ≤4.2 % (100 % de la malla <5 %), el flujo cerrado ~1–2.7 % de media, la impureza mediana 3–10 % (v1: 12–29 %); el modelo de tasas lorentziano, que es un sistema lineal de 2M estados, error en P ≤0.4 % y mediana de impureza 2–6 %. En toda la malla el error de P llega a 22 % (n=2, κ=3e-4, Ω grande) y solo 70–86 % de los puntos están <5 %. El error relativo de la impureza es la métrica honesta: P=0.94–0.98 oculta errores de 10 %.

**Resonancia (`data/resonance.txt`, máximo del flujo exacto vs δ_res; γ_c=(nκ+γ+γφ)/2):** n=2, λ=0.14: error 0.001, 0.011, 0.14, 0.70, 2.2 γ_c para Ω=0.05, 0.1, 0.2, 0.3, 0.4 (3.7e-6, 3.5e-5, 4.6e-4, 2.3e-3, 7.4e-3 en δ); n=3: 0.00, 0.002, 0.026, 0.13, 0.42 γ_c. Bin-III (−√(n²−4Ω²)+λ²) es mejor solo para Ω≥0.3 (n=2: 0.56 vs 0.70 γ_c a Ω=0.3, 1.0 vs 2.2 a 0.4). Para λ→0, ε_n=2Ω²/n=n−√(n²−4Ω²)+O(Ω⁴) (test). D1: picos de n=1..4 en δ_L=−0.9605, −1.9702, −2.9737, −3.9754, error 5e-4 (=paso de rejilla; n=1), <5e-7 (n=2,3), 7e-9 (n=4).

## 4. Límites de validez (con evidencia)

- **Ω**: δ_res (2.º orden) válida para Ω≲0.2 (n=2) y Ω≲0.3 (n=3) (error <0.15 γ_c). Las fórmulas de tasas requieren régimen sobreamortiguado ξ=Ω_n/γ_c(0)≲0.5; en D2 la región ξ≥0.5 (n=2: 30–70 % de la malla) tiene error de P hasta 22 %, de flujo hasta 51 %.
- **κ**: la cerrada falla para κ≲1e-3 (λ=0.14: n=2 κ≲1.3e-3; n=3 κ≲7.5e-4), donde la fusión satura (r_j→1) y P_exacta<0.85: en D2 solo 10–55 % de la fila tiene error <5 % para κ=3e-4..7.5e-4 (n=2) y 55–95 % para κ=3e-4..6e-4 (n=3). Para κ≥1.3e-3 (n=2) y κ≥7.5e-4 (n=3) el error de P es <5 % en el 100 % de la malla con Ω≤0.3.
- **γ**: a γ grande (≥5e-4, κ=0.003) la cerrada subestima la impureza (D7: γ=1.6e-3, P exacta 0.764 vs 0.815; impureza −22 %); el mapa (γ,γφ) en D7 tiene impureza con error medio 18 %.
- **λ**: probado λ=0.1, 0.14, 0.2 (y 0.05–0.3 en D5). Para n=3 el error es menor a λ pequeño. n=4 no es usable con los parámetros base: en la línea n=4, P_4=0.185 y el 78 % de los episodios son de 1 fonón (flujo de línea 1.6e-7, comparable al fondo de fonones únicos entre líneas ~1e-7, D1).
- **n**: n=2,3 validados a fondo; n=1 (picos y distribución) en D1; n=4 solo posiciones.
- **Dominio del modelo**: T=0, un solo modo, maestra vestida (b sin componente DC) y Raman resonante; sin baño continuo de fonones, sin inhomogeneidad. La maestra vestida es la que se valida contra el MC, pero el MC usa la misma maestra, así que valida la estadística de conteo y no la propia maestra.

## 5. Afirmaciones citables (con respaldo)

1. **Resonancia**: δ_L=λ²−n+ε_n, ε_n=−Ω²Σ_{m≠n}d_m0²/(m−n)+Ω²Σ_{k≥1}d_nk²/k≈2Ω²/n, reproduce el máximo del flujo exacto con error <0.15 γ_c para Ω≤0.2 (n=2) y Ω≤0.3 (n=3) y recupera Bin-III hasta O(Ω⁴) cuando λ→0 (`data/resonance.txt`, tests).
2. **Flujo cerrado** J=(n+λ²)/(1/Γ_0+1/γ), Γ_0=4Ω²d_n0²/(nκ+γ+γφ): error medio 1.0–2.7 % y ≤3–11 % en la región ξ<0.5 (y P≥0.85); 4–9 % de media y hasta 51 % en toda la malla (`tablas.md`). El MC reproduce el flujo exacto: |J_MC/J−1| medio 0.9 % en 16 puntos (z medio 1.0).
3. **Pureza por episodios, cerrada (v2)**: error ≤4.2 % en P (100 % de puntos <5 %) en la región ξ<0.5 & P≥0.85 para n=2,3 y λ=0.1,0.14,0.2; error de la impureza 1−P: mediana 3–10 %. Falla (hasta 22 % en P) para κ≲1e-3 y Ω grande. La fórmula previa subestimaba la impureza 12–29 %.
4. **Modelo de tasas lorentziano** (`rate_model`): P con error ≤0.4 % (medio) y impureza mediana 2–6 % en la región de validez; reproduce las colas lorentzianas y el efecto del desfase en una sola expresión.
5. **Piso de Franck-Condon**: la pureza máxima exacta sobre κ (Ω=0.1 n=2; Ω=0.2 n=3) es: n=2: 0.9910, 0.9815, 0.9675, 0.9491, 0.9019 para λ=0.05, 0.1, 0.15, 0.2, 0.3 (κ\*=0.0026, 0.0061, 0.0096, 0.0129, 0.0189), frente al piso e^{−λ²} = 0.9975, 0.9900, 0.9778, 0.9608, 0.9139; la cerrada (v2) da el máximo con error de impureza medio 4.9 % (máx. 7 %) en n=2 y mediana 4.7 % (máx. 21 % en λ=0.05, óptimo en el borde κ=3e-4) en n=3, y κ\* dentro de ~10 %. Para n=2, P_n^max está 0.65–1.3 % por debajo del piso (impureza 3.6× la del piso en λ=0.05, 1.14× en λ=0.3); para n=3, 3.3–4.5 % por debajo. Por ciclos el máximo es mucho menor (n=2: 0.78–0.92).
6. **Maestra local vs vestida** (n=2, Ω=0.1; D4, `data/analysis.txt`): P_cyc(κ=0.003) = 0.805 (vestida; MC 0.803±0.004) vs 0.609 (local; MC en laboratorio 0.615±0.004). El exceso de fonones por ciclo (local−vestida) = κλ²/γ con razón medida/predicha 0.996 para κ≤5.5e-3 (0.94 a κ=0.0145); n=3 (Ω=0.2): 1.00–1.08. P_loc≈P_vest·e^{−κλ²/γ} con error mediano 2 % (n=2) / 4 % (n=3), pero hasta 58–78 % para κλ²/γ>1. Tendencia con κ: vestida crece monótonamente (0.40→0.92 por ciclos); local sube hasta κ≈2e-3 (0.61) y cae después (0.38 en κ=0.0145). Por episodios (clasificación pre-salto, local) P_2: 0.72 vs 0.94 (κ=0.003). La elección de maestra cambia las conclusiones y hay que declararla; no he derivado la maestra vestida a partir de un baño: es un supuesto (válido si κ≪λ²ω_b y κ≪ω_b).
7. **Validación operativa por MC (400 trayectorias/punto, 16 puntos; `tablas.md` §3)**: con agrupación por huecos <τ_c, la pureza MC depende fuertemente de τ_c y **no** coincide con la de episodios salvo κ/γ≳500: a τ_c=2/κ P_MC≈0.62–0.75 (se parte el 13.5 % de los bundles de n=2: e^(−κτ_c)), a 5/κ 0.53–0.95 y a 10/κ 0.29–0.96 (se fusionan bundles consecutivos: el hueco entre bundles ≈Exp(γ)+Exp(Γ_0)+Exp(nκ) es <τ_c con probabilidad q∼γτ_c). El mejor τ_c da P_MC−P_ep = −0.156 (base, 5/κ), −0.024 (κ/γ=50), −0.030 (κ/γ=150), −0.0014 (κ/γ=500, n=2: 0.9641±0.0020 vs 0.9655) y −0.007 (n=3, κ/γ=150). Una predicción operativa P_op(τ_c)=P_ep(1−q)(1−s)/(1+s), con s=1−Π_j(1−e^{−jκτ_c}) (partición) y q=CDF hipoexponencial (fusión), reproduce el MC con |z| medio 1.8 y diferencia absoluta media 0.0075 (máx. 0.031; 8 de 48 valores >3σ, 15 >2σ) usando solo cantidades analíticas. Conclusión: la pureza por episodios es la pureza intrínseca de la cascada; la pureza operativa por ventana temporal necesita κ/γ≫100 (τ_c≈5–10/κ) para igualarla, y en los parámetros base (κ/γ=15) está limitada a ≈0.78–0.80 (n=2) por coincidencias accidentales. Las tres cifras de τ_c deben citarse juntas.
8. **Estadística por ciclos validada por MC**: P(k=0..5) por ciclo (exacta vs MC, 19 000 ciclos) coinciden con |z|<3.1 en 5 configuraciones (vestida κ=0.003 y 0.01; local en laboratorio y en polarón, κ=0.003 y 0.01). La pureza por ciclos (0.805) mezcla particiones por decaimientos a mitad de cascada (p_2=0.032, p_1=0.062).
9. **Rol del desfase** (D7, n=2): el desfase γφ apenas cambia P (0.9378→0.9407 para γφ de 1e-6 a 7e-3) pero reduce el flujo al ensanchar la línea (3.38e-4→3.05e-4); el decaimiento γ domina la pureza: P=0.972 (γ=2e-5), 0.942 (1.8e-4), 0.764 (1.6e-3), y el flujo crece ∝γ. La fórmula cerrada a γ grande subestima la impureza hasta 22 %.
10. **Espectro** (D1): líneas n=1..4 con flujo 1.9e-4, 3.4e-4, 1.5e-5, 1.6e-7 y pureza (fracción de episodios con n fonones) 0.947, 0.938, 0.939, 0.185. Fondo entre líneas 7e-8..4e-6 (fonones únicos, Franck-Condon y colas), portadora (δ_L=λ²) 2.9e-6 con 99 % de episodios de 1 fonón.
11. **Diseño**: Ω_max(κ;P≥0.95) (cerrada, λ=0.14, n=2, γ=2e-4, γφ=4e-4): 0.042 (κ=0.002), 0.085 (0.003), 0.22 (0.005), sin límite (≥0.01, hasta 0.3); para P≥0.90: 0.109, 0.36, ∞. n=3 (λ=0.14): P≥0.95 solo es alcanzable para Ω≲0.05 (máximo cerrado 0.9515 en κ=0.0015; 0.9497 a Ω=0.1; 0.9449 a Ω=0.2; el exacto a Ω=0.2 y λ=0.15 es 0.9397), y P≥0.90 se cumple para todo Ω≤0.3 con κ≥0.002. Frente de Pareto (D3, λ=0.14, n=2; cerrada/exacta): P=0.9750/0.9762 con J=4.3e-5, 0.9720/0.9700 con J=2.9e-4, 0.9699/0.9640 con J=3.4e-4 (Ω≤0.17: |ΔP|≤0.006); falla en Ω=0.3, κ≲1.5e-3 (0.82/0.85 y 0.44/0.55).
12. **Convergencia**: N=n+6 da |ΔP|<1e-6 en n=2,3,4 (`data/convergence.txt`).

**D6** (`data/D6_trajectories.npz`): 8 trayectorias de T=2×10⁴ por n con tiempos de fonón (canal 0), foto-decaimiento radiativo (1) y desfase (2); `n{n}_rep` marca la de nº de fonones más cercano a la media. Con los parámetros base el registro es escaso por construcción (flujo×T = 3.7, 6.7, 3.3 fonones para n=1 (Ω=0.015), n=2 (Ω=0.1), n=3 (Ω=0.3, κ=0.0015)); 2 de las 8 trayectorias de n=3 no tienen ningún evento, y para n=3 conviene usar T mayor o la trayectoria con 9 eventos (índice 3).

## 6. Tabla MC (C): P_ep exacta, P cerrada v2 y P_MC(τ_c)

| n | Om | kap | gam | kap/gam | P_ep (exacta) | P_cerr (v2) | P_MC tau=2/k | 5/k | 10/k | pred. operativa 2/k, 5/k, 10/k | J_MC / J_exacto |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.1 | 0.003 | 2e-04 | 15 | 0.9381 | 0.9442 | 0.6988±0.0039 | 0.7824±0.0040 | 0.5879±0.0057 | 0.697, 0.796, 0.599 | 0.996±0.007 |
| 2 | 0.05 | 0.003 | 2e-04 | 15 | 0.9607 | 0.9636 | 0.7239±0.0044 | 0.8900±0.0034 | 0.7861±0.0049 | 0.726, 0.900, 0.793 | 1.009±0.008 |
| 2 | 0.2 | 0.003 | 2e-04 | 15 | 0.9160 | 0.9163 | 0.6793±0.0038 | 0.7043±0.0044 | 0.5129±0.0058 | 0.656, 0.700, 0.509 | 1.007±0.008 |
| 2 | 0.1 | 0.006 | 2e-04 | 30 | 0.9674 | 0.9700 | 0.7377±0.0038 | 0.9190±0.0026 | 0.8593±0.0035 | 0.734, 0.928, 0.870 | 1.001±0.008 |
| 2 | 0.1 | 0.01 | 2e-04 | 50 | 0.9704 | 0.9719 | 0.7480±0.0040 | 0.9467±0.0022 | 0.9385±0.0024 | 0.738, 0.951, 0.943 | 1.012±0.008 |
| 2 | 0.1 | 0.0015 | 2e-04 | 8 | 0.8671 | 0.8624 | 0.6150±0.0044 | 0.5301±0.0060 | 0.2868±0.0078 | 0.587, 0.520, 0.271 | 0.996±0.007 |
| 2 | 0.1 | 0.003 | 2e-05 | 150 | 0.9724 | 0.9762 | 0.7429±0.0043 | 0.9423±0.0025 | 0.9249±0.0029 | 0.739, 0.944, 0.928 | 1.012±0.011 |
| 2 | 0.1 | 0.01 | 2e-05 | 500 | 0.9655 | 0.9750 | 0.7397±0.0043 | 0.9549±0.0022 | 0.9641±0.0020 | 0.735, 0.952, 0.963 | 0.985±0.010 |
| 3 | 0.2 | 0.0015 | 2e-04 | 8 | 0.9336 | 0.9409 | 0.6893±0.0046 | 0.8833±0.0035 | 0.8285±0.0043 | 0.685, 0.894, 0.840 | 0.997±0.009 |
| 3 | 0.3 | 0.002 | 2e-04 | 10 | 0.9297 | 0.9390 | 0.6857±0.0046 | 0.8845±0.0035 | 0.8325±0.0043 | 0.682, 0.890, 0.833 | 0.998±0.009 |
| 3 | 0.4 | 0.003 | 2e-04 | 15 | 0.9142 | 0.9314 | 0.6763±0.0045 | 0.8806±0.0034 | 0.8578±0.0038 | 0.672, 0.886, 0.856 | 1.009±0.009 |
| 3 | 0.2 | 0.003 | 2e-04 | 15 | 0.9342 | 0.9359 | 0.6918±0.0045 | 0.9168±0.0029 | 0.9140±0.0029 | 0.688, 0.918, 0.918 | 1.010±0.009 |
| 3 | 0.3 | 0.0015 | 2e-04 | 8 | 0.9145 | 0.9285 | 0.6845±0.0047 | 0.8331±0.0042 | 0.7292±0.0055 | 0.668, 0.845, 0.730 | 0.997±0.009 |
| 3 | 0.2 | 0.0008 | 2e-04 | 4 | 0.8582 | 0.8885 | 0.6194±0.0053 | 0.7022±0.0058 | 0.5095±0.0076 | 0.617, 0.730, 0.541 | 0.994±0.009 |
| 3 | 0.3 | 0.003 | 2e-05 | 150 | 0.9292 | 0.9473 | 0.6816±0.0044 | 0.9107±0.0029 | 0.9221±0.0028 | 0.685, 0.916, 0.925 | 1.014±0.008 |
| 3 | 0.2 | 0.0015 | 2e-05 | 75 | 0.9627 | 0.9678 | 0.7034±0.0044 | 0.9445±0.0024 | 0.9451±0.0024 | 0.710, 0.946, 0.949 | 1.025±0.008 |

(P cerrada recalculada con v2; `C_mc.npz` guarda la v1.)

## 7. Reproducir

```
cd /home/user/nphonons-bundles/nuevo/bundles
python3 tests/run_tests.py                       # 14 pruebas, ~8 s
python3 src/run_D2.py; python3 src/run_D2b.py; python3 src/run_D1.py; python3 src/run_D3.py
python3 src/run_D4.py; python3 src/run_D5.py; python3 src/run_D6.py; python3 src/run_D7.py
python3 src/run_C.py; python3 src/run_C2.py      # MC, ~5 min cada uno con 2 procesos
python3 src/run_res.py; python3 src/run_conv.py; python3 src/analyze.py; python3 src/mk_tables.py
```
Los scripts usan caché (`--force` para recalcular) y fijan BLAS a 1 hilo.
