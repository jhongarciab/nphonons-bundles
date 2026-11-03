# Informe V: verificación independiente (partes A–D hechas sin abrir D/ ni E/; E al final)
Código en este directorio: `A.py` (modelo A), `B.py` (modelo B), `C.py`,`C2.py` (η), `mc.py` (motor de trayectorias de tiempo de espera), `D.py`,`D2.py`,`D3.py` (clusters). OMP_NUM_THREADS=1.

## A. Modelo A: G y resonancia (diagonalización exacta, N=30 y 45 idénticos a 1e-6)
Anticruce = mínimo en ω_q del hueco entre los dos autoestados con más peso en |e,0>,|g,2>; G_num = hueco/(2√2).
| (gx,gz)/ω | G_num/(2gxgz/ω) | e^{-2(gz/ω)²} | ω_q*−2ω | −4gx² | cociente | predicción 2º orden gx (exacta en gz) |
|---|---|---|---|---|---|---|
| 0.006,0.06 | 0.99280 | 0.99283 | −1.414e-4 | −1.440e-4 | 0.982 | −1.414e-4 |
| 0.01,0.1 | 0.98015 | 0.98020 | −3.804e-4 | −4.0e-4 | 0.951 | −3.804e-4 |
| 0.02,0.1 | 0.97999 | 0.98020 | −1.522e-3 | −1.6e-3 | 0.951 | −1.521e-3 |
| 0.03,0.15 | 0.95567 | 0.95600 | −3.215e-3 | −3.6e-3 | 0.893 | −3.212e-3 |
| 0.05,0.1 | 0.97888 | 0.98020 | −9.536e-3 | −1.0e-2 | 0.954 | −9.509e-3 |
| 0.03,0.3 | 0.83531 | 0.83527 | −2.228e-3 | −3.6e-3 | 0.619 | −2.226e-3 |
Derivación (polarón, exacta en g_z a primer orden en g_x): con λ=g_z/ω, ⟨g,2|g_xσ_x(a+a†)|e,0⟩ en la base desplazada D(∓λ) = −2√2 g_x λ e^{−2λ²}. Por tanto **G = (2 g_x g_z/ω) e^{−2g_z²/ω²}** (1 − ≈0.5 g_x²/ω², ajuste numérico, coef. 0.5–0.6). A g_z/ω ≤ 0.06 es 0.7 %; a 0.15 es 4.4 %; a 0.3 es 16 %.
Resonancia: el −4g_x²/ω es solo el límite λ→0 (|e,0> +g_x²/ω, |g,2> −3g_x²/ω). Con g_z el corrimiento se reduce (factor Franck–Condon): −3.93 g_x² (λ=0.06), −3.80 (0.1), −3.57 (0.15), −2.5 (0.3). La suma de segundo orden en la base polarónica (`shift_pred` en A.py) lo reproduce a 0.1–0.2 %. No derivé analíticamente la forma cerrada del factor (solo numérica, sumando intermedios).
Veredicto A: G=2g_xg_z/ω correcta como orden dominante; la corrección e^{−2λ²} es necesaria si λ ≳ 0.1. El corrimiento −4g_x²/ω solo vale a λ≲0.03.

## B. Modelo B (transmon, RWA)
El bloque N=2 es exactamente 3×3 (|2,0>,|1,1>,|0,2>), así que no hay "otros intermedios": la única corrección es de orden superior en g/Δ. Con ambos acoplos √2 g, la resonancia es EXACTA en ω_q=ω−α/2 (simetría), |−>=(|20>−|02>)/√2 queda desacoplado y 2V = [√(Δ²+16g²) − Δ]/2, Δ=|α|/2. Es decir **V = [√(α²/4+16g²) − |α|/2]/4 = (4g²/|α|)(1 − 16g⁴... )**; expansión: V = 4g²/|α| − 64g⁴/|α|³ + …
| g, α (MHz) | g/|α| | V_num (=½ hueco) | 4g²/|α| | cociente | 3×3 exacta |
|---|---|---|---|---|---|
| 15.2, −138 | 0.110 | 5.7414 | 6.6968 | 0.857 | 5.7414 |
| 15.2, −300 | 0.051 | 2.9634 | 3.0805 | 0.962 | 2.9634 |
| 5, −250 | 0.020 | 0.3975 | 0.4000 | 0.994 | 0.3975 |
(ω=2.3 GHz; K=5, M=10; corrimiento de resonancia numérico ≤5e-12 respecto de ω−α/2.) Por tanto 4g²/|α| sobreestima 14 % para el caso publicado PnC; V=5.74 MHz es el valor correcto.
Escala (resonancia propia de cada N: ω_q=ω−(N−1)α/2; V_N = ½ hueco): g=15.2, α=−138: V_3=1.038, V_4=0.1048 MHz; V_3/V_2=0.181, V_4/V_3=0.101. Para (15.2,−300): 0.078, 0.045; para (5,−250): 0.030, 0.018. Perturbativo: V_3≈6g³/α² (1.107, +6 %), V_4≈(16/3)g⁴/|α|³ (0.108, +3 %). La afirmación "0.1–0.2 por fonón para g/|α|≈0.11" es correcta (0.18 y 0.10). Nota: V_2,V_3,V_4 se dan en ω_q distintos (no simultáneamente resonantes).

## C. η
Convenciones exactas: jumps √κ_m a y √κ_q σ−; κ_m, κ_q tasas de energía (amplitud κ/2); |g,2> pierde población a 2κ_m (amplitud κ_m), |g,1>→|g,0> a κ_m; V=√2G; η = P(2 clics de a, 0 saltos del qubit, luego nada) desde |e,0>. Como tras el primer clic el qubit está en g, el segundo clic es seguro: η = 2κ_m∫|c_{g2}|²dt. Derivación propia (momentos de Lyapunov): 1/η = 1 + y₁/y₂ + y₁(y₁+y₂)/V², y₁=κ_q/2, y₂=κ_m. Sustituyendo da **exactamente** la fórmula afirmada: 1/η−1=κ_q/(2κ_m)+κ_qκ_m/(4G²)+κ_q²/(8G²). 
Numérico: Liouvilliano con matrices de renovación, H_ef con a truncada (N=8), 20 puntos (κ_q/G∈{0.05,0.2,0.5,1.4} × κ_m/G∈{0.5,1,√2,2,4}): máx |renovación − fórmula| = 4e-9. Ejemplos: (0.05,√2)→0.96556; (0.2,√2)→0.87228; (0.5,√2)→0.72212; (1.4,√2)→0.44744; (1.4,0.5)→0.3546. MC de trayectorias (2·10⁴): 0.8766±0.0023 vs 0.8723 y 0.4419±0.0035 vs 0.4357 (≈1.8σ, sesgo de dt=0.01). κ_m*=√2G verificado (optimizador: 1.41421343), 1/η−1 en el óptimo = κ_q/(√2G)+κ_q²/(8G²) exacto. Umbral η≥0.95: κ_q ≤ 0.07348 G (≈0.073; redondeo correcto 0.074).
Modelo completo A (autoestados vestidos, jumps seculares de frecuencia negativa, G=G_num): η_completo − fórmula = −0.5 pp (gx,gz=0.006,0.06; κ_q=0.2G), −0.26 pp (κ_q=1.4G), −1.5 pp (0.01,0.1), −1.6 pp (0.02,0.1, κ_q=0.2), −0.7 pp (κ_q=1.4), −2.5 pp (0.03,0.15, κ_q=0.5,κ_m=G): crece como ~λ² (renormalización de tasas por vestido; no son errores de la fórmula del subsistema). Condición de validez: la fórmula es exacta para H_ef; en el modelo completo es válida con error relativo ≈ O(2λ²).

## D. Pureza de cluster (MC, H_ef, N=6–7, G=1, κ_m=√2, τ_c=5/κ_m, x=5; clusters primero y último de cada trayectoria descartados; bombeo incoherente √W σ+)
Ambigüedad del enunciado: "Ω_bombeo" lo interpreto de dos formas. (i) W=Ω (rápido, bombeo no débil); (ii) Ω amplitud coherente con W=4Ω²/(κ_q+4G²/κ_m) (como en E, aquí tomado como tasa efectiva de absorción).
(ii) [la interpretación adecuada de "bombeo débil"]:
| κ_q | Ω | W | clusters | P2 (MC) | (1−2e^{−x})=0.9865 | q implícito | R·τ_c |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.05 | 0.00347 | 21761 | 0.9757±0.0010 | | 0.011 | 0.011 |
| 0.05 | 0.15 | 0.03127 | 18458 | 0.8901±0.0023 | | 0.098 | 0.085 |
| 1.4 | 0.05 | 0.00236 | 22634 | 0.9813±0.0009 | | 0.005 | 0.0035 |
| 1.4 | 0.15 | 0.02128 | 21183 | 0.9507±0.0015 | | 0.036 | 0.030 |
(i) W=Ω: (κ_q=0.05: W=0.05→0.8355, W=0.15→0.5991; κ_q=1.4: W=0.05→0.9016 (27950 clusters), W=0.15→0.7621): hay fusión grande (clusters de ≥4 clics 15–39 %).
Conclusión: la forma (1−2e^{−x})(1−q) es correcta: aparece de que cada par cuyo segundo clic se retrasa >τ_c produce 2 clusters de tamaño 1; exactamente (1−e^{−x})/(1+e^{−x})=tanh(x/2)=0.9866≈1−2e^{−x}. Fracción de clusters de tamaño 1 observada: 0.9–1.5 % (≈ 2e^{-x}/(1+..)=1.3 %). Con q_implícito ≈ 1.0–1.4·R·τ_c (R=tasa de clusters), una estimación Poisson q≈R τ_c es de orden correcto pero subestima 10–40 % (clusters 2+2 fusionados dominan P4+; el bombeo no es Poisson: refractario). P2 ≥0.95 se cumple solo con Ω ≲0.1–0.15 G. Aviso: en el punto (0.05,0.15) hubo 18458 clusters (<2·10⁴).

## E. Comparación con rutas/E/informe_E.md
Coinciden: G (0.993, 0.980, 0.980, 0.956: idénticos a los míos), η cerrada (misma fórmula, 0.073-0.074), κ_m*=√2G, V_B=5.74 MHz (E: 5.74; 4g²/α=6.70), V_3=1.04, V_4=0.105, razones 0.18/0.10, P2 con Ω≈0.1 G ≈0.95 (consistente con mis P2(ii): 0.976 a 0.05, 0.89–0.95 a 0.15), resonancia −3.93g_x² (λ=0.06; coincide con mi tabla). 
Discrepancias/observaciones: (1) E presenta "−3.93 g_x²" y dice "H_ef predice −4": no explica que es efecto de g_z (factor Franck–Condon); yo lo cuantifico. (2) E llama a la reducción de G "~e^{−λ²/2…}": la correcta es e^{−2λ²}. (3) E dice η_completo = efectivo +0.2–0.4 pp; yo obtengo −0.3 a −2.5 pp (según λ) con la probabilidad exacta de 2 clics; E usa E[M]/2 (clics esperados/2, no P(exactamente 2)) y su completo incluye otros canales, por lo que no son la misma magnitud; diferencia de signo no resuelta del todo. (4) E presenta V=4g²/α como "V" y la lista como 5.74 en tabla (correcto) pero en texto "V=4g²/|α|" sin avisar que sobreestima 14 %: la expresión citable es la de 3×3 exacta. (5) No usa que el bloque N=2 es 3×3 exacto (nada de "otros intermedios").

## F. Veredicto
- G=2g_xg_z/ω: correcta a primer orden; citar G=(2g_xg_z/ω)e^{−2g_z²/ω²} si g_z/ω≳0.1.
- ω_q*−2ω=−4g_x²/ω: solo el límite g_z→0; con g_z/ω=0.1 es 0.95 de eso, con 0.3 es 0.62. Citar el valor corregido o numérico.
- V=4g²/|α|: correcto a orden dominante; para g/|α|=0.11 sobreestima 14 %; usar V=¼[√(α²/4+16g²)−|α|/2] (exacta, RWA). Resonancia ω_q=ω−α/2 exacta.
- Escala V_N: 0.18 y 0.10 verificados (g/|α|=0.11).
- η y κ_m*=√2G: correctas (derivadas y verificadas por renovación a 4e-9); convenciones arriba; umbral κ_q≤0.0735G. Con modelo completo hay −0.3…−2.5 pp.
- P2 de clusters: (1−2e^{−x})(1−q) correcta con q de fusión; q≈R τ_c·(1.0–1.4) (aproximación, no fórmula cerrada); citar P2 como valor numérico.
Qué se puede citar: lo anterior con las correcciones. No verificado: dephasing, T1 en escalera, bombeo coherente (Rabi) real en D.
