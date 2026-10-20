# F7–F9. Memoria fonónica con transferencia Raman y varios emisores (16 oct 2026)

Enfoque: **teoría primero; la viabilidad de construcción es otra pregunta.** Todo en modelo reducido (excitón de un solo cuanto ⊗ modo con Ncut=3), sin biexcitón.
Cómputo en pcjhon. Scripts `src/f7_raman_memory_n2.py`, `f7b_raman_refine.py`, `f8_raman_lifetime.py`, `f9_dual_rail_n3.py`, `f9b_scan.py`, `f9_diag.py`;
datos `data/f7*.json`, `f8_raman_lifetime.json`, `f9*.json`, logs `data/f7b.log`, `f8.log`, `f9.log`.

## Idea que cambia el diseño (F7)
E28–E31 exigían |D⟩ a un fonón de |B⟩ y un **apagado hipotético** de la separación B–D. Con **transferencia Raman** el láser fija la resonancia de dos fotones
|G,1⟩↔|D,0⟩; en espera (láser apagado), |D,0⟩ queda fuera de resonancia con |B,1⟩ si gap=E_D−E_B≠ω_b. **No hace falta interruptor** y la ranura se elige con la frecuencia del láser.

## F7/F7b: N=2, pulso Raman, κ=0.001, γφ=4×10⁻⁴, Γc=0.0015 (relajación a T=0 hacia el nivel inferior), pulsos cuadrados, escritura efectiva (D≥0.8)
| gap=E_D−E_B | espera | mejor score (P·C·V)^⅓ | P / C / V | pasivo P / C |
|---|---:|---:|---|---|
| +0.5, +1.5 (D arriba) | 500 | 0.49–0.52 | 0.30 / 0.45–0.49 / 0.85–0.97 | 0.61 / 0.78 |
| −0.3, −0.5, −1.0 (**D abajo**) | 500 | 0.78–0.80 | 0.67–0.72 / 0.73–0.77 / 0.91–0.98 | 0.61 / 0.78 |
| −0.3, −0.5, −1.0 | 2000 | 0.68–0.69 | 0.62–0.67 / 0.53 / 0.92–0.97 | **0.14 / 0.37** |
| −0.3…−1.0, κ=0.005 | 500 | sin escritura ≥0.8 | — | — |
Con D **por debajo** de B la relajación del continuo B→D deja de ser pérdida (alimenta el almacén), y ese es justamente el orden del dímero vertical de E30/E32.
Con κ=0.005 la escritura (≥130/ω_b) no compite con la fuga del modo.

## F8: vida de la memoria N=2, D abajo (pulsos de F7b fijos). C = coherencia, P = recuperación
| gap, κ, γφ, Γc | t=1000 | t=2000 | t=4000 | t=8000 |
|---|---|---|---|---|
| −1.0, 0.001, 4×10⁻⁴, 0.0015 | 0.65/0.66 | 0.62/0.53 | 0.60/0.35 | 0.54/0.16 |
| −1.0, 0.001, 1×10⁻⁴, 0.0015 | 0.72/0.79 | 0.70/0.74 | 0.70/0.67 | 0.68/0.54 |
| −1.0, 0.001, 2×10⁻³, 0.0015 | 0.43/0.25 | 0.39/0.09 | 0.33/0.01 | 0.24/0.00 |
| pasivo κ=0.001 | 0.37/0.61 | 0.14/0.37 | 0.02/0.14 | 0.00/0.02 |
| pasivo κ=0.0003 | 0.74/0.86 | 0.55/0.74 | 0.30/0.55 | 0.09/0.30 |
- Punto de equilibrio en coherencia frente a κ=0.001: t≈700. A γφ=10⁻⁴ la memoria supera incluso al pasivo con κ=0.0003 desde t≈2000 (0.74 vs 0.74) y a t=8000 (0.54 vs 0.30).
- La población P decae muy lento (la relajación Γc la mantiene en D); **el límite es la coherencia, fijada por γφ**: e-fold de C ≈ 4500/ω_b a γφ=4×10⁻⁴.
- Con γφ=2×10⁻³ la memoria **pierde** frente al pasivo con κ=0.001 en C desde t≈1000.

## F9: N=3, dos ranuras oscuras, qubit de doble carril de dos *time-bins* (carga y descarga ideales)
Qubit (|0⟩_R|bin1⟩+|1⟩_R|bin2⟩)/√2; escritura Raman bin1→D1 (gap −0.5) con el láser en resonancia de D1, luego bin2→D2 (gap −1.0) en resonancia de D2;
lectura en orden inverso. Acoplos fonón-excitón g(B–D1)=0.057, g(B–D2)=0.061 (λ=(0.20,0.14,0) sobre la base B,D1,D2 del triángulo equilátero), γφ local.
F = fidelidad de entrelazamiento con la referencia R (fase corregida); s = prob. de heraldo (A1+A2=1); F|h = F/s.
| pérdidas | F | s |
|---|---:|---:|
| ninguna (κ=γ=γφ=Γc=0) | **0.921** | 0.952 |
| solo κ=0.001 | 0.717 | 0.746 |
| κ, γ | 0.716 | 0.743 |
| κ, γ, γφ=4×10⁻⁴ | 0.563 | 0.699 |
| + Γc=0.0015 en todos los pares incluido D1→D2 | 0.304 | 0.587 |
→ **El protocolo funciona en el límite ideal (F=0.92) sin interruptor.** La relajación inter-ranura D1→D2 (Γs) es el mayor enemigo: mezcla los carriles y el error **no** se hereda.
Con pulsos fijos, κ=0.001, γφ=4×10⁻⁴, Γc=0.0015, **Γs=0**:
| T | F memoria | s | F|h memoria | pasivo (2 modos) F / s / F|h |
|---:|---:|---:|---:|---|
| 500 | 0.558 | 0.700 | 0.80 | 0.307 / 0.317 / 0.97 |
| 2000 | 0.404 | 0.681 | 0.59 | 0.069 / 0.071 / 0.97 |
| 8000 | 0.193 | 0.631 | 0.31 | 0.000 / 0.000 / 0.97 |
Con κ=0.0003: T=500 el pasivo gana (0.70 vs 0.67), T=2000 empatan (0.45 vs 0.49), T=8000 gana la memoria (0.23 vs 0.07).
Efecto de Γs a T=2000: 0 → F=0.40; 3×10⁻⁴ → 0.31; 10⁻³ → 0.22. Efecto de γφ a T=2000 (Γs=0): 10⁻⁴ → 0.60 (F|h 0.83); 4×10⁻⁴ → 0.40; 2×10⁻³ → 0.15.

## Lectura
1. **Funciona en teoría, N=2:** transferencia Raman con D por debajo de B, sin interruptor, supera al fonón pasivo a t≳700 (κ=0.001) y retiene coherencia ≈0.5 hasta t≈2000.
2. **Funciona en teoría, N=3:** dos ranuras seleccionadas por frecuencia de láser, qubit de doble carril; F=0.92 sin pérdidas.
3. **El doble carril no elimina el desfase local:** F|h cae a 0.59 a T=2000 (γφ=4×10⁻⁴), peor que el pasivo heralding (0.97), aunque el éxito es ×10. El desfase **común** (ruido de carga correlacionado) sí lo cancelaría (ambos carriles tienen un excitón); aquí el modelo usa desfase local independiente. **Prueba pendiente decisiva.**
4. **Relajación inter-ranura:** Γs≳3×10⁻⁴ ya cuesta ≈25 % de F. Con T=0 las ranuras ordenadas por energía siempre tienen Γs>0; hay que fijar su escala física.
5. **No se hizo:** optimización completa de pulsos en N=3 (se interrumpió la búsqueda por coordenadas), carga/descarga reales, biexcitón, g_s derivado de una geometría, N>3, ni variación de las brechas (−0.5, −1.0 fijas).

## F10. Ruido de desfase común vs local (`src/f10_common_noise.py`, `data/f10_common_noise.json`)
χ = fracción del desfase que es ruido común (acopla a n₁+n₂+n₃); el resto es local independiente; varianza total por punto igual a γφ en todos los casos.
κ=0.001, γ=2×10⁻⁴, Γc=0.0015; N=2 un carril (pulso gap −1.0) y N=3 doble carril (pulsos de F9). Γs = relajación D1→D2.

| γφ | T | χ | N=2 un carril: P / C | N=3 doble carril, Γs=0: F / s / F\|h | N=3, Γs=Γc: F\|h |
|---:|---:|---:|---|---|---:|
| 4×10⁻⁴ | 2000 | 0 | 0.62 / 0.53 | 0.40 / 0.68 / 0.59 | 0.34 |
| 4×10⁻⁴ | 2000 | 0.5 | 0.67 / 0.53 | 0.50 / 0.70 / 0.72 | 0.35 |
| 4×10⁻⁴ | 2000 | 1 | 0.72 / 0.53 | 0.64 / 0.72 / **0.89** | 0.37 |
| 4×10⁻⁴ | 8000 | 0 | 0.54 / 0.16 | 0.19 / 0.63 / 0.31 | 0.30 |
| 4×10⁻⁴ | 8000 | 1 | 0.73 / 0.16 | 0.60 / 0.71 / **0.85** | 0.30 |
| 2×10⁻³ | 2000 | 1 | 0.69 / 0.09 | 0.50 / 0.67 / 0.75 | 0.34 |
| 2×10⁻³ | 8000 | 1 | 0.69 / 0.00 | 0.47 / 0.66 / 0.71 | 0.30 |

- **N=2 un carril es insensible a χ** (C idéntica): el desfase de la coherencia G–D es γφ/2 tanto local como común.
- **El doble carril de N=3 sí se beneficia del ruido común**: F|h pasa de 0.59 (χ=0) a 0.89 (χ=1) a T=2000, y conserva 0.85 a T=8000 (≈13 ns a 100 GHz) donde el un carril tiene C=0.16 y el pasivo tiene éxito ≈0.
- **Con Γs=Γc la ventaja desaparece** (F|h≈0.30–0.37 para cualquier χ). **La condición decisiva es Γs ≲ 3×10⁻⁴** (relajación entre las dos ranuras oscuras), no γφ.
- Ruido común puro (χ=1) es el límite favorable: es una hipótesis del modelo (dos o tres puntos apilados que ven el mismo campo eléctrico), no una propiedad demostrada.

## F11. Γs por regla de oro (`src/f11_gamma_s.py`)
Relajación T=0 por fonones LA entre los niveles de un triángulo equilátero de 3 puntos (B, D1, D2): Γ(a→b)=2πJ(ω)Σᵢⱼ mᵢmⱼ sinc(q dᵢⱼ), mᵢ=⟨b|nᵢ|a⟩,
GaAs de memoria (D_e=7.0, D_h=−3.5 eV, ρ=5370, c=5110 m/s, a=3 nm; incertidumbre ≈×3). Lado d fijado por la escala dipolar desde J=gap₁/3 (0.13 meV a 6 nm). Γ en unidades de ω_b:
| f_b | gaps (D1, D2) | d (nm) | Γ(B→D1) | Γ(B→D2) | **Γ(D1→D2)** |
|---:|---|---:|---:|---:|---:|
| 50 GHz | −0.5, −1.0 | 9.3 | 1.3×10⁻⁵ | 4.0×10⁻⁴ | **6.4×10⁻⁶** |
| 100 GHz | −0.5, −1.0 | 7.4 | 1.3×10⁻⁴ | 3.8×10⁻³ | **6.3×10⁻⁵** |
| 150 GHz | −0.5, −1.0 | 6.5 | 4.8×10⁻⁴ | 1.3×10⁻² | **2.4×10⁻⁴** |
| 200 GHz | −0.5, −1.0 | 5.9 | 1.2×10⁻³ | 2.9×10⁻² | 6.0×10⁻⁴ |
La relajación entre ranuras D1→D2 es **la mitad** de B→D1 y escala como ω⁵ (el factor 1−sinc≈(qd)²/6 la suprime a gap pequeño); F9 había supuesto Γs=Γc=1.5×10⁻³, muy pesimista a ≤150 GHz. En cambio B→D2 (gap 1.0) es grande.

## F12. F9 con las tasas de F11 por transición (`src/f12_golden_rates.py`, `data/f12_golden_rates.json`)
Pulsos de F9 sin reoptimizar, γ=2×10⁻⁴, γφ=4×10⁻⁴, χ=1 (ruido común) y χ=0 (local). Columna: F / s / F|herald. Pasivo de dos modos (κ=0.001): T=500: 0.31/0.32/0.97; T=2000: 0.07/0.07/0.97; T=8000: 0/0/0.97.
| f_b | κ | χ | T=500 | T=2000 | T=8000 |
|---:|---:|---:|---|---|---|
| 50 GHz | 0.001 | 1 | 0.66/0.73/0.91 | 0.65/0.72/**0.90** | 0.61/0.71/0.85 |
| 50 GHz | 0.001 | 0 | 0.56/0.70/0.80 | 0.40/0.66/0.61 | 0.18/0.58/0.31 |
| 100 GHz | 0.001 | 1 | 0.63/0.72/0.89 | 0.59/0.71/**0.83** | 0.47/0.71/0.67 |
| 100 GHz | 0.001 | 0 | 0.54/0.69/0.78 | 0.38/0.68/0.56 | 0.19/0.64/0.29 |
| 150 GHz | 0.001 | 1 | 0.55/0.68/0.81 | 0.46/0.68/0.67 | 0.28/0.68/0.40 |
| 100 GHz | 0.0003 | 1 | 0.76/0.85/0.89 | 0.71/0.84/0.84 | 0.57/0.83/0.69 |
(κ=0.0003: **la memoria gana en F** a T=500, 0.755 vs 0.698 del pasivo, y a T=2000, 0.710 vs 0.445; verificado contra `data/f12_golden_rates.json`.)

Lectura: con las tasas de regla de oro y ruido común, el doble carril de N=3 conserva **F|herald ≈ 0.83–0.90 a T=2000 a 50–100 GHz** (≈3 ns a 100 GHz), con éxito 0.7 frente a 0.07 del pasivo (κ=0.001).
El ruido local (χ=0) lo reduce a 0.56–0.61. A 150 GHz el B→D2 (1.3×10⁻²) y el D1→D2 ya penalizan.
Reservas: tasas ±×3; pulsos no reoptimizados con estas tasas ni con otra brecha; χ=1 es hipótesis; el solapamiento de los modos B y D con la cavidad no varía con f_b aquí.

## F13. Reoptimización de pulsos y brechas (`src/f13_reoptimize.py`, `f13_final.py`; `data/f13_*.json|log`)
Búsqueda por patrones (paso decreciente) sobre 8 parámetros: (Ω, duración, δ2) de cada ranura y las brechas g₁=|E_D1−E_B|, g₂=|E_D2−E_B|. Las tasas de relajación se recalculan por regla de oro (F11) para cada par de brechas.
Objetivo: F a T=2000, κ=0.001, γ=2×10⁻⁴, γφ=4×10⁻⁴, χ=1, f_b=100 GHz. La búsqueda usó Ncut=2 y tol=10⁻⁶ (comprobado: F cambia ≈2 %); la evaluación final usa Ncut=3, tol=10⁻⁸. Tres arranques:
| arranque | brechas finales (g₁, g₂) | pulso D1 (Ω, dur, δ2) | pulso D2 (Ω, dur, δ2) | F búsqueda | tasas a 100 GHz: B→D1, B→D2, D1→D2 |
|---|---|---|---|---:|---|
| 0 | 0.7, 1.0 | 0.132, 320, −0.001 | 0.29, 172, −0.015 | 0.686 | 5.3×10⁻⁴, 3.0×10⁻³, 4.0×10⁻⁶ |
| **1** | **0.35, 0.6** | **0.197, 170, −0.02** | **0.29, 150, −0.025** | **0.768** | 2.7×10⁻⁵, 3.9×10⁻⁴, 2.6×10⁻⁶ |
| 2 | 1.0, 1.275 | 0.14, 370, −0.001 | 0.335, 172, −0.021 | 0.656 | 2.4×10⁻³, 7.6×10⁻³, 2.0×10⁻⁶ |
(F9 sin reoptimizar, brechas 0.5/1.0, mismas condiciones: F=0.59.) Las brechas pequeñas ganan: la relajación a menor energía escala como ω⁵ y deja de dominar.

Evaluación final del arranque 1 (Ncut=3). Pasivo de dos modos con κ=0.001 para los mismos tiempos: T=500: F=0.365, s=0.369; T=2000: F=0.081, s=0.082; T=8000: 0/0.
| f_b | χ | T=500: F / s / F\|h | T=2000: F / s / F\|h | T=8000: F / s / F\|h |
|---:|---:|---|---|---|
| 50 GHz | 1 | 0.77 / 0.81 / 0.95 | 0.76 / 0.80 / **0.94** | 0.73 / 0.80 / 0.92 |
| 100 GHz | 1 | 0.77 / 0.81 / 0.94 | 0.75 / 0.81 / **0.93** | 0.72 / 0.81 / **0.90** |
| 150 GHz | 1 | 0.76 / 0.81 / 0.93 | 0.74 / 0.81 / 0.91 | 0.69 / 0.81 / 0.85 |
| 100 GHz | 0 | 0.64 / 0.75 / 0.86 | 0.45 / 0.68 / 0.66 | 0.17 / 0.52 / 0.33 |
| 50 GHz | 0 | 0.64 / 0.74 / 0.87 | 0.44 / 0.65 / 0.69 | 0.15 / 0.43 / 0.35 |
Mejora respecto a F12 (pulsos de F9): F|h a T=2000 pasa de 0.83 a 0.93 a 100 GHz con ruido común, y a T=8000 de 0.67 a 0.90.
Con ruido local (χ=0) la fidelidad condicionada sigue siendo ≈0.66 a T=2000 y ≈0.33 a T=8000: la ventaja de N=3 es sobre todo contra ruido común.

Reservas: óptimo local de una búsqueda por patrones (los arranques dan 0.66–0.77, el mejor está en brechas pequeñas y no se exploraron g₁<0.3); optimizado solo a 100 GHz, χ=1, T=2000;
brechas pequeñas implican J pequeño y detuning Raman pequeño (el modelo reducido no incluye biexcitón ni otros niveles); la separación geométrica para g₁=0.35 a 100 GHz es ≈8 nm (escala dipolar); tasas ±×3.

## F16. Modelo completo de 3 puntos con biexcitón (`src/f16_full3dot.py`; `data/f14_dxx_*.log`)
(Nota: la serie F14/F14b de covarianza espacial es de otro autor; este bloque era «F14» en mi numeración y pasa a F16 para evitar la colisión.)
Sin el truncamiento a un excitón: 8 estados electrónicos (|000⟩…|111⟩), H₁=Σₐ Eₐ|a⟩⟨a| construido en la base de puntos (B: 0, D1: −0.35, D2: −0.6), energía de pares Δxx Σ_{i<j}nᵢnⱼ,
láser Ω/√3 Σ(σᵢ+σᵢ†), relajación como operadores de salto Σ loᵢ hiⱼ σᵢ†σⱼ (válidos también en sectores multiexcitónicos), fonón Ncut=2, doble carril, **pulsos de F13 arranque 1 sin reoptimizar**,
carga y descarga ideales, f_b=100 GHz, χ=1, κ=0.001, γφ=4×10⁻⁴, T=2000. Referencia modelo reducido (mismos pulsos, Ncut=2): F≈0.77.
| Δxx/ω_b | F | s | F\|herald |
|---:|---:|---:|---:|
| −12 | 0.753 | 0.809 | 0.931 |
| −9 | 0.740 | 0.797 | 0.928 |
| −6 | 0.703 | 0.766 | 0.919 |
| −4.5 | 0.658 | 0.729 | 0.902 |
| −3 | 0.415 | 0.556 | 0.747 |
| −1.5 | 0.078 | 0.142 | 0.552 |
| 0 | 0.186 | 0.270 | 0.688 |
| +1.5 | 0.567 | 0.658 | 0.862 |
| +3 | 0.670 | 0.739 | 0.907 |
| +6 | 0.733 | 0.790 | 0.927 |
| +9 | 0.751 | 0.807 | 0.931 |
- **El biexcitón importa:** con |Δxx|≲3 ω_b (tampoco funciona Δxx=0) F cae a 0.08–0.4, porque el láser de la segunda escritura puede excitar un segundo excitón en el punto que ya guarda el primero.
  Con |Δxx|≳6 ω_b se recupera F≈0.70–0.75 (el reducido). A 100 GHz (ħω_b=0.41 meV) eso es |Δxx|≳2.5 meV; con las ligaduras de 1–4 meV de InAs/GaAs (de memoria, sin verificar), el borde bajo no basta y a 150 GHz (0.62 meV) menos.
- **Los pulsos no se reoptimizaron con Δxx finito:** parte de la caída es un desplazamiento de resonancia que una reoptimización de δ2 podría recuperar; no se comprobó.
- El signo antibinding (+Δxx) funciona antes que el binding (−Δxx) a igual |Δxx| (+1.5 → 0.57; −1.5 → 0.08).

## F17. Carga real del fonón por fuente en cascada (`src/f17_real_load_reduced.py`, `data/f17_*.log`)
Modelo reducido {G,B,D1,D2} (Ncut=2), pulsos y brechas de F13 arranque 1, f_b=100 GHz, χ=1, κ=0.001, γφ=4×10⁻⁴, Γ de F11, T=2000 salvo indicación.
Fuente en cascada (Gardiner/Kiilerich–Mølmer) con paquete exponencial creciente (óptimo, duración 4/κ_in, se corta cuando queda 2 % en la fuente). Descarga ideal.
Correcciones de implementación durante la validación: (i) signo de H_casc = −(i/2)√(κ_in γ_s)(b†a − a†b); (ii) la fuente debe tener la frecuencia del modo (ω_b a†a). Prueba mínima fuente→cavidad: p_b=0.96 al final.

| caso | η carga (bin1 / bin2) | F | s | F\|herald |
|---|---|---:|---:|---:|
| carga ideal, τ12=100 | 1 | 0.767 | 0.823 | 0.932 |
| real, κ_in=κ (paquete 4000/ω_b) | 0.961 / 0.950 | 0.473 | 0.766 | 0.617 |
| real, κ_in=κ/2 (pérdida intrínseca κ/2) | 0.44 / 0.44 | 0.190 | 0.379 | 0.502 |
| real, κ_in=3κ durante la carga; κ después | 0.961 / 0.948 | 0.600 | 0.769 | 0.781 |
| real, κ_in=5κ | 0.961 / 0.948 | 0.641 | 0.770 | 0.832 |
| real, κ_in=10κ | 0.961 / 0.948 | 0.677 | 0.770 | **0.878** (T=8000: 0.845) |
| real, κ_in=20κ | 0.961 / 0.949 | 0.697 | 0.771 | **0.904** |

**Hallazgo: la separación entre time-bins limita el doble carril.** Con carga ideal y T=2000 (χ=1), F|herald vs τ12: 100→0.932, 400→0.898, 1000→0.858, 2000→0.788, 4000→0.692 (χ=0: 0.660 → 0.541 a 4000).
Mecanismo: mientras llega el segundo time-bin, una rama guarda un excitón y la otra ninguno; la coherencia entre N=1 y N=0 decae a γφ/2 aunque el ruido sea común. El doble carril solo protege **después** de escribir ambos.
Con κ_in=κ fijo (paquete 4000/ω_b) F|herald=0.617, de acuerdo con τ12≈4000 (0.692) más pérdidas de carga.
El intervalo sin protección entre el final de la primera escritura y el final de la segunda es ≈ t_L + τ12 + d₂ con t_L=[4+ln 0.98]/κ_in (revisión externa, `REVISION_F16_F17_CONJUNTA.md`):
≈648/ω_b a 10κ y ≈449/ω_b a 20κ; por eso 10κ da 0.878 y solo 20κ cruza 0.90 con carga real.
Recuperar F|herald≈0.88 exige un puerto de carga ≈10× más rápido que la retención (κ_in≈0.01 al cargar, κ≈0.001 al retener) — un acoplador conmutable, aquí modelado como instantáneo.
Reservas: descarga ideal; η de carga ≈0.96 limitada por el corte del paquete (2 %) y el tiempo de fuga; el modelo con biexcitón (F16) y la carga real **no se combinaron** (el modo `real` de `f16_full3dot.py` tenía un corte fijo del paquete `TL−20`, que a 10κ dejaba ≈18 % en la fuente; corregido a `TL+ln(1−tail)/κ_in`; corrida conjunta Δxx=−9ω_b, κ_in=10κ lanzada, resultado pendiente).

## F16b. Corrida conjunta: biexcitón (Δxx=−9ω_b) + carga real con acoplador conmutable (κ_in=10κ)
`python3 src/f16_full3dot.py real -9 0.01`, modelo completo de 3 puntos (8 estados electrónicos), Ncut=2, pulsos y brechas de F13 arranque 1, f_b=100 GHz, χ=1, κ=0.001, T=2000, τ12=100, descarga ideal; corte del paquete corregido (2 % en la fuente). ~2 h de CPU. Log: `data/f16_real_dxxm9_kin0p01.log`.
| modelo | carga | F | s | F\|herald |
|---|---|---:|---:|---:|
| reducido (F13/F17) | ideal | 0.767 | 0.823 | 0.932 |
| reducido (F17) | real, κ_in=10κ | 0.677 | 0.770 | 0.878 |
| completo, Δxx=−9 (F16) | ideal | 0.740 | 0.797 | 0.928 |
| **completo, Δxx=−9** | **real, κ_in=10κ** | **0.693** | **0.753** | **0.920** |
Eficiencia de carga del bin 1: 0.961 (igual que el reducido). a=0.358, b=0.380, |c|=0.324.
Lectura: las dos condiciones (biexcitón y carga real) **se mantienen juntas** en el modelo completo: F|herald=0.92, sin pérdida respecto a la carga ideal del completo (0.928).
Discrepancia sin resolver: el modelo completo con carga real da **más** (0.920) que el reducido con carga real (0.878), mientras que con carga ideal ambos coinciden (0.928 vs 0.932).
No debe leerse como una ganancia por el biexcitón. Diferencias de implementación candidatas: la relajación (operadores de salto con hopping en el completo, proyectores en el reducido), el operador de desfase común (ΣN en todo el espacio frente a la suma de proyectores del sector de un excitón) y la pérdida del bin 2 (en el completo no se registró η del segundo bin).
Hasta aclararlo, la cifra citable es el intervalo 0.88–0.92 para κ_in=10κ.

## F17b. Diagnóstico reducido vs completo y coste de una lectura retardada
**Diagnóstico** (`f16_full3dot.py diag`, `f17_real_load_reduced.py diag`; Δxx=−9ω_b, κ_in=10κ, hasta la segunda carga): población del fonón tras cargar el bin 1: 0.9613 en ambos; D1 tras escribir el bin 1: 0.841 (reducido) / 0.830 (completo);
D1 tras cargar el bin 2: 0.825 / 0.822; fonón tras cargar el bin 2: 0.948 / 0.955. Los pasos iniciales coinciden en ≈1 %. La diferencia final (F: +0.016, s: −0.017 en el completo) amplifica a ΔF|h≈0.04 al dividir. No se localizó un fallo; la cifra honesta a 10κ es F|h=0.88–0.92 (±0.02 por modelo).
**Lectura retardada (proxy):** espera extra tw entre las dos descargas ideales (la rama R=1 ya sin excitón, la R=0 con el suyo en D1). Modelo reducido, T=2000, χ=1.
| carga | tw | F | s | F\|herald |
|---|---:|---:|---:|---:|
| ideal | 0 | 0.767 | 0.823 | 0.932 |
| ideal | 200 | 0.734 | 0.820 | 0.895 |
| ideal | 400 | 0.735 | 0.821 | 0.895 |
| ideal | 800 | 0.707 | 0.820 | 0.862 |
| real, κ_in=10κ | 0 | 0.677 | 0.770 | 0.878 |
| real, κ_in=10κ | 400 | 0.650 | 0.769 | 0.846 |
Una espera de ≈400/ω_b entre lecturas (≈ ventana de emisión de 4/κ_out con κ_out=10κ) cuesta ≈0.03–0.04 en F|herald. La no monotonía 200→400 (0.895 → 0.895) no se investigó; el valor de tw=800 (0.862) es consistente con el decaimiento e^{−γφ tw/2} de la coherencia N=1↔N=0.
Con carga y lectura reales a 10κ la estimación es F|herald≈0.85 (reducido). Es un proxy: no hay cascada de salida ni eficiencia de captura de la lectura.

## F17c. Carga y lectura reales (entrada y salida) en el modelo reducido
`src/f17_real_load_reduced.py` (… `real 10 T 100 1.0 0 real κ_out/κ`), datos `data/f17_ro_*.log`. Lectura: tras la pulsación Raman de lectura de cada ranura, la cavidad emite hacia la guía con κ_out conmutable durante 4/κ_out
y el paquete (exponencial decreciente) es capturado por una cavidad sumidero A_k con acoplo temporal óptimo γ_c(t)=κ_out e^{−κ_out t}/(ε+1−e^{−κ_out t}) (ε=0.02). Orden de lectura: ranura 2 primero, luego 1. Pérdida intrínseca κ despreciada durante la ventana (igual que en la carga).
Eficiencia de captura aislada (un fonón en la cavidad): **η_R=0.962** (κ_out=10κ). Mismos pulsos de F13, f_b=100 GHz, χ=1, κ=0.001, γφ=4×10⁻⁴, τ12=100.
**Pasivo**: dos cavidades de fonón con κ intrínseco; mismas eficiencias de carga η_L=(0.961, 0.948) y de captura η_R, bin 1 espera t_L2+T+t_E, bin 2 espera T (con t_L2=t_E=4/κ_in,out). F_pas=¼(p₁+p₂)+½√(p₁p₂), s_pas=½(p₁+p₂).
| carga | κ_out | T | F | s | F\|herald | pasivo F / s / F\|herald |
|---|---:|---:|---:|---:|---:|---|
| real 10κ | 10κ | 500 | 0.602 | 0.703 | 0.857 | 0.388 / 0.403 / 0.964 |
| real 10κ | 10κ | 2000 | 0.607 | 0.715 | **0.849** | 0.087 / 0.090 / 0.964 |
| real 10κ | 10κ | 8000 | 0.581 | 0.712 | 0.816 | 0.0002 / 0.0002 / 0.964 |
| real 10κ | 5κ | 2000 | 0.586 | 0.715 | 0.821 | 0.074 / 0.081 / 0.923 |
| real 10κ | 20κ | 2000 | 0.618 | 0.717 | 0.861 | 0.094 / 0.096 / 0.979 |
| ideal | 10κ | 2000 | 0.686 | 0.763 | 0.899 | 0.108 / 0.109 / 0.990 |
Lectura:
- **Memoria de entrada y salida con carga y lectura reales**: F≈0.58–0.62 casi independiente de T (500…8000), éxito s≈0.71, F|herald 0.82–0.86. Con κ_out=20κ y carga real: 0.861; con carga ideal: 0.899.
- **Frente al pasivo** (mismos criterios de captura): F sobre 7× a T=2000 y sin comparación posible a T=8000 (pasivo ≈0). El punto de equilibrio en F está en T≈50–100/ω_b (F_pas(T=0)≈0.64, F_pas(100)≈0.58); desde T≈100 la memoria gana en F y en éxito.
- **En fidelidad condicionada el pasivo gana** (0.92–0.98 frente a 0.82–0.86): el esquema multi-emisor compra éxito (≈0.7 frente a ≈0.09 a T=2000) a costa de F|herald; no mejora el estado heraldado.
- Falta de simulación de la sensibilidad a la forma del paquete emitido (aquí exponencial puro), de ε_cap y de que la pérdida intrínseca sí actúa durante la ventana.

## F17d. Corrección numérica: la discrepancia F16b era tolerancia del integrador (17 oct 2026)
Con `F14_TOL`/`F9_TOL=1e-6` (rtol; atol=1e-8) el modelo completo de 3 puntos deja matrices densidad **no positivas** (autovalor mínimo −2.8×10⁻² tras la primera carga, sin desfase). Con 1e-9: −2×10⁻⁹. `src/dbg_load1.py`, `src/cmp_stages.py`.
Carga real 10κ, lectura ideal, T=2000, χ=1, Ncut=2, tol 1e-9: reducido F/s/F_h = 0.689/0.770/0.894; completo proyectado a N≤1 (F16_PROJ=1) idéntico; completo Δxx=−9: 0.675/0.753/0.896. Con tol 1e-6 el completo daba 0.920 y el reducido 0.878 → ambos erróneos, el completo más. Poblaciones de ambos modelos coinciden a 1e-4 en cada etapa. Biexcitón a −9ωb: efecto ≈0.002 en F_h.
F17c repetido con 1e-9 (`data/f17tol9_*.log`, `data/memory_figs/operational_T*.json`; los de 1e-6 en `old_tol1e6/`): T=500/2000/8000 → F 0.613/0.618/0.591, s 0.703/0.715/0.712, F_h 0.872/0.864/0.830. Ncut=3, T=2000: 0.616/0.727/0.848.
**No recalculados con 1e-9:** F16 (barrido Δxx), F16b, F17 (κ_in), F17b (tw), F18, F18b. Las cifras absolutas del completo (F16, F16b) no son fiables.
