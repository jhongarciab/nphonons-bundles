# Parte 2 — Captura directa con puerto fijo: borrador de resultados (modelo efectivo, validado parcialmente)

Estado: borrador de trabajo con números reproducibles (`nuevo/rutas/C2/`, scripts `captura.py`, `qubit.py`, `canal2.py`, `validar_*.py`, `mapa.py`, `robustez.py`; figuras en `figs/`). Todo lo que sigue es del modelo efectivo salvo donde se indica "completo".

## 1. Modelo efectivo (Obj. 2)
Tras eliminar adiabáticamente el estado brillante, el paquete itinerante, la cavidad y el estado oscuro `D_s` obedecen
`ȧ = −(κ/2)a − iG(t)d − √κ_e u(t)`, `ḋ = −iG(t)a`, con `κ = κ_e + κ_i`, `G(t) = Ω(t)g_s/Δ_I` y salida `b_out = u + √κ_e a`. Con el corrimiento Stark compensado por chirp, la captura queda determinada por la envolvente `u(t)` y por `G(t)`.
Sin pérdidas, la captura perfecta existe si y solo si `δ²(t) = ∫₀ᵗu² − u²/κ ≥ 0` (el paquete no crece más rápido que `e^{κt}`), con `G(t) = −√κ δ̇/u`. (Derivación propia; no contrastada con la literatura de captura en cavidad.)

**Validación contra el modelo Raman completo** (4 niveles excitónicos × modo × fuente en cascada, paquete gaussiano `T_p = 100`, control del modelo efectivo sin reoptimizar): eficiencia 0.830 (D1) y 0.844 (D2) frente a 0.892 del modelo efectivo. El modelo efectivo sobreestima en ≈0.05–0.06 por captura. Reoptimizando el control en el modelo completo (D1, rejilla de 27 puntos: escala de G ×0.9/1.0/1.1, retraso −20/0/+20 y desintonía constante −0.004/0/+0.004; `validar_opt.out`) el mejor resultado es 0.831, sin mejora respecto de 0.830: la diferencia no se corrige con un reajuste simple de esos tres parámetros (±0.004 de desintonía ya baja a 0.68–0.71), así que es una pérdida real del modelo completo (población virtual del brillante, nube de polarón, fugas), no un desajuste de calibración. La rejilla es gruesa y no se reoptimizó la forma de G(t).

## 2. Límites de la captura con puerto fijo (Obj. 3)
Con `G_max = 0.012 ω_b` (Ω ≈ 0.3 ω_b) y `κ_i = 10⁻³`, la eficiencia óptima converge, para paquetes largos, a **`η_cap → κ_e/(κ_e + κ_i)`**: 0.909 (`κ_e = 0.01`), 0.952 (0.02), 0.975 (0.04) [Fig. 2(a)]. Los paquetes cortos (`T_p = 50`) quedan por debajo porque `G_max` limita la absorción, y con `κ_e` pequeño hace falta `T_p` grande (`κ_e = 0.005`: 0.75 a `T_p = 100`, 0.83 a 250). El límite es, por tanto, la pérdida intrínseca de la cavidad, no la intensidad del láser dentro del rango explorado.

## 3. Canal completo (Obj. 4)
Mismo ruido (`γ_φ = 4×10⁻⁴`, correlación `χ`), misma entrada y mismos puertos (`κ_e = 10⁻²`, `κ_i = 10⁻³`); separación entre paquetes igual a la mínima del protocolo de dos etapas. `s` = recuperación, `F_h = (1+V)/2` (aproximación `P₀≈P₁`). [Fig. 3]

| Entrada, `χ=1` | Captura directa (`s`, `F_h`) | Dos etapas, puerto conmutable | Modos pasivos (`s`) |
|---|---|---|---|
| Gaussiano | 0.754, 0.862 | 0.198, 0.857 | 0.096 (T=0), 0.013 (2000), 0 (8000) |
| Exponencial creciente | 0.586, 0.830 | 0.643, 0.843 | 0.217 (T=0), 0.029 (2000), 0 (8000) |

- Frente a los modos pasivos, la memoria con estados oscuros conserva `s` a esperas largas; los pasivos lo pierden.
- Frente al protocolo de dos etapas, la ventaja de la captura directa **depende de la forma del paquete**: grande con paquetes gaussianos, ninguna (0.59 frente a 0.64) con el paquete exponencial creciente que el protocolo de dos etapas necesita. Con puerto fijo el protocolo de dos etapas falla (captura ≈0.35).
- Con `χ<1`, `F_h` decae con la espera igual en ambos protocolos: gaussiano directo, `χ=0.9`: 0.843/0.821/0.763 a `T=0/2000/8000`; `χ=0`: 0.710/0.608/0.515.
- La captura directa admite paquetes más cercanos: a separación 450, `V_c = 0.93` (con `s` menor, 0.68, y población en la ranura no direccionada 5.5×10⁻³); a 890, `V_c = 0.85`.
- Corrigiendo el sesgo del modelo efectivo (≈0.93 por etapa), `s` bajaría a ≈0.6–0.66 en el caso gaussiano directo (estimación que supone el mismo error en la lectura).

## 4. Robustez (Obj. 5)
Eficiencia de captura de referencia 0.892 [Fig. 2(b–d)]: error de llegada ±30 → 0.873 (±60 → 0.82); amplitud ±10 % → 0.87 (±20 % → 0.82); compensación Stark incompleta del 10 % → 0.874; **desintonía paquete–cavidad** ±0.45κ → 0.77, ±0.9κ → 0.53 (la sensibilidad crítica); forma del paquete: sech 0.885, exponenciales 0.44–0.45 (el control debe reoptimizarse).

## 5. Limitaciones que deben quedar escritas
Modelo efectivo con sesgo ≈+0.05–0.06 por captura; forma de G(t) no reoptimizada en el modelo completo (solo escala, retraso y desintonía); lectura y canal solo en el modelo efectivo; sin transferencia de población `D1↔D2` por ruido local ni decaimiento del estado oscuro durante la espera; sin biexcitón; una sola excitación; pasivo calculado de forma analítica simple; `F_h` con `P₀≈P₁`.

## Figuras (estilo de artículo; `figs_C2_v2.py`)
1. `figs/fig1_limites_captura.png`: eficiencia de captura frente a `(κ_e, T_p)` con el límite analítico `κ_e/(κ_e+κ_i)`.
2. `figs/fig2_canal_completo.png`: recuperación y `F_h` frente a la espera, y mapa `F_h(χ, T)`.
3. `figs/fig3_dinamica_robustez.png`: dinámica de la captura y robustez.
`F_h(χ,T)` del mapa usa `V(χ,T) = V₀(χ) e^{−(5γ_φ/6)(1−χ)T}` con `V₀(χ)` simulado (`chi_V0.npy`); la ley exponencial se verificó contra la simulación a `χ = 0` y `0.9` (cociente 0.513 y 0.936 frente a 0.513 y 0.936).
