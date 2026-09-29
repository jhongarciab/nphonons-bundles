# V5 — Reporte de verificación independiente de R5 (baño filtrado)

Fecha: 2026-09-27. Código: `v5_filtro.py`, `run_v5.sh`. Salidas: `res_v5/estatico_gz0.txt`, `res_v5/estatico_N8.txt`, `res_v5/log_fl_*.txt`, `res_v5/*.npz`.
Del trabajo solo se consultaron los informes de la Tarea 43 y de la 45(b, c). No se leyó su código.

## Modelo
El qubit decae a través de un modo filtro b con frecuencia ω_f = 2ω = 12 y ancho κ_f. Acoplamiento J(σ₊b + σ₋b†), en RWA porque J ≤ 0.09 ≪ ω_f.
J se fija por 4J²/κ_f = κ = 0.03. El baño plano corresponde a κD[σ₋] sin filtro.
Parámetros de la Tarea 43: los de Ma (ω=6, g_x=−0.212, g_z=0.212, κ=0.03), ω_q = ω_p = 2(ω − 4g_x²/3ω), |α|² = 2, Ω = |α|²G, κ₂/κ = 1.
Predicciones: Γ₁⁻ = g_x²κ_eff(ω)/ω², Γ₁⁺ = g_x²κ_eff(3ω)/(9ω²), con κ_eff(δ) = κκ_f²/(4δ² + κ_f²).

## (a) Método nuevo: Liouvilliano estático sin drive (canales por separado)
- Marco de laboratorio sin drive. El autovalor real no nulo más lento (diagonalización densa) es la relajación de población del oscilador, Γ₁⁻ − Γ₁⁺.
- El n̄ estacionario, menos el n virtual del fundamental exacto de H (1.39e-4), da Γ₁⁺ = (Γ₁⁻ − Γ₁⁺)(n̄ − n_virt).
- **g_z = 0 en esta parte.** Con g_z ≠ 0 y sin drive, el intercambio de pares (qubit resonante) mezcla κ₂ con la relajación de población:
  en el baño plano Γ₁⁻ salía 1.44 veces el predicho (`estatico_N8.txt`), y con filtro 1.04.
  Γ₁± no dependen de g_z a segundo orden (V1).

| baño | N | N_f | Γ₁⁻ medido / predicho | razón | Γ₁⁺ medido / predicho | razón | κ₁ = Γ₁⁻ + Γ₁⁺ razón |
|---|---|---|---|---|---|---|---|
| plano | 8 | — | 3.7516e-5 / 3.7500e-5 | 1.0004 | 4.1644e-6 / 4.1667e-6 | 0.9995 | 1.0003 |
| plano | 12 | — | 3.7516e-5 / 3.7500e-5 | 1.0004 | 4.1643e-6 / 4.1667e-6 | 0.9994 | 1.0003 |
| κ_f = 1 | 8 | 2 | 2.5811e-7 / 2.5862e-7 | 0.9980 | 3.2162e-9 / 3.2125e-9 | 1.0011 | 0.9981 |
| κ_f = 1 | 8 | 3 | 2.5811e-7 / 2.5862e-7 | 0.9980 | 3.2162e-9 / 3.2125e-9 | 1.0011 | 0.9981 |
| κ_f = 0.3 | 8 | 2 | 2.3370e-8 / 2.3423e-8 | 0.9977 | 2.8965e-10 / 2.8933e-10 | 1.0011 | 0.9978 |
| κ_f = 0.3 | 8 | 3 | 2.3370e-8 / 2.3423e-8 | 0.9977 | 2.8965e-10 / 2.8933e-10 | 1.0011 | 0.9978 |
| κ_f = 0.3 | 12 | 3 | 2.3370e-8 / 2.3423e-8 | 0.9977 | 2.8965e-10 / 2.8933e-10 | 1.0011 | 0.9978 |

**Los dos canales (δ = ω y δ = 3ω) se confirman por separado al 0.2%.** Esto incluye que el canal de ganancia Γ₁⁺ se filtre con κ_eff(3ω).
El filtro de 2 niveles y el armónico (N_f = 3) coinciden a 1e-6, y N = 8 y 12 coinciden.

## (b) Floquet con drive (tasa de paridad del gato)
Propagador de un período T_p en el laboratorio y modo de paridad espectral, como en V4. |α|² = 2, N_f = 2.
Comparación con la fórmula corregida B, 2[Γ₁⁻|α|² + Γ₁⁺(|α|²+1)], evaluada con |α|² = |⟨a²⟩|.

| baño | N | r_par | \|⟨a²⟩\| | P_c estac. | razón A (nom.) | **razón B (\|⟨a²⟩\|)** | κ₁ = r/(2\|α\|²) (nom.) | κ₁ Tarea 43 (espectral) |
|---|---|---|---|---|---|---|---|---|
| plano | 16 | 1.72762e-4 | 1.9972 | 0.99818 | 1.0366 | **0.9886** | 4.319e-5 | 4.32e-5 |
| plano | 22 | 1.72754e-4 | 1.9972 | 0.99818 | 1.0365 | 0.9885 | 4.319e-5 | — |
| κ_f = 1 | 16 | 1.03766e-6 | 1.9975 | 0.99862 | 0.9908 | **0.9859** | 2.5942e-7 | 2.5942e-7 |
| κ_f = 0.3 | 16 | 9.39896e-8 | 1.9980 | 0.99862 | 0.9909 | **0.9859** | 2.3497e-8 | 2.3497e-8 (N=16), 2.3490e-8 (N=20) |

- κ₁ espectral coincide con la Tarea 43 a 4–5 cifras en los tres baños, con formulación distinta (laboratorio y T_p).
- Con B y |⟨a²⟩| las razones son 0.986–0.989, el déficit de −1.1% a −1.4% que V4b encuentra en régimen saturado (aquí κ₂/κ = 1).
  El exceso de A en el baño plano (+3.7% con |α|² = 2) es el término Γ₁⁺(|α|²+1), que es 1/(10|α|²) = 5% para |α|² = 2.
- Con filtro, A nominal sale 0.991 porque el filtro suprime Γ₁⁺ relativamente más que Γ₁⁻: Γ₁⁺/Γ₁⁻ baja de 0.11 (plano) a 0.012.

**Factor de mejora respecto al baño plano (cociente de tasas de paridad):**

| κ_f | medido | predicho con B (\|⟨a²⟩\|) | medido/pred | predicho con κ₁ (fórmula R5) | Tarea 43 medido |
|---|---|---|---|---|---|
| 1 | 166.5 | 166.1 | 1.003 | 159.1 | 166 |
| 0.3 | 1838 | 1833 | 1.003 | 1757 | 1.8e3 |

La mejora medida coincide con la predicción corregida al 0.3%. Con κ₁ = Γ₁⁻ + Γ₁⁺ (R5 tal cual) la predicción queda 4% por debajo.
Ese 4% es exactamente el efecto del +1: el filtro elimina casi todo Γ₁⁺, que en el baño plano pesaba el término Γ₁⁺(|α|²+1).

## Validaciones y convergencia
- ρ estacionarias (a) y (b): |Tr ρ − 1| ≤ 1.1e-15, ‖ρ − ρ†‖ = 0, mínimo autovalor ≥ −1.1e-9 (el κ_f = 0.3 estático con N_f = 3). Todas las tolerancias se cumplen.
- Convergencia:
  - (a): N = 8 → 12 cambia κ₁ en <1e-5; N_f = 2 → 3 en <1e-6.
  - (b): plano N = 16 → 22, r_par cambia 5e-5.
  - Con filtro no se repitió a N = 22 (propagador de 33 min con D = 64). La Tarea 45(c) reporta 3e-4 para κ_f = 0.3 entre N = 16 y 20.
- Modo de paridad con filtro: peso de borde ≤ 2.3e-5 e identificación inequívoca (|Tr(PR)|/‖R‖ = 1.40).

## Veredicto
**Confirma R5.**
- Las tasas Γ₁⁻ y Γ₁⁺ filtradas se reproducen canal por canal al 0.2% con κ_eff(ω) y κ_eff(3ω).
- La mejora medida es 166× (κ_f = 1) y 1838× (κ_f = 0.3), igual que el trabajo.
- Matiz para el paper: comparar mejoras con κ₁ = Γ₁⁻ + Γ₁⁺ subestima la mejora un 4% a |α|² = 2. Con la tasa de phase-flip corregida, 2[Γ₁⁻|α|² + Γ₁⁺(|α|²+1)], la predicción es exacta al 0.3%.

## Dudas y aproximaciones
- Acoplamiento qubit–filtro en RWA; no se incluyeron términos J(σ₊b† + h.c.).
- La parte (a) usa g_z = 0 por el motivo explicado arriba. Que g_z no afecte a Γ₁± solo queda probado indirectamente, porque (b), con g_z, reproduce las mismas tasas vía la fórmula.
- No se midió la brecha con filtro (la Tarea 43 reporta su caída). Tampoco se hicieron κ_f = 3 ni 0.1.
- En (b) el filtro es de 2 niveles. (a) muestra que N_f = 2 y 3 coinciden a 1e-6 sin drive; con drive no se comprobó.
