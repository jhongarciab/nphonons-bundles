# V4b — |α|² efectivo y término Γ₁⁺(|α|²+1) en la tasa de paridad

Fecha: 2026-09-26. Código: `v4_espectral.py` (con la variable de entorno `AL2` para |α|² nominal; Ω = |α|²G), `run_v4b.sh`, `v4b_analisis.py`.
Salida: `res_v4/v4b_salida.txt`, `res_v4/logb_*.txt`, `res_v4/b_*.npz`. Solo método espectral (propagador de un período en el laboratorio, como en V4).

Fórmulas comparadas (Γ₁⁻ = g_x²κ/(ω²+κ²/4), Γ₁⁺ = g_x²κ/(9ω²+κ²/4)):
- **A:** r = 2|α|²(Γ₁⁻ + Γ₁⁺), la usada en el trabajo.
- **B:** r = 2[Γ₁⁻|α|² + Γ₁⁺(|α|² + 1)]. Corrección relativa respecto a A: Γ₁⁺/((Γ₁⁻+Γ₁⁺)|α|²) = 1/(10|α|²).

La razón es r_par(espectral)/r_predicha. Para A con |α|² = 4 nominal coincide con la razón (κ₁/κ₂)/(5/72)(κ/g_z)² de V4.

## (1) Puntos de V4 con |α|² nominal (4) y con |α|² = |⟨a²⟩| del estado estacionario

| g_x | ω | g_z/κ | κ₂/κ | \|⟨a²⟩\| | A (nom.) | A (\|⟨a²⟩\|) | B (nom.) | B (\|⟨a²⟩\|) |
|---|---|---|---|---|---|---|---|---|
| 0.03 | 6 | 4 | 0.006 | 3.982 | 1.018 | 1.023 | 0.993 | 0.998 |
| 0.05 | 5 | 4 | 0.026 | 3.975 | 1.015 | 1.021 | 0.990 | 0.996 |
| 0.1 | 7 | 4 | 0.052 | 3.989 | 1.018 | 1.021 | 0.993 | 0.996 |
| 0.15 | 6 | 4 | 0.160 | 4.003 | 1.016 | 1.015 | 0.991 | 0.990 |
| 0.05 | 6 | 5 | 0.028 | 3.973 | 1.014 | 1.021 | 0.990 | 0.996 |
| 0.12 | 7 | 7 | 0.230 | 3.964 | 1.009 | 1.018 | 0.984 | 0.993 |
| 0.05 | 7 | 10 | 0.082 | 3.924 | 0.998 | 1.017 | 0.973 | 0.992 |
| 0.08 | 5 | 10 | 0.410 | 3.854 | 0.973 | 1.010 | 0.949 | 0.984 |
| 0.1 | 4 | 10 | 1.000 | 3.776 | 0.947 | 1.003 | 0.924 | 0.977 |
| 0.05 | 6 | 12 | 0.160 | 3.854 | 0.975 | 1.012 | 0.951 | 0.986 |

**Rangos:**
- A nominal: 0.947–1.018.
- A con |⟨a²⟩|: 1.003–1.023.
- B nominal: 0.924–0.993.
- **B con |⟨a²⟩|: 0.977–0.998.**

- La caída en los puntos saturados (0.947 en κ₂/κ = 1) se debe casi por completo a que el gato real es más pequeño que el nominal (|⟨a²⟩| = 3.78 en vez de 4). Con |⟨a²⟩|, A pasa a 1.003.
- Con A y |⟨a²⟩| queda un exceso sistemático de +1.5–2.3%, que B elimina en los puntos no saturados.

## (2) Barrido de |α|² a g_x = 0.05, ω = 6, g_z/κ = 4

| \|α\|² nom. | N | \|⟨a²⟩\| | P_c estac. | r_par | A (nom.) | corrección prevista 1/(10\|α\|²) | B (nom.) | A (\|⟨a²⟩\|) | B (\|⟨a²⟩\|) |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 22 | 1.9945 | 0.99950 | 9.6829e-6 | **1.0458** | +5.0% | 0.9960 | 1.0486 | **0.9986** |
| 4 | 22 | 3.9819 | 0.99953 | 1.8843e-5 | **1.0175** | +2.5% | 0.9927 | 1.0222 | **0.9971** |
| 6 | 26 | 5.9607 | 0.99950 | 2.7948e-5 | **1.0061** | +1.7% | 0.9896 | 1.0128 | **0.9961** |
| 6 | 32 | 5.9607 | 0.99950 | 2.7948e-5 | 1.0061 | +1.7% | 0.9896 | 1.0127 | 0.9960 |

- El exceso de A decrece con |α|²: +4.6%, +1.8% y +0.6% con |α|² nominal, o +4.9%, +2.2% y +1.3% con |⟨a²⟩|. Sigue la forma 1/(10|α|²) que predice B.
- **B con |⟨a²⟩| da 0.996–0.999 en los tres casos. La hipótesis B se sostiene al 0.4%.**
- El residuo de B, −0.1% a −0.4%, crece ligeramente con |α|². No se investigó.

## Validaciones y convergencia
- Estado estacionario: |Tr ρ − 1| ≤ 3.3e-16, ‖ρ − ρ†‖ = 0, mínimo autovalor ≥ −6.3e-11 (N=26). Todas las tolerancias se cumplen.
- Convergencia a |α|² = 6 (el caso más exigente), N = 26 → 32: r_par pasa de 2.79473e-5 a 2.79479e-5 (2e-5 relativo); P_c y ⟨a²⟩ idénticos a 5 cifras.
  El peso de borde del modo de paridad es 2.5e-6 (N=26) y 4.5e-10 (N=32).
- El punto |α|² = 4 de (2) repite el de (1) con distinto g_z/κ. Coincide con (0.05, 6, 5) en A nominal: 1.0175 frente a 1.0144.

## Veredicto
**Se confirma la hipótesis B.** La tasa de paridad es 2[Γ₁⁻|α|² + Γ₁⁺(|α|²+1)] con |α|² = |⟨a²⟩|:
- (2): razones 0.996–0.999 en |α|² = 2, 4 y 6.
- (1): razones 0.990–0.998 en los puntos no saturados (κ₂/κ ≤ 0.23).
- El exceso de +1.6% de V4 es el término Γ₁⁺ adicional, y el defecto en saturación es el |α|² efectivo menor que el nominal.

Implicación para R4: la figura de mérito corregida es κ₁/κ₂ = (5/72)(κ/g_z)²·[1 + 1/(10|α|²)], definiendo κ₁ = r/(2|α|²). Sigue sin depender de g_x ni de ω.

## Dudas y aproximaciones
- En los puntos saturados (κ₂/κ ≥ 0.4), B con |⟨a²⟩| queda 1.6–2.3% por debajo. Puede ser una corrección no adiabática en κ₂/κ, o que |⟨a²⟩| no sea el |α|² adecuado cuando el código está vestido. No se investigó.
- |⟨a²⟩| se toma del estado estacionario, que es una mezcla de |±α⟩. Con P_c ≈ 0.9995, la contaminación fuera del código es despreciable a este nivel.
- Solo un punto de (g_x, ω, g_z) se usó en el barrido de |α|².
