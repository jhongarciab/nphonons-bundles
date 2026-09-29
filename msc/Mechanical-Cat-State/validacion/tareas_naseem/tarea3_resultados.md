# Tarea 3 — Aislar el término de dos fonones

Script: `tarea3_dos_fonones.py`. Datos: `tarea3_dos_fonones_resultados.npz`.

## Configuración

Modelo COMPLETO (qubit + oscilador), **sin drive** (Ω=0), partiendo de
|g⟩⊗|n=4⟩ (Na=2, Nb=18). Disipadores: decaimiento del qubit (κ) y
amortiguamiento mecánico de un fonón (γ_m, muy pequeño). Se mide ⟨n⟩(t) y se
compara contra la fórmula analítica

    Γ₂₋(g_eff) = 2 g_eff² Re S₂₋,   Re S₂₋ = x/(x²+Δ₂₋²) = 2  (resonancia exacta)

para g_eff = g = 0.36 (código original) → Γ₂₋(g) = 0.5184
y g_eff = 2g = 0.72 (hipótesis factor 2) → Γ₂₋(2g) = 2.0736  (4× mayor).

## Hallazgo 1: la dinámica NO es un decaimiento markoviano limpio

⟨n⟩(t) bruto oscila de forma coherente y no monótona en el corto plazo
(4.00 → 4.87 → 5.38 → 3.76 → 4.03 → 3.45 → 3.63 → 2.64 ... en κt=0 a 2.5),
en vez de decaer suavemente. Esto ocurre porque, al apagar el drive y partir
de un Fock n=4 (no del vacío), los acoplamientos DIRECTOS de un fonón
(g_x=6, g_z=60, en unidades de κ) —aunque muy desintonizados de la
transición del qubit (Δ₁∓≈ω_m=1000)— generan oscilaciones coherentes tipo
Rabi de amplitud apreciable a n finito, que compiten con y enmascaran el
proceso secular de dos fonones que se quiere aislar. La validación de
positividad del estado reducido también empeora respecto a Tareas 1-2
(mín. autovalor −4.9e-5, vs ~1e-6 antes), consistente con dinámica más
rápida/oscilatoria mal resuelta por las tolerancias por defecto del solver.

## Hallazgo 2: tendencia secular (suavizada) sí decae, pero más lento que ambas fórmulas

Suavizando ⟨n⟩(t) con promedio móvil (ventana ≈ 1/κ) para remover las
oscilaciones coherentes rápidas, se obtiene una tendencia claramente
decreciente:

| κt | ⟨n⟩ suavizado |
|---|---|
| 1.0 | 4.13 |
| 2.0 | 3.25 |
| 3.0 | 2.53 |
| 4.0 | 1.97 |
| 6.0 | 1.12 |
| 8.0 | 0.68 |
| 10.0 | 0.44 |
| 12.0 | 0.32 |
| 14.0 | 0.26 |

Usando el ansatz `dn/dt = -2 Γ₂₋ n(n-1)` sobre esta tendencia (región
κt∈[1,4], antes de que n caiga por debajo de 1 y el ansatz deje de ser
válido), la tasa medida es:

    Γ₂₋(medido) ≈ 0.10  (mediana en la región válida)

| Comparación | Γ₂₋ fórmula | Γ₂₋(medido)/fórmula |
|---|---|---|
| g_eff = g  | 0.5184 | **0.201** (medido 5× menor) |
| g_eff = 2g | 2.0736 | **0.050** (medido 20× menor) |

## Interpretación

Ninguna de las dos fórmulas reproduce cuantitativamente la tasa medida
—ambas sobreestiman el decaimiento real—, pero la discrepancia con
**g_eff = g es 5 veces menor** que con g_eff = 2g. Esto es evidencia
adicional (aunque más débil y con más incertidumbre metodológica que la de
la Tarea 2) en contra de la hipótesis del factor 2: si g_eff realmente
fuera 2g, se esperaría una tasa medida ~4× más grande que la observada bajo
g_eff=g, y en cambio la tasa medida está del lado de g_eff=g (mismo orden
de magnitud, factor ~5) y muy lejos de g_eff=2g (factor ~20).

La discrepancia global (medido ≈ 0.2× la fórmula de g_eff=g) es esperable:
el ansatz `n(n-1)` markoviano asume una separación de escalas de tiempo
(κ⁻¹ ≪ tiempo de decaimiento del fonón) que aquí no está bien satisfecha
—g_x=6κ es del mismo orden que κ, y la ventana de suavizado (≈1/κ) se
superpone con el propio tiempo de memoria del qubit—, además de que el
ansatz `⟨a†²a²⟩≈⟨n⟩(⟨n⟩-1)` deja de ser exacto una vez el estado del
oscilador se entrelaza con el qubit y se aparta de un Fock puro.

## Conclusión Tarea 3

El aislamiento directo del canal de dos fonones (apagando el drive) **no
da una medición limpia y precisa de g_eff** por las razones anotadas
(oscilaciones coherentes de un fonón residuales, régimen no estrictamente
markoviano a estos parámetros). Sin embargo, en la medida en que el método
sí es informativo, **favorece g_eff = g sobre g_eff = 2g** por un margen de
~4×, reforzando la conclusión de la Tarea 2: no hay evidencia numérica que
sostenga la necesidad de duplicar g_eff.

## Archivos generados

- `tarea3_dos_fonones_resultados.npz`: `tlist`, `nb_t`, `dndt`,
  `Gamma2_measured`, `Gamma2_meas_plateau`, `Gamma2_g`, `Gamma2_2g`,
  `g_dim`, `trs`, `herms`, `mineigs`.
