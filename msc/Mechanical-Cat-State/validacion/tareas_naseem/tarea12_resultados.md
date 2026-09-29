# Tarea 12 — Estado oscuro en el estacionario (ε=1.44)

Script: `tarea12_estado_oscuro.py`. Datos: `tarea12_resultados.npz`.
Nb=30, τ_max=60, estroboscópico, g_eff=2g, ε=Ω_full=1.44.

## Resultados en el estacionario

| Cantidad | Valor |
|---|---|
| p_e_ss | 2.236e-3 |
| \|⟨σ₋⟩\|_ss | 3.673e-2 |
| ⟨n⟩_ss | 1.9047 |
| ⟨a²⟩_ss (complejo) | 1.9600 + 0.0250j |
| \|⟨a²⟩\|_ss | 1.9602 |
| fase(⟨a²⟩) | 0.0127 rad (≈0.73°, casi real positivo) |
| Paridad_ss | 0.8725 |

## Validación

Traza, hermiticidad y positividad perfectas (mín. autovalor ~−9.3e-25,
muy por dentro del umbral −1e-9).

## Comparación con la predicción del estado oscuro

Predicción: |α²| = ε/g_eff = 1.44/0.72 = **2.0000** (exacto, por
construcción del punto de operación).
Medido: |⟨a²⟩|_ss = **1.9602** → razón medido/predicción = **0.980**
(2.0% por debajo). Excelente acuerdo — confirma cuantitativamente que
el estacionario es, en muy buena aproximación, el estado oscuro de
amplitud α²≈ε/g_eff predicho por la teoría efectiva.

## Fidelidad con el gato par |C+⟩

Deduciendo α directamente de ⟨a²⟩_ss medido (α=√(1.9600+0.0250j) =
1.4000+0.0089j, sin asumir la fase, solo tomando la raíz de lo medido):

    F(ρ_ss, |C+⟩⟨C+|) = 0.9669

Alta fidelidad (~97%) con el gato de Schrödinger par de esa amplitud,
consistente con Paridad_ss=0.873 (menor que 1, pero razonablemente alta —
la diferencia entre 0.967² (~0.935, cota tipo pureza-fidelidad) y la
paridad observada 0.873 es consistente con una mezcla residual, no con un
estado completamente distinto del gato par).

## Conclusión Tarea 12

El estado estacionario a ε=1.44 (Ω_full, el punto de operación de
fig1/fig2 del paper) es, con muy buena aproximación (F≈0.97, |α²|
correcto dentro de 2%), el **estado oscuro tipo gato par** predicho por
la teoría efectiva con g_eff=2g — confirmación adicional, en el dominio
del estado estacionario, de que la normalización g_eff=2g (Tareas 7-8) es
la correcta, y de que la hipótesis (2) del enunciado (p_e_ss≈0 porque el
estacionario es un estado oscuro) es consistente con las demás piezas de
evidencia numérica de esta ronda.
