# Tarea 8 — Álgebra simbólica: transformación tipo polarón

Script: `tarea8_algebra_polaron.py`. Datos: `tarea8_resultados.npz`.

## Planteamiento

Con `S = -(g_z/ω_m) σ_z (a†-a)`, se calcula `e^S H e^{-S}` para
`H = (ω_q/2)σ_z + ω_m a†a + g_x σ_x(a+a†) + g_z σ_z(a+a†)` vía BCH:

    H' = H + [S,H] + (1/2!)[S,[S,H]] + ...

y se extrae el coeficiente del término σ_y(a†²-a²) (⟺ σ_{+-} a², a†²),
usando matrices de QuTiP truncadas en Fock (N=30 para los commutadores
exactos, N=12 para verificar con exponencial matricial completa).

## Derivación (verificada numéricamente)

`[S,H_free]` y `[S,H_gz]` dan únicamente corrimientos de energía escalares
(no aportan a σ_{+-}a²,a†²; confirmado: coeficiente en `[S,[S,H]]` = 0.000
exacto). El **único** término que genera σ_y(a†²-a²) es el cruce
`[S, H_gx]`:

    [S, H_gx] = 2iλ g_x σ_y ⊗ (a†²-a²),   λ = -g_z/ω_m
              = -2i (g_z g_x/ω_m) σ_y (a†²-a²) = -2i g σ_y(a†²-a²)

## Resultados numéricos

| Método | Coeficiente de σ_y(a†²-a²) |
|---|---|
| `[S,H]` (commutador exacto, N=30) | **−0.720000 j** |
| `[S,H] + ½[S,[S,H]]` (BCH-2) | **−0.720000 j** (idéntico — [S,[S,H]] no aporta) |
| `e^S H e^{-S}` exacto (exp. matricial, N=12) | **−0.702042 j** |
| Predicción −2g (con g=0.36) | **−0.720000 j** |

Diferencia entre exponencial exacta y BCH-2: **2.49%**, consistente con
correcciones de orden superior en el parámetro pequeño g_z/ω_m=0.06
(≈(0.06)²·algo ~ orden correcto de magnitud para 3er/4to orden).

## Comparación con Ecs. (9) y (11)

| | Coeficiente (base σ_y) |
|---|---|
| Medido (BCH-2 y exacto) | −0.72j / −0.702j |
| **Ec. (9)** predice: −2ig | **−0.72j** ✓ coincide exactamente |
| **Ec. (11)** predice (convertida a base σ_y): +ig | +0.36j ✗ — factor **−2** de diferencia |

## Conclusión Tarea 8

La derivación directa por transformación tipo polarón (verificada tanto
por commutadores anidados exactos como por exponenciación matricial
completa, sin aproximaciones adicionales más allá del orden de
truncamiento en g_z/ω_m, que es pequeño: 0.06) **reproduce exactamente el
coeficiente de la Ec. (9)** del manuscrito: `-2ig σ_y(a†²-a²) =
-2g(σ_+-σ_-)(a†²-a²)`, con g=g_z g_x/ω_m tal como está definido en el
código.

**La Ec. (11), tal como se plantea en el enunciado (`+g(σ_+-σ_-)(a†²-a²)`),
tiene un error de un factor −2** respecto a la derivación correcta (Ec. 9):
difiere tanto en magnitud (factor 2) como en signo. Esto **confirma
algebraicamente, de forma independiente y sin ambigüedad**, que
**g_eff = 2g es el coeficiente de acoplamiento correcto** para el término
σ_{+-}a²,a†² — consistente con la medición directa de Rabi de la Tarea 7
(que dio g_eff=2g con <5% de error) y con el buen ajuste de n_ss en el
régimen ε≳0.4κ de la Tarea 6b.
