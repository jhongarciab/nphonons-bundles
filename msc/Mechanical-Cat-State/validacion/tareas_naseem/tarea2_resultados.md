# Tarea 2 — Hipótesis del factor 2

Script: `tarea2_factor2.py`. Datos: `tarea2_factor2_resultados.npz`.

## Planteamiento

Ec. (9): `-2i g σ_y (a² - a†²) = -2g(σ_+ - σ_-)(a†² - a²)`
Ec. (11) (usada para derivar el modelo efectivo): `+g(σ_+ - σ_-)(a†² - a²)`

→ posible factor 2 perdido en el acoplamiento de dos fonones g_eff entre la
derivación (Ec. 9) y la ecuación efectiva usada (Ec. 11).

Se comparan tres modelos efectivos contra el modelo completo (Ω = 4g, el drive
real tal como está codificado):

| Modelo | g_eff | ε (drive efectivo) | Interpretación |
|---|---|---|---|
| A — paper/código original | g | 2g | prescripción original del código |
| B — corregido | 2g | 4g (= Ω_full) | corrige g por el factor 2 y usa ε = drive real |
| C — drive 4g sin corregir g | g | 4g (= Ω_full) | usa ε = drive real, sin tocar g |

## Resultados

| Modelo | g_eff/g | ε/g | n̄_ss | t₉₀ (κ⁻¹) | F_min | F(κt=39) |
|---|---|---|---|---|---|---|
| **A_paper** | 1.0 | 2.0 | 1.9284 | 3.068 | 0.7306 | **0.9797** |
| **B_corregido** | 2.0 | 4.0 | 1.9285 | 0.813 | 0.5009 | **0.9799** |
| **C_drive_4g** | 1.0 | 4.0 | 3.9973 | 1.801 | 0.6179 | **0.8163** |
| **FULL** (referencia) | — | — | 1.8956 | 6.914 | — | — |

Trayectorias ⟨n⟩(t) en puntos clave (κt):

| κt | FULL | A_paper | B_corregido | C_drive_4g |
|---|---|---|---|---|
| 0.98 | 0.6240 | **0.6240** | 1.8540 | 2.3312 |
| 1.97 | 1.0054 | 1.3518 | 1.9273 | 3.7227 |
| 4.92 | 1.4868 | 1.9038 | 1.9281 | 3.9967 |
| 10.16 | 1.8173 | 1.9281 | 1.9282 | 3.9973 |
| 39.00 | 1.9030 | 1.9284 | 1.9285 | 3.9973 |

**Observación central**: el modelo A (sin ningún "corregir"; el que ya usa el
código/paper) reproduce el valor exacto de ⟨n⟩ del modelo completo en κt=0.98
(0.6240 vs 0.6240), y converge a un n̄_ss dentro de 1.3% del completo (1.9284
vs 1.8956), con fidelidad final alta (0.98). El modelo B, que "corrige" g_eff→2g
y usa ε=Ω_full=4g, mantiene la misma razón ε/g_eff que A y por eso llega al
mismo n̄_ss —pero con una dinámica transitoria completamente distinta y mucho
más rápida (t₉₀ 8.5× menor que el completo), sin acuerdo temprano. El modelo C
—que usa ε=Ω_full sin corregir g— da un n̄_ss equivocado por un factor ~2.1,
y la peor fidelidad final (0.82).

Ningún modelo reproduce exactamente t₉₀ del completo (A: 3.07 vs 6.91 —el más
cercano; B: 0.81; C: 1.80), lo que sugiere que hay una escala de relajación en
el modelo completo no capturada del todo por la eliminación adiabática a este
orden, independiente de la cuestión del factor 2.

## Relación de marcos (verificación analítica)

**Modelo completo**: en el código, `sz`, `sx`, `b` son operadores de Schrödinger
sin transformar (tensor(sigmaz(), qeye), etc.), y el término estático
`H0 = (ω_q/2) σ_z` permanece explícito en H(t). Esto es la firma de que **el
qubit se mantiene en marco de laboratorio** (no se le aplicó ninguna
transformación unitaria dependiente del tiempo). En cambio, los términos de
acoplamiento aparecen como `g_x σ_x (b e^{-iω_m t} + b† e^{iω_m t})`: la fase
`e^{∓iω_m t}` adosada a `b`/`b†` es exactamente la forma que toman los
operadores de escalera en el **cuadro de interacción respecto al Hamiltoniano
libre mecánico** `H_m = ω_m b†b` (el término libre bruto ya no aparece en H,
fue removido y reemplazado por esta rotación de fase). El término de drive
`Ω σ_x (e^{iω_d t} + e^{-iω_d t})` es simplemente la escritura explícita de un
campo clásico oscilante `2Ω σ_x cos(ω_d t)` en el marco de laboratorio —no es
un artefacto de cambio de marco.

**Conclusión de marcos**: qubit en marco de laboratorio (H0 explícito, sin
transformar), mecánico en cuadro de interacción respecto a `ω_m b†b`. Esto
coincide con lo indicado en el enunciado de la tarea.

**Modelo efectivo**: `H_eff = χ* a†² + χ a² + δ_k(a†a)²` no tiene dependencia
temporal explícita — está escrito en un marco donde el proceso de dos fonones
es estacionario. Esto es consistente con el marco del modelo completo
únicamente bajo la condición de resonancia `ω_q = 2ω_m` (usada en todos los
scripts): en tal caso, el proceso virtual de dos fotones del qubit (que oscila
a `2ω_m` en el cuadro de interacción del mecánico) cancela exactamente la fase
`e^{2iω_m t}` que adquiriría un término `a†²` en dicho cuadro. La consistencia
de marcos entre completo y efectivo depende de esa resonancia exacta, pero **no
introduce ningún factor 2 adicional** — es ortogonal a la cuestión algebraica
de las Ecs. (9)/(11).

## Conclusión Tarea 2: ¿el factor 2 es real?

**No, dentro de lo que se puede probar aquí**: el factor 2 aparente entre
Ec. (9) y Ec. (11) no se traduce en un error físico detectable en la
comparación completo-vs-efectivo. La prescripción original del código/paper
(g_eff=g, ε=2g — Modelo A) es la que **mejor reproduce tanto el transitorio
corto (⟨n⟩ exacto en κt≈1) como el estado estacionario** (n̄_ss dentro de
1.3%) del modelo completo con Ω=4g real. Intentar "corregir" g_eff→2g
(Modelo B) no cambia el estado estacionario (por preservar ε/g_eff=2) pero
destruye el acuerdo transitorio; usar ε=Ω_full literalmente sin ajustar g
(Modelo C) da un estado estacionario equivocado por un factor ~2.

Esto sugiere que ε en el código **no es simplemente el Ω del qubit
trasladado 1:1** al modelo reducido —hay un paso de eliminación adiabática
entre el drive real del qubit (Ω) y el drive efectivo de dos fonones (ε) que
ya absorbe cualquier factor de normalización, independientemente de si la
Ec. (11) tiene o no el factor 2 de la Ec. (9) tal como está escrita en el
manuscrito. La discrepancia Ec.(9)/Ec.(11) podría ser una inconsistencia
puramente notacional/tipográfica en el manuscrito que no afecta los
resultados numéricos ya validados en fig1/fig2/fig4/fig5, **o bien estar ya
compensada por una redefinición de g en otro punto de la derivación** que no
se pudo aislar solo con esta comparación completo-vs-efectivo. Se recomienda
revisar el álgebra simbólica completa entre Ecs. (9)-(11) del manuscrito para
confirmar cuál interpretación aplica, pero el respaldo numérico favorece
fuertemente la normalización ya usada en el código (Modelo A) sobre cualquiera
de las alternativas "corregidas" probadas.
