# V6 — Reporte de verificación independiente de R6 (brecha exacta del modelo mínimo)

Fecha: 2026-09-27. Código: `v6_brecha.py`. Salida: `res_v6/v6_salida.txt`. No se leyó código del trabajo.

## Método
- Modelo mínimo con α = 0: H = G(a²σ₊ + a†²σ₋), disipación κD[σ₋], κ = 1.
- Liouvilliano estático en QuTiP, diagonalización densa con N = 20 y N = 26.
- La brecha es el menor |Re λ| no nulo. El espacio oscuro span{|0⟩, |1⟩}⊗|g⟩ da exactamente 4 autovalores nulos (poblaciones y coherencias); se verificó el conteo en todos los puntos.
- Diferencia con el original: el trabajo usó Floquet en el modelo de Ma; aquí el modelo mínimo es estático y basta con diagonalizar.

## Derivación de la fórmula
La coherencia entre el sector oscuro y el par {|g,2⟩, |e,0⟩} evoluciona con el Hamiltoniano no hermítico [[0, √2G], [√2G, −iκ/2]].
Sus autovalores son −iκ/4 ± √(2G² − κ²/16).
- En régimen sobreamortiguado (32G²/κ² ≤ 1, es decir Γ₂ ≤ κ/8), la tasa lenta es (κ/4)[1 − √(1 − 8Γ₂/κ)], con Γ₂ = 4G²/κ.
- Por encima del umbral, todas las ramas tienen parte real κ/4.
- Los sectores n ≥ 3 tienen acoplamiento mayor (√(n(n−1))G) y relajan más rápido, y las poblaciones decaen al doble.
- Por tanto el modo más lento es esta coherencia, y la brecha coincide con R6.

## Resultados (κ = 1)

| Γ₂/κ | Δ numérico (N=20) | Δ predicho | dif. rel. | Δ(N=26)/Δ(N=20) − 1 |
|---|---|---|---|---|
| 0.005 | 5.05102572e-3 | 5.05102572e-3 | −2e-13 | −4e-13 |
| 0.01 | 1.02084238e-2 | 1.02084238e-2 | −1e-13 | 9e-14 |
| 0.02 | 2.08712153e-2 | 2.08712153e-2 | −8e-14 | 2e-14 |
| 0.05 | 5.63508327e-2 | 5.63508327e-2 | −6e-14 | 3e-14 |
| 0.08 | 0.100000000 | 0.100000000 | −3e-14 | −3e-14 |
| 0.10 | 0.138196601 | 0.138196601 | −4e-14 | 2e-15 |
| 0.12 | 0.200000000 | 0.200000000 | −3e-14 | −9e-16 |
| 0.124 | 0.227639320 | 0.227639320 | −1e-14 | −1e-14 |
| **0.125 (umbral)** | 0.249999976 | 0.25 | −9.5e-8 | 2e-8 |
| 0.126 | 0.25 | 0.25 | −1e-13 | 7e-14 |
| 0.13–5.0 (8 puntos) | 0.25 | 0.25 | ≤ 2e-13 | ≤ 2e-13 |

- En el umbral Γ₂ = κ/8 hay un punto excepcional: dos autovalores coalescen. Allí el error numérico escala como √ε_máquina, lo que explica el 1e-7. Es esperable y no indica discrepancia.
- Por encima del umbral la brecha vale exactamente κ/4 y los modos lentos pasan a ser oscilantes (Im λ ≠ 0).

## Validaciones y convergencia
- Exactamente 4 ceros en todos los puntos, como corresponde al espacio oscuro {|0⟩, |1⟩}⊗|g⟩.
- N = 20 → 26: la brecha cambia ≤ 4e-13, salvo en el punto excepcional (2e-8).
- Aquí no hay evolución temporal, así que no se generan matrices densidad; los chequeos de traza, hermiticidad y positividad no aplican.

## Veredicto
**Confirma R6 a precisión de máquina (≤ 2e-13) en 18 valores de Γ₂/κ entre 0.005 y 5,** incluido el umbral Γ₂ = κ/8, donde el error de 1e-7 es propio del punto excepcional.

## Dudas y aproximaciones
- Por encima del umbral, muchos modos comparten Re λ = κ/4 exactamente. El modo que el script elige para mostrar es arbitrario entre ellos y a veces es de borde (peso de borde ≈ 1 en la salida).
  Esto no afecta a la brecha: el valor mínimo es κ/4 y ningún modo interior está por debajo, y es estable entre N = 20 y 26.
- La fórmula es para α = 0. Con α ≠ 0 (el gato) el modelo mínimo tiene otra brecha, que no se verificó aquí.
