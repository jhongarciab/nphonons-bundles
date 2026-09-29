# V1 — Reporte de verificación independiente de R1

Fecha: 2026-09-26. Código: `v1_diag.py`, `v1_signo.py` (QuTiP 5.3.1, venv propio `.venv`). Salidas: `v1_salida.txt`, `v1b_salida.txt`.
Derivación analítica: `V1_derivacion.md`. No se leyó ni importó código del trabajo original.

## Método
1. Derivación propia de H_ef por James–Jerke, con los términos rotantes (ω_q−ω) y contrarrotantes (ω_q+ω) de g_x y el término g_z σ_z (a+a†).
   El resultado coincide con R1, más una constante −g_z²/ω irrelevante. El término de pares sale del cruce entre g_x rotante y g_z:
   (g_xg_z/ω)[σ₊a, σ_z a] = −(2g_xg_z/ω)σ₊a². De los corrimientos, 3/4 viene de la rama rotante y 1/4 de la contrarrotante.
2. Diagonalización exacta del Hamiltoniano sin drive (ω=1, ω_q=2, N=40). Los niveles se identifican por máximo solapamiento con los estados desnudos.
   El doblete resonante {|g,2⟩, |e,0⟩} se trata aparte. El signo de pares se obtiene reconstruyendo el bloque efectivo 2×2 (ortonormalización de Löwdin).
   Diferencia con el original: el trabajo usó el marco rotante con Floquet; este chequeo espectral estático es nuevo.

## Resultados, g_z = 0

| g_x | δ_osc(g): exacto / predicho −4g_x²/3ω | rel | δ_osc(e) rel | δ_qubit rel |
|---|---|---|---|---|
| 0.005 | −3.3332e-5 / −3.3333e-5 | −3e-5 | −1e-4 | −4e-5 |
| 0.01 | −1.3332e-4 / −1.3333e-4 | −1.3e-4 | −4e-4 | −1.4e-4 |
| 0.02 | −5.3305e-4 / −5.3333e-4 | −5e-4 | −1.8e-3 | −6e-4 |
| 0.04 | −2.1288e-3 / −2.1333e-3 | −2e-3 | −7e-3 | −2.3e-3 |

El error relativo escala como g_x², es decir, es una corrección de cuarto orden: el coeficiente de segundo orden es exacto.

## Resultados, g_z ≠ 0

| g_x | g_z | δ_osc(g) rel | separación del doblete: exacta / √((4g_x²/ω)²+8G²) | rel |
|---|---|---|---|---|
| 0.01 | 0.01 | −4e-4 | 6.925e-4 / 6.928e-4 | −4e-4 |
| 0.01 | 0.03 | −2.8e-3 | 1.740e-3 / 1.744e-3 | −2.0e-3 |
| −0.02 | 0.02 | −1.7e-3 | 2.767e-3 / 2.771e-3 | −1.7e-3 |
| 0.02 | 0.04 | −5.3e-3 | 4.781e-3 / 4.800e-3 | −4.1e-3 |
| −0.01 | 0.05 | −7.6e-3 | 2.842e-3 / 2.857e-3 | −5.2e-3 |

Signo del elemento de pares ⟨g,2|H_ef|e,0⟩ frente a −√2·G (G = 2g_xg_z/ω), en los cuatro cuadrantes de signo:

| g_x | g_z | H_ef[g2,e0] | −√2 G | rel |
|---|---|---|---|---|
| +0.01 | +0.02 | −5.651e-4 | −5.657e-4 | −1.1e-3 |
| −0.01 | +0.02 | +5.651e-4 | +5.657e-4 | −1.1e-3 |
| +0.02 | −0.03 | +1.692e-3 | +1.697e-3 | −2.9e-3 |
| −0.02 | −0.03 | −1.692e-3 | −1.697e-3 | −2.9e-3 |

La energía absoluta E(g,0) + ω_q/2 coincide con −g_x²/(3ω) − g_z²/ω con error ≤ 2e-4.

## Validaciones
- Convergencia en N: pasar de N=40 a N=46 cambia E(g,1) − E(g,0) en 7e-15.
- No hay matrices densidad en esta verificación.

## Comparación con Ma et al., PRA 99, 022302 (2019), Ec. (4) (PDF: `msc/ma2019.pdf`)
Ma obtiene su Ec. (4) a partir de la Ec. (3), en la que descarta los contrarrotantes (RWA) y fija ω_p = δ = 2ω:

H_ef^Ma = (3g_x²/ω)|e⟩⟨e| + (2g_x²/ω) a†a |e⟩⟨e| − (2g_xg_z/ω)(σ₊a² + h.c.) + Ω(σ₊ + σ₋).

Mi derivación restringida a RWA da (g_x²/ω)[(n+1)|e⟩⟨e| − n|g⟩⟨g|] = (g_x²/ω)[(2n+1)|e⟩⟨e| − n]. Las diferencias son tres:
1. **Término de pares:** idéntico al de R1.
2. **Término −(g_x²/ω)n que multiplica a la identidad del qubit:** Ma lo omite. Ese término es el corrimiento del oscilador cuando el qubit está en |g⟩, y de ahí sale su δ_osc = 0.
   No se puede absorber en el marco rotante, porque el drive de pares debe resonar con 2(ω + δ_osc).
3. **Constante de |e⟩⟨e|:** Ma pone 3g_x²/ω, cuando en RWA sale g_x²/ω y en el cálculo completo 4g_x²/(3ω).
   La diagonalización exacta (1.3331e-4 para g_x=0.01) confirma 4/3 y descarta 3. No encontré ninguna derivación que dé 3.
   Coincide con el valor RWA para n=1 ((2n+1)=3), pero eso es especulación. Lo más probable es una errata o una suma incorrecta.

Además, al descartar los contrarrotantes Ma pierde el 1/4 restante de los corrimientos (factor 4/3 en lugar de 1).

## Veredicto
**Confirma R1:** δ_osc(g) = −4g_x²/(3ω), corrimiento del qubit +4g_x²/(3ω) y coeficiente de pares −2g_xg_z/ω, a precisión de cuarto orden.
Ma acierta en el coeficiente de pares, pero omite el corrimiento del oscilador y su corrimiento del qubit no se reproduce.

## Dudas y aproximaciones
- El corrimiento del qubit depende de n: vale (8n/3 + 4/3)g_x²/ω. Solo se verificó n = 0.
- La identificación del bloque del doblete por proyección introduce un error de O(g²), que es menor que las tolerancias reportadas.
- Naseem (arXiv:2508.10500) solo se revisó con grep, no se leyó; Liu no se ha leído. Ninguno de los dos se necesita para V1.
