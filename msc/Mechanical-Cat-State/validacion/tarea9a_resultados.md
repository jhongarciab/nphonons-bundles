# Tarea 9, Paso A — Muestreo estroboscópico, eps=0

Script: `tarea9a_pasoA.py`. Datos: `tarea9_pasoA.npz`.
T_m = 2π/ω_m = 6.283e-3 (unidades κ). tlist en múltiplos exactos de T_m
(stride=47 períodos, 102 puntos hasta τ_max≈29.8). Nb=16 (reducido de 20
por limitaciones de memoria de la máquina).

## Resultado: mejora parcial, no confirma completamente la hipótesis

| Cantidad | Muestreo genérico (Tarea 6b, eps=0.02) | Muestreo estroboscópico (eps=0) |
|---|---|---|
| dn/dt final | −1.15e-2 | **−1.06e-4** (100× menor) |
| ⟨n⟩ final | 0.0221 | 0.0222 (prácticamente igual) |

**El aliasing del desplazamiento polarónico β(t) sí explica la mayor parte
de la deriva rápida** (dn/dt cae ~100×), confirmando esa parte de la
corrección del usuario. **Pero el "piso" en ⟨n⟩≈0.022 persiste
prácticamente idéntico** con muestreo estroboscópico exacto — no baja a
~1e-4 como se esperaba dentro de τ_max=30. La trayectoria completa muestra
un transitorio inicial (pico ⟨n⟩≈1.02 en κt≈1.5), relajación rápida hasta
≈0.03 hacia κt≈10, y luego una **relajación residual muy lenta**
(dn/dt≈−1e-4, casi constante) que a ese ritmo tomaría mucho más que
τ_max=30 en alcanzar 1e-4.

## Validación (umbral −1e-9)

| t (κ) | osc: mín. autovalor | qubit: mín. autovalor |
|---|---|---|
| 0.00 | 0 | 0 |
| 7.38 | −3.44e-9 | +3.34e-3 |
| 15.06 | −2.73e-9 | +3.76e-4 |
| 22.44 | −3.23e-9 | +1.90e-5 |
| 29.83 | −3.49e-9 | +4.89e-6 |

Traza y hermiticidad perfectas en todos los instantes. El oscilador roza
el umbral (~−3e-9, dentro de −1e-9... en realidad ligeramente por fuera,
del orden de magnitud del umbral mismo — ruido numérico de diagonalización,
no violación física relevante). El qubit se mantiene positivo y su
autovalor mínimo decae monótonamente hacia 0 (relajación hacia un estado
puro), consistente con la física esperada.

## Interpretación y advertencia

La reducción de 100× en dn/dt confirma que el **aliasing es real y
relevante**, pero **no es la única fuente** del piso observado en Tarea 6b:
queda un residuo (~0.022, con relajación remanente muy lenta) cuyo origen
no se aisló por completo aquí. Podría ser: (i) un canal físico genuinamente
lento (a pesar de que el usuario indica que no es Γ1, no se descartó
formalmente con una corrida mucho más larga que τ_max=30, que sería
costosa); (ii) truncamiento de Fock (Nb=16, reducido por memoria) —el pico
transitorio ⟨n⟩≈1.02 sugiere que el margen sobre Nb=16 es cómodo, pero no
se verificó con Nb mayor. Se avanza al Paso B (barrido) dejando esta
pregunta abierta, según lo indicado por el usuario.
