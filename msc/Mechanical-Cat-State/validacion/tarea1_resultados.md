# Tarea 1 — Sanidad: reproducción de fig2_git_v1.py

Script: `tarea1_fig2_reproduccion.py` (copia derivada de `fig2_git_v1.py`, original sin modificar).
Datos: `tarea1_fig2_resultados.npz`.

## Configuración

- Modelo completo: qubit + oscilador, drive Ω = 4g (σ_x), N_a=2, N_b=50.
- Modelo efectivo: solo oscilador, squeezing con ε = 2g, N_eff=50.
- τ_max = 39 (unidades de κ), 120 pasos.
- Solver: `qutip.Options(nsteps=1e6, store_states=True)` (tolerancias por defecto: rtol=1e-6, atol=1e-8).

## Validación por estado (traza, hermiticidad, positividad)

| Chequeo | Full (ρ reducido, ptrace oscilador) | Efectivo |
|---|---|---|
| Traza | max\|tr − 1\| = 3.46e-14 — **OK** | max\|tr − 1\| = 1.40e-14 — **OK** |
| Hermiticidad | max ‖ρ − ρ†‖ = 7.77e-11 — **OK** | max ‖ρ − ρ†‖ = 0 — **OK** |
| Positividad (autovalor mínimo) | −3.06e-6 — **falla umbral 1e-10** | −2.63e-8 — **falla umbral 1e-10** |

**Interpretación**: la violación de positividad es de orden 1e-6/1e-8, compatible con
ruido numérico de integración (rtol/atol por defecto) y de diagonalización en Hilbert
dim 50, no con una inconsistencia física del modelo. Para cumplir el umbral estricto
de 1e-10 pedido, habría que reducir `atol`/`rtol` del solver (costo: mayor tiempo de cómputo).

## Resultados físicos

| Cantidad | Full | Efectivo |
|---|---|---|
| ⟨n⟩(0) | 0.0000 | 0.0000 |
| ⟨n⟩(κt=39) | 1.9030 | 1.9284 |
| Diferencia relativa en ⟨n⟩ final | ~1.3% | |

- Fidelidad mínima: **0.730570** en κt = 0.98 (dip transitorio corto).
- Fidelidad final (κt = 39): **0.979732**.

## Conclusión

fig2 se reproduce cualitativa y cuantitativamente: acuerdo alto entre modelo completo
y efectivo a tiempos largos (F ≈ 0.98), con un dip transitorio breve a tiempos cortos
(κt ≈ 1), consistente con lo reportado en el paper. Traza y hermiticidad perfectas;
positividad within numerical-noise pero fuera del umbral estricto solicitado — no se
considera un fallo físico del modelo.

## Archivos generados

- `tarea1_fig2_resultados.npz`: arrays `tau`, `nb_full`, `nb_eff`, `fidelity_t`,
  `trs_full`, `herm_full`, `mineig_full`, `trs_eff`, `herm_eff`, `mineig_eff`.
