# Tarea 20 — Identificar los modos lentos (punto del paper, ε=1.44)

Script: `tarea20_modos_lentos.py`. Datos: `tarea20_resultados.npz`.
Nb=20, propagador de Floquet (58s de cómputo total).

## Tabla de clasificación

| k | λ (κ, parte real) | \|μ_k\| | Clasificación | Overlap dominante |
|---|---|---|---|---|
| 0 | 0 (trivial) | 1.000000 | punto fijo | — |
| 1,2 | 4.822e-4 | 0.999997 | **a (coherencia lógica/bit-flip)** | overlap con a: 1.34-1.39 (≫ resto) |
| **3** | **6.912e-4** | 0.999996 | **P (phase-flip)** | overlap con P: 0.992 (dominante) |
| **4** | **9.911e-2** | 0.999377 | **n (confinamiento)** | overlap con n: 2.944 (dominante) |
| 5 | 0.1578 | 0.999009 | n (confinamiento) | overlaps todos ~1e-6 (modo casi ortogonal a la base de referencia; oscila rápido, Im(λ)≈±72) |
| 6,7 | 0.1579-0.1609 | 0.999 | a² (confinamiento) | overlaps pequeños, ligera preferencia por a² |

## Confirmación explícita

**El modo 4 (λ≈0.099) SÍ es la brecha de confinamiento** — overlap
dominante con el operador número n (2.944, muy por encima de los demás:
a²=0.894, σ_z=0.470, P=0.044, a=0.056).

## Jerarquía de escalas de tiempo identificada

| Canal | λ (κ) | τ=1/λ (κ⁻¹) |
|---|---|---|
| **Bit-flip / coherencia lógica** (modo 1,2) | 4.82e-4 | 2074 |
| **Phase-flip** (modo 3) | 6.91e-4 | 1447 |
| **Confinamiento** (modo 4) | 9.91e-2 | 10.1 |
| Confinamiento/oscilaciones rápidas (modos 5-7) | 0.158-0.161 | ~6.3 |

**El canal de bit-flip (coherencia lógica, overlap con `a`) es incluso
más lento que el de phase-flip** (4.82e-4 vs 6.91e-4, τ=2074 vs 1447
κ⁻¹) — el cat state, una vez formado, es extraordinariamente estable
frente a la pérdida de coherencia lógica (mucho más que frente al
phase-flip, que a su vez es ~100× más lento que la brecha de
confinamiento). Esta jerarquía (confinamiento ≪ phase-flip ≲ bit-flip
en tiempo, es decir τ_confinamiento ≪ τ_phase-flip ≲ τ_bit-flip) es
exactamente la estructura deseable para un cat state usado como qubit
lógico: el proceso más rápido es simplemente que el estado se "asiente"
en el subespacio de dos fonones (confinamiento, τ~10κ⁻¹), y una vez ahí,
los dos canales de error lógico (bit-flip y phase-flip) son ambos
extremadamente lentos (τ~1500-2000 κ⁻¹).

## Conclusión Tarea 20

La estructura espectral del propagador de Floquet en el punto de
operación del paper revela una separación de escalas de tres niveles:
(1) formación/confinamiento del gato (τ~10κ⁻¹, rápida), (2) phase-flip
(τ~1447κ⁻¹) y (3) bit-flip/coherencia lógica (τ~2074κ⁻¹, la más lenta de
todas). El modo 4 (λ≈0.099) queda confirmado sin ambigüedad como la
brecha de confinamiento, con el overlap dominante sobre el operador
número, muy por encima de cualquier otro candidato.
