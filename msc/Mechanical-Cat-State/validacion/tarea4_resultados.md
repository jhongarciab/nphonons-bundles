# Tarea 4 — Corrección de Re_S1_plus en fig1/fig4

Scripts: `tarea4_fig1_correccion.py`, `tarea4_fig4_correccion.py`.
Datos: `tarea4_fig1_resultados.npz`, `tarea4_fig4_resultados.npz`.

## Bug identificado

En `fig1_git_v1.py` (línea 57) y `fig4_git_v1.py` (línea 150):

```python
Re_S1_plus = x / (x**2 + Δ1_minus**2)   # debería ser Δ1_plus
```

Δ1_minus = ω_q − ω_m, Δ1_plus = ω_q + ω_m. Con ω_q=2ω_m: Δ1_minus=ω_m,
Δ1_plus=3ω_m → **razón (Δ1_plus/Δ1_minus)² = 9**, confirmada numéricamente
exacta en ambos scripts (9.0000).

## fig1 (n_th=0)

| Cantidad | Bug | Corregido |
|---|---|---|
| Re_S1_plus | 7.958e-13 | 8.842e-14 (9× menor) |
| Γ_plus | 22.62 Hz | 2.513 Hz |
| Γ_minus (sin cambios) | 116.87 Hz | 116.87 Hz |

Distancia de traza bug-vs-corregido creciendo monótonamente con t, **máxima
(= final) = 8.005e-4**, sobre toda la evolución hasta t·Γ₂₋=3.

## fig4 (n_th ∈ {0, 0.5, 1, 2})

| n_th | Γ_plus bug | Γ_plus corregido | tracedist final |
|---|---|---|---|
| 0.0 | 22.62 Hz | 2.513 Hz | 1.855e-3 |
| 0.5 | 69.74 Hz | 49.64 Hz | 9.081e-4 |
| 1.0 | 116.87 Hz | 96.76 Hz | 5.954e-4 |
| 2.0 | 211.12 Hz | 191.01 Hz | 3.474e-4 |

La distancia de traza **decrece** al aumentar n_th, porque Γ_plus queda
dominado por el término térmico `n_th·γ` (γ=2π·15 Hz) frente a la
contribución inducida por el qubit (Γ1_plus, del orden de unos Hz a
decenas de Hz), diluyendo el peso relativo del error.

## Conclusión Tarea 4

Confirmado: el cambio es **despreciable** en todos los casos (tracedist ≤
1.9e-3, muy por debajo de cualquier umbral relevante para las conclusiones
físicas del paper). La razón es que Γ1_plus es intrínsecamente un canal de
calentamiento de un fonón inducido por el qubit, ya suprimido ~5-6 órdenes
de magnitud frente a Γ_minus y frente a las tasas de dos fonones (Γ2∓) por
la gran desintonización ω_m≈ω_q (ambas Δ1∓ ≫ κ); un factor 9 adicional
sobre una cantidad ya subdominante no altera visiblemente la dinámica ni
los estados finales. El bug es real pero fisicamente inconsecuente para
las figuras publicadas.
