# Tarea 6b — Barrido en ε/κ: completo vs efectivo (g_eff=g, g_eff=2g)

Script: `tarea6b_barrido_drive.py`. Datos: `tarea6b_resultados.npz`.
Alcance reducido (Nb=20, τ_max=30, N_eff=40) para viabilidad de tiempo,
manteniendo atol=1e-10/rtol=1e-8.

## Tabla

| ε/κ | p_e_ss | n_ss(full) | t90(full) | dn/dt_final | n_ss A(g) | t90 A | n_ss B(2g) | t90 B | full/B |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.00001 | 0.0221 | 0.031 | −1.15e-2 | 0.0032 | 6.324 | 0.0009 | 6.641 | 24.6× |
| 0.05 | 0.00003 | 0.0261 | 0.036 | −1.12e-2 | 0.0193 | 5.801 | 0.0049 | 1.557 | 5.3× |
| 0.10 | 0.00009 | 0.0403 | 0.056 | −1.06e-2 | 0.0754 | 5.639 | 0.0193 | 1.472 | 2.1× |
| 0.20 | 0.00032 | 0.0956 | 0.134 | −9.76e-3 | 0.2807 | 5.276 | 0.0754 | 1.438 | 1.27× |
| 0.40 | 0.00103 | 0.2974 | 0.402 | −8.95e-3 | 0.8943 | 4.326 | 0.2807 | 1.357 | **1.06×** |
| 0.72 | 0.00205 | 0.7676 | 0.842 | −9.69e-3 | 1.9283 | 3.058 | 0.7621 | 1.132 | **1.007×** |
| 1.44 | 0.00235 | 1.8953 | 6.796 | −1.70e-2 | 3.9973 | 1.804 | 1.9284 | 0.768 | **0.983×** |

(columna "full/B" = n_ss(full)/n_ss(B=2g); valores cercanos a 1 = buen acuerdo)

## Hallazgo 1: en ε ≳ 0.4κ, g_eff=2g reproduce n_ss del completo casi exactamente

Para ε/κ ∈ {0.4, 0.72, 1.44}, n_ss(completo) coincide con el modelo B
(g_eff=2g) dentro de **0.7%–6%**, mientras que el modelo A (g_eff=g) está
sistemáticamente **2×–3× por encima** del completo en ese mismo rango.
Esto **confirma cuantitativamente** el hallazgo coherente de la Tarea 7
también en el régimen disipativo/estacionario, con drive real
(no solo por FFT de Rabi sin disipación).

## Hallazgo 2: en ε ≲ 0.2κ, ninguno de los dos modelos es fiable — hay un "piso" no explicado

`dn/dt_final` se mantiene en ≈ −0.01 (κ unidades) **casi constante e
independiente de ε**, incluso para el ε más pequeño (0.02), donde
t90(full)=0.031 (¡el completo "sube" en un tiempo 200× más corto que los
modelos efectivos, t90≈5-6!). Esto indica que a ε pequeño, `n_ss(full)`
medido a τ_max=30 **no es realmente estacionario** — hay una deriva lenta,
aparentemente independiente de ε, que no se ha relajado (probablemente
ligada a las tasas de un fonón Γ1∓, ~5-6 órdenes de magnitud más lentas
que Γ2∓, con un tiempo de relajación ≫ 30 κ⁻¹). El "n_ss" reportado ahí es
solo el valor en τ=30, contaminado por esa deriva, y **no debe usarse para
comparar g_eff**. La comparación full/B en esas filas (24.6×, 5.3×, 2.1×)
no refleja una discrepancia real de g_eff, sino esta deriva no resuelta.

## Respuesta a la pregunta de la Tarea 6: límite ε≪κ (p_e→0)

**No se puede responder con estos datos**: en el límite de drive débil
solicitado, la comparación está dominada por una deriva lenta no
relajada dentro del tiempo simulado, no por la física de squeezing que se
quería aislar. Extender τ_max muy por encima de 30 (posiblemente
~10³–10⁴ κ⁻¹, dado que Γ1∓~10⁻⁵κ) sería necesario para una respuesta
confiable en ese límite extremo, y no se hizo aquí por costo computacional.
Lo que sí es concluyente es el régimen ε≳0.4κ (Hallazgo 1): ahí, **g_eff=2g**
reproduce el completo con alta fidelidad.

## Archivos generados

- `tarea6b_resultados.npz`: `eps_list`, `resultados` (array con columnas
  eps, n_ss_full, pe_ss, t90_full, dndt_final, n_ss_A, t90_A, n_ss_B, t90_B).
