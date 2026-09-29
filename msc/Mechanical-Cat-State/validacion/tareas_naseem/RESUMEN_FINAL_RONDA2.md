# Resumen final — Ronda 2 de validación (Tareas 6-8)

Continuación de `RESUMEN_FINAL.md` (Tareas 1-5). Esta ronda **revisa y
corrige** las conclusiones de la Tarea 2/3 anteriores sobre el factor 2,
que el usuario correctamente identificó como no concluyentes. Todos los
mesolve de esta ronda usan `atol=1e-10, rtol=1e-8`; validación de
positividad con umbral −1e-9.

## Tabla por tarea

| Tarea | Método | Resultado | Veredicto sobre el factor 2 |
|---|---|---|---|
| 6a — Saturación (Ω=4g) | Modelo completo, p_e_ss vs fórmula de saturación resonante | p_e_ss medido=0.0021 vs fórmula=0.472 (99.6% de diferencia) | N/A — muestra que el qubit está fuertemente hibridizado con la mecánica (g_x=6κ, g_z=60κ ≫ κ, ε), no es un qubit libre saturable |
| 6b — Barrido ε/κ | n_ss(completo) vs efectivo g_eff=g, 2g | Para ε≳0.4κ: g_eff=2g coincide con el completo dentro de 0.7-6%; g_eff=g está 2-3× por encima. Para ε≲0.2κ: comparación no fiable (deriva lenta no relajada) | **A favor de g_eff=2g** (en el régimen donde la medición es fiable) |
| 7 — Rabi de 2 fotones (unitario, sin disipación) | Frecuencia de oscilación |g,2⟩↔|e,0⟩ medida por FFT | Ω_R medido coincide con 2√2·(2g) dentro de <5% en los 3 valores de g_z probados; escala g_eff∝g_z confirmada (var. 6%) | **Fuertemente a favor de g_eff=2g** |
| 8 — Álgebra (transformación polarón) | e^S H e^{-S} hasta 2do orden, verificado con matrices exactas | Coeficiente medido = −0.720j (BCH-2) / −0.702j (exacto) = Ec.(9) exactamente. Ec.(11) predice +0.36j — factor −2 de diferencia | **Confirma algebraicamente que Ec.(9) es correcta y Ec.(11) tiene un error de factor 2** |

## Conclusión explícita: distinguiendo normalización de g_eff vs. saturación del qubit

### (a) Normalización de g_eff — SÍ hay un factor 2 real

A diferencia de la Ronda 1 (Tareas 2-3, con métodos indirectos —
comparación completo-vs-efectivo con drive externo y disipación, o
medición de tasas en presencia de acoplamiento fuerte de 1 fonón—, ambos
confundidos por efectos que no aislaban limpiamente g_eff), esta ronda usa
**tres métodos independientes** que sí aíslan la cantidad de interés:

1. **Álgebra exacta** (Tarea 8): la transformación tipo polarón, verificada
   con commutadores exactos Y exponenciación matricial completa (sin
   atajos), da el coeficiente `-2g` para el término σ_{+-}a²,a†², idéntico
   a la Ec. (9) del manuscrito. La Ec. (11), como se usa para derivar el
   modelo efectivo, tiene coeficiente `+g` — un error de factor −2.
2. **Medición coherente directa** (Tarea 7): la frecuencia de Rabi de dos
   fotones observada en la dinámica unitaria exacta (sin disipación, sin
   drive externo, sin aproximación markoviana) coincide con `2√2·(2g)`,
   no con `2√2·g`.
3. **Ajuste del estado estacionario disipativo** (Tarea 6b), en el régimen
   donde la medición es fiable (ε≳0.4κ): `g_eff=2g` reproduce n̄_ss del
   modelo completo dentro de pocos %; `g_eff=g` sobreestima por 2-3×.

**Los tres métodos convergen: g_eff = 2g es el valor correcto.** Las
conclusiones de la Ronda 1 (Tareas 2-3), que favorecían g_eff=g, eran
artefactos de: (i) en la Tarea 2, comparar con Ω_full=4g fijo sin escanear
el régimen de drive, con el ansatz de comparar fidelidad/tiempos de subida
que no aislaba limpiamente la normalización de g del efecto de la
saturación del qubit ni del drive; (ii) en la Tarea 3, medir en un régimen
de acoplamiento fuerte (g√(n(n-1))~1.25κ, señalado por el usuario) donde
el ansatz markoviano simplemente no aplica y las oscilaciones coherentes
de 1 fonón dominan sobre la señal de interés.

### (b) Efecto de la saturación del qubit — separado y real, pero no es "el factor 2"

La Tarea 6a muestra que el qubit, en este sistema, **no se comporta como
un qubit libre driveado y disipativo**: la fórmula estándar de saturación
`ε²/(2ε²+κ²/4)` falla por 2 órdenes de magnitud (predice p_e_ss≈0.47,
la dinámica real da p_e_ss≈0.002). Esto es porque g_x=6κ y g_z=60κ son
**mucho mayores** que κ y que ε — el qubit está fuertemente vestido/
hibridizado con la mecánica, y su población "desnuda" excitada queda muy
suprimida respecto a la imagen de un qubit aislado. Este es un efecto
**físicamente distinto** del factor 2 en g_eff: no cambia la normalización
del acoplamiento efectivo de dos fonones, pero sí significa que **si se
quisiera usar la fórmula de saturación para estimar límites de validez del
modelo (p.ej. "¿hasta qué ε el modelo efectivo es válido?"), esa fórmula
no sirve aquí** — habría que usar el p_e_ss real (mucho menor) medido
directamente, lo cual en la práctica hace el rango de validez de la
aproximación de qubit-poco-excitado **más amplio** de lo que la fórmula
ingenua sugeriría (el qubit se satura mucho menos de lo esperado).

## Conclusión combinada

1. **Sí hay un factor 2 real en la normalización de g_eff**: debe usarse
   g_eff=2g (Ec. 9) en vez de g_eff=g (como en Ec. 11 / como estaba
   implícito en `fig1/fig2/fig4/fig5_git_v*.py`). Esto es un hallazgo que
   **contradice y reemplaza** la conclusión de la Ronda 1.
2. La saturación del qubit es un fenómeno aparte, dominado por el
   acoplamiento fuerte qubit-mecánica (no por el drive), y no debe
   confundirse con la cuestión del factor 2.
3. Persisten dos limitaciones abiertas: (i) el régimen de drive débil
   (ε≲0.2κ) no se pudo validar por una deriva lenta no relajada dentro del
   tiempo de simulación usado (Tarea 6b); (ii) no se verificó si aplicar
   g_eff=2g en las figuras publicadas (fig1/fig2/fig4/fig5) con el drive ε
   tal como está definido allí requeriría TAMBIÉN re-escalar ε de forma
   consistente (dado que en Ec.(9)/(11) ambas cantidades, g y el drive
   efectivo, podrían estar acopladas en la derivación completa que no se
   tuvo a la vista, solo las dos ecuaciones aisladas que dio el usuario).

## Archivos generados esta ronda

- `tarea6a_fig2_saturacion.py`, `tarea6a_resultados.npz`, `tarea6a_resultados.md`
- `tarea6b_barrido_drive.py`, `tarea6b_output.log`, `tarea6b_resultados.npz`, `tarea6b_resultados.md`
- `tarea7_rabi_dos_fotones.py`, `tarea7_resultados.npz`, `tarea7_resultados.md`
- `tarea8_algebra_polaron.py`, `tarea8_resultados.npz`, `tarea8_resultados.md`
- `RESUMEN_FINAL_RONDA2.md` (este archivo)
