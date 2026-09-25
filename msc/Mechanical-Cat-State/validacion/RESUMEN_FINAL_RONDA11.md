# Ronda 11 — Tareas 31-34: ¿qué produce el máximo de la brecha en Γ₂/κ≈0.13?

Tablas completas: `tarea31_34_resultados.md`; gráfico: `tarea32_escalera.png`. Código: `modelo_ladder.py`, `ladder_worker.py`,
`tarea33_worker.py`, `tarea34_worker.py`, `tarea31_34_analisis.py`, `run_ladder.sh`, `run_t33_34.sh`.

## Conclusión en una línea
**Ningún ingrediente estático (Δ_q, δ_m, Lamb, 1 fonón, Kerr, no resonantes) produce el máximo ni la saturación en 0.10-0.17: solo el propagador de Floquet lo reproduce, así que el origen es la dependencia temporal (términos contrarrotantes/no resonantes del Hamiltoniano completo) que el efectivo no captura; no es un efecto de la definición de brecha.**

## Tarea 31 (mínimo, buffer explícito N=20×2)
- α²=0 coincide con la fórmula exacta (dif. relativa máx 3.3e-5; 4 cifras), incluida la saturación en κ/4 desde Γ₂=κ/8.
- α²=2: brecha **monótona**, satura en κ/4=0.25 (desde Γ₂/κ≈0.49).
- Definición robusta (Tarea 27) y simple (excluir los 4 modos del código por peso en él) **idénticas en los 16 puntos**: la definición NO es el sospechoso.

## Tarea 32 (escalera, α²=2)
- Ningún paso (0→5) tiene máximo interior. Todos monótonos.
- La saturación en Γ₂/κ=1: mínimo 0.250; paso 1 (+Δ_q, δ_m) 0.239; paso 2 (+Lamb δ₁) 0.234; pasos 3-5 (1 fonón, Kerr residual, no resonantes) ≈0.234 sin cambio. **Solo baja a ~0.234, no a 0.10-0.17.**
- Prueba inversa: quitar Δ_q devuelve 0.250 (Δ_q es el único ingrediente que baja algo la saturación); quitar (δ_m+δ₁), 1 fonón, no resonantes o Kerr residual no cambia nada.
- δ_m sin Lamb (paso 1) empeora mucho la brecha a Γ₂ pequeño (0.0048 vs 0.0129 a Γ₂/κ=0.005): se cancela con δ₁ como esperado.
- Kerr: el Δ_q del buffer explícito ya genera el Kerr g²·ImS₂₋ por eliminación adiabática. Por eso el paso 4 añade solo la parte residual g²·ImS₂₊ (efecto nulo) y la variante literal 4L (Kerr completo encima, doble conteo) baja la saturación a 0.166 sin dar máximo. 4L es solo por completitud.
- El efectivo bosónico de `modelo_comun` (sin buffer) NO satura: la brecha crece sin límite (2.2 en Γ₂/κ=1), porque D[a²] con Γ₂ arbitrario no tiene la saturación del buffer.

## Tarea 33 (Floquet)
| Γ₂/κ | Floquet | paso 5 | efectivo |
|---|---|---|---|
| 0.03 | 0.0624 | 0.0652 | 0.0663 |
| 0.08 | 0.1331 | 0.1377 | 0.1767 |
| 0.13 | 0.1681 (Im −6.98) | 0.1774 | 0.2870 |
| 0.3 | 0.1288 (Im +13.0) | 0.2187 | 0.6622 |
| 1.0 | 0.0901 | 0.2338 | 2.2072 |
- **El paso (5) no reproduce el máximo y Floquet sí**; ambos coinciden (≤5%) solo hasta Γ₂/κ≈0.08. Sobre 0.13 la brecha de Floquet es un cuádruplete con Im≠0 (7-37) y overlap ~0 con P, n, a y con los operadores del qubit (Tarea 28), que no existe en el modelo estático. El máximo es el cruce entre la rama lenta (que sube como el paso 5) y ese cuádruplete (que baja).

## Tarea 34 (qubit térmico, Γ₂/κ=0.13, α²=2)
| n_q | pureza | \|⟨a²⟩\| (ideal 2) | P(qubit exc.) | P(código) | γ_pf | γ_bf | brecha |
|---|---|---|---|---|---|---|---|
| 0 | 0.500 | 1.997 | 1e-4 | 0.9998 | 7.7e-4 | 6.4e-7 | 0.168 |
| 0.1 | 0.303 | 1.834 | 0.105 | 0.840 | 2.5e-3 | 6.1e-3 | 0.179 |
| 0.3 | 0.139 | 1.539 | 0.252 | 0.584 | 5.8e-3 | 2.0e-2 | 0.098 |
| 0.6 | 0.057 | 1.016 | 0.387 | 0.319 | 1.2e-2 | 3.9e-2 | 0.049 |
(pureza 0.5 en n_q=0 = mezcla lógica máxima, estado estacionario.) **n_q=0.6 destruye el gato como qubit**: solo 32% de población en el espacio del código, |⟨a²⟩| baja a la mitad de α², γ_bf sube 5 órdenes de magnitud y supera a γ_pf (se pierde la protección contra bit-flip). Ya con n_q=0.1 γ_bf sube 4 órdenes (6e-7 → 6e-3) y supera a γ_pf.

## Validaciones y caveats (reportados, no ocultos)
- Escalera (145 estacionarios únicos): traza, hermiticidad y autovalores de ρ OK (<1e-10). El modelo mínimo tiene 4 modos con tasa 0 (estacionario no único): N/A.
- **Tarea 34, n_q=0: ||ρ−ρ†||=6e-10 > 1e-10** (falla la tolerancia pedida), causada por el autovalor casi degenerado del bit-flip (1−μ≈4e-9) y la precisión del propagador (rtol 1e-12). Los otros n_q cumplen (≤9e-12); autovalores mín. de ρ ≥ −7e-13.
- **Convergencia en N**: escalera exacta (Δrel ~1e-9). **Floquet en Γ₂/κ=0.13: 0.1681→0.1629 (Δrel 3.1%) al pasar a N=26: no está convergido a N=20**, así que el valor del máximo tiene ±3% de incertidumbre (la ubicación del cruce debe verificarse a N mayor). **Tarea 34 n_q=0.6: brecha 0.0488→0.0386 (21%) y |⟨a²⟩| 1.016→1.134 con N+6: NO convergida** (el calentamiento puebla Fock altos); la conclusión cualitativa (protección perdida) es robusta, los valores numéricos no.
- Siguiente paso sugerido: quitar de a uno los términos contrarrotantes de Floquet (drive de 4ω_r, acoples gx de 3ω_r) para identificar cuál sostiene el cuádruplete.
