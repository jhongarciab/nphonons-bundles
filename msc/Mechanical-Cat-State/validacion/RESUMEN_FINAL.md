# Resumen final — validación de fig1/fig2/fig4/fig5 (Naseem, PRA 113, 013732 (2026), arXiv:2508.10500v2)

Todos los scripts en `./validacion/`, derivados de los originales sin
modificarlos. Entorno: venv Python 3.11 + QuTiP 4.7.6 (los originales usan
`Options()`, removido en QuTiP 5).

## Tabla resumen

| Tarea | Pregunta | Resultado | Veredicto |
|---|---|---|---|
| 1 — Sanidad (fig2) | ¿Se reproduce fig2 (Ω=4g completo vs ε=2g efectivo)? | Traza y hermiticidad perfectas; positividad falla el umbral estricto 1e-10 por ruido numérico (~1e-6, tolerancia default del solver); F(κt=39)=0.9797, dip transitorio F_min=0.7306 en κt≈1. | **Reproducido** ✓ |
| 2 — Factor 2 (Ecs. 9 vs 11) | ¿g_eff debería ser 2g? | Modelo A (g, ε=2g, código original) reproduce ⟨n⟩ exacto en κt≈1 y n̄_ss dentro de 1.3% del completo. "Corregir" g→2g (Modelo B) preserva el n̄_ss pero rompe el transitorio; usar ε=Ω_full=4g sin corregir g (Modelo C) da n̄_ss erróneo ×2.1 y F final 0.82. | **Factor 2 NO confirmado** |
| 3 — Aislar 2 fonones (Ω=0, \|g,4⟩) | Medir g_eff directamente | Dinámica dominada por oscilaciones coherentes de 1 fonón (g_x=6κ, g_z=60κ), no markoviana limpia. Tendencia secular suavizada da Γ₂₋(medido)≈0.10, 5× más cerca de la fórmula con g_eff=g (0.518) que con g_eff=2g (2.074, 20× de diferencia). | **Refuerza rechazo del factor 2** (evidencia más débil, método con incertidumbre) |
| 4 — Bug Re_S1_plus (fig1/fig4) | ¿Corregir Δ1_minus→Δ1_plus cambia algo? | Confirmado bug (factor 9 en Re_S1_plus). Cambio en estados finales: tracedist ≤ 8.0e-4 (fig1), ≤ 1.9e-3 (fig4, todos n_th). | **Cambio despreciable, confirmado** ✓ |
| 5 — fig4: init térmico vs baño térmico | ¿Qué degrada más el cat state con n_th? | ⟨n⟩ idéntico (≈10) en todas las variantes. Baño térmico solo (init frío) apenas cambia negatividad/paridad (<2%). Estado inicial térmico (aun con baño frío) reduce negatividad ~2-5× y paridad ~2× respecto a partir de vacío. | **La degradación viene del estado inicial, no del baño** |

## Conclusión explícita: ¿el factor 2 (Ec. 9 vs Ec. 11) es real?

**No hay evidencia numérica de que g_eff deba duplicarse.** Dos líneas de
evidencia independientes (Tarea 2: comparación completo-vs-efectivo con
Ω=4g real; Tarea 3: medición directa del canal de dos fonones aislando el
drive) apuntan en la misma dirección: la normalización g_eff=g, ε=2g ya
usada en el código original es la que mejor reproduce la dinámica del
modelo completo (Ω=4g). Intentar aplicar el factor 2 sugerido por la
comparación literal Ec.(9)/Ec.(11) —ya sea doblando g_eff o usando el
drive real Ω como ε directamente— produce peor acuerdo (transitorio roto
en un caso, estado estacionario erróneo ×2 en el otro).

La discrepancia algebraica entre Ec.(9) y Ec.(11) del manuscrito es
probablemente **notacional** (una convención de signos/normalización que
se absorbe en un paso posterior de la derivación no visible en las dos
ecuaciones aisladas), no un error que se propague a los resultados
numéricos y figuras publicadas. Se recomienda, aun así, que los autores
verifiquen el álgebra simbólica completa entre esas dos ecuaciones para
documentar explícitamente de dónde sale la normalización final de g_eff,
dado que un lector que solo mire Ecs.(9) y (11) aisladamente concluiría
—incorrectamente, según esta validación numérica— que falta un factor 2.

## Hallazgos secundarios relevantes

1. El bug de `Re_S1_plus` (Δ1_minus en vez de Δ1_plus) es real pero
   físicamente inconsecuente (Tarea 4).
2. La calidad del cat state (negatividad de Wigner, paridad) en fig4 se
   degrada con n_th principalmente por la **mezcla del estado inicial**,
   no por el calentamiento del baño durante la generación (Tarea 5) —
   implicación práctica: enfriar el estado inicial importa más que aislar
   del baño durante el proceso.
3. La validación estricta de positividad (autovalor > −1e-10) falla
   sistemáticamente por ruido numérico de las tolerancias por defecto de
   QuTiP (`rtol=1e-6`, `atol=1e-8`), especialmente en regímenes con
   dinámica rápida/oscilatoria (Tarea 3, mín. autovalor −4.9e-5). No se
   interpreta como error físico, pero si se requiere el umbral estricto
   hay que apretar tolerancias (mayor costo computacional).

## Archivos

- `tarea1_fig2_reproduccion.py`, `tarea1_fig2_resultados.npz`, `tarea1_resultados.md`
- `tarea2_factor2.py`, `tarea2_factor2_resultados.npz`, `tarea2_resultados.md`
- `tarea3_dos_fonones.py`, `tarea3_dos_fonones_resultados.npz`, `tarea3_resultados.md`
- `tarea4_fig1_correccion.py`, `tarea4_fig1_resultados.npz`
- `tarea4_fig4_correccion.py`, `tarea4_fig4_resultados.npz`, `tarea4_resultados.md`
- `tarea5_fig4_separacion.py`, `tarea5_fig4_resultados.npz`, `tarea5_resultados.md`
- `RESUMEN_FINAL.md` (este archivo)
