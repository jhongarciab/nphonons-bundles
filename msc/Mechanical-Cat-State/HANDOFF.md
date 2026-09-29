# HANDOFF — paper de generalización: qubits gato estabilizados por un qubit auxiliar con acoplamientos (g_x, g_z)

Documento de traspaso para organizar el escrito (PRA, autor único). Consolidado el 2026-09-29 a partir del HANDOFF del chat y del estado real del repositorio.
Idioma: español. **[VERIFICAR]** marca lo que no se ha podido comprobar contra archivos. La sección 11 (superado) prevalece sobre cualquier documento viejo.

Fuentes vivas en este repositorio (`msc/Mechanical-Cat-State/`):
- `figuras_finales/README.md`: métodos, parámetros, cifras y validaciones de cada figura.
- `figuras_finales/PENDIENTES_Y_HALLAZGOS.md`: pendientes, discrepancias y correcciones adoptadas (P1–P10, E1–E8).
- `verificacion_independiente/V1..V6_reporte.md`, `V4b_reporte.md`: verificación independiente de R1–R6.
- `figuras_finales/C6_polaron.md`: base polarónica.

---

## 0. Índice
1. Autor y preferencias · 2. Idea y mensaje · 3. Modelo y notación · 4. Derivaciones · 5. Resultados consolidados ·
6. Errores de la literatura · 7. Plataformas · 8. Térmico · 9. Criterios numéricos · 10. Inventario de archivos ·
11. Superado y retirado · 12. Pendientes · 13. Lo que falta para escribir · 14. Convenciones · 15. Riesgos

---

## 1. Autor y preferencias
- Jhon Sebastián García Barrera (MSc Ciencias Físicas, UNAL Manizales). Director: Edgar Gómez. Escribe en español.
- Objetivo: paper para Physical Review A. Familia de esquemas: Ma 2019, Hou 2024, Liu 2025, Naseem 2026.
- Exigencias: derivaciones correctas y verificación numérica independiente; nada inventado; revisar solo la física (el estilo visual lo hace Jhon).
- LaTeX: clase report; sin \textbf, \textit ni \emph; sin rayas (em-dash) en la prosa; derivaciones paso a paso; resultados en caja; .tex y .bib juntos; sin \newpage; babel/babelprovide y onehalfspacing.
- Trabajo: dialogar los cambios antes de editar; un comando de terminal a la vez; honestidad sobre los riesgos de credibilidad.
- Código: Python/QuTiP completo, comentarios en español, validar traza, hermiticidad y positividad de ρ, respetar γ ≪ κ.

## 2. Idea y mensaje
Un oscilador acoplado a un qubit con g_xσ_x + g_zσ_z, conducido cerca de la resonancia de dos fotones, genera un intercambio de pares G = 2g_xg_z/ω. Con el qubit disipativo, eso da pérdida de pares κ₂ = 4G²/κ, que estabiliza un gato.
Se derivan y verifican: el hamiltoniano efectivo correcto (con los corrimientos que la literatura omite), la resonancia vestida, la tasa de phase-flip, la figura de mérito, el baño filtrado, el confinamiento y el límite térmico.

**Mensajes centrales:**
1. κ₁/κ₂ = (5/72)(κ/g_z)² no depende de g_x ni de ω. Contradice la receta de Naseem de aumentar κ.
2. El gato solo se forma en la resonancia vestida ω_p* = 2(ω − 4g_x²/3ω), no en 2ω.
3. Un filtro a 2ω baja el umbral en g_z/κ en un factor ≈ κ_f/2ω, hasta donde lo permite la pérdida intrínseca.

## 3. Modelo y notación
H(t) = ω a†a + (ω_q/2)σ_z + (a + a†)(g_xσ_x + g_zσ_z) + Ω(σ₊e^{−iω_pt} + h.c.), con ω_q ≈ 2ω y σ_z = |e⟩⟨e| − |g⟩⟨g|.
- Disipación a T = 0: κD[σ₋] y γD[a], con D[o]ρ = oρo† − ½{o†o, ρ}.
- Disipación térmica: κ[(n_q+1)D[σ₋] + n_qD[σ₊]] y γ[(n_m+1)D[a] + n_mD[a†]].
- Definiciones: G = 2g_xg_z/ω, κ₂ = 4G²/κ, α² = Ω/G (típicamente |α|² = 4), κ₁ = Γ₁⁻ + Γ₁⁺, χ = (8/3)g_x²/ω.
- Régimen de validez: γ ≪ κ; κ, g ≪ ω (Liu viola κ/ω = 0.9); χ|α|² ≲ 0.3κ (P9).

## 4. Derivaciones (paso a paso en `verificacion_independiente/V1_derivacion.md`)

### 4.1 Hamiltoniano efectivo (James–Jerke, segundo orden)
Componentes de la interacción:
- a frecuencia ω: g_xσ₋a† (rotante) y g_zσ_za;
- a frecuencia 3ω: g_xσ₋a (contrarrotante).

Con H_ef = Σ (1/ω_k)[h_k†, h_k] y los cruces resonantes:

  **H_ef = (g_x²/ω)[(4n/3 + 1)|e⟩⟨e| − (4n/3 + 1/3)|g⟩⟨g|] − G(σ₊a² + h.c.) − g_z²/ω**

- Corrimiento del oscilador en |g⟩: −4g_x²/3ω. El del qubit con n = 0 es +4g_x²/3ω; 3/4 viene de la rama rotante y 1/4 de la contrarrotante.
- **Verificado por diagonalización exacta (V1):** el error relativo escala como g² (cuarto orden). El signo del término de pares se comprobó en los cuatro cuadrantes de signo.

### 4.2 Resonancia vestida
El estado oscuro |ψ,g⟩ con a²|ψ⟩ = (Ω/G)|ψ⟩ existe si la desintonía del oscilador en |g⟩ se anula, es decir en ω_p* = 2(ω − 4g_x²/3ω).
- Ma: con ω_p = 12, P_c ≈ 0.81 (no hay gato); con 11.98, P_c = 0.9975. Máximo medido 11.97993 ± 0.0005, predicho 11.9800 (V2).
- **Regla universal:** en x = (ω_p − ω_p*)/G el pico cae en x ≈ 0 (|x| ≤ 0.09, salvo un caso con 0.16).
  FWHM ≈ 3G con |α|² = 4 y ≈ 3.7–4.2G con |α|² = 2; la asimetría crece con κ₂/κ (P1, colapso parcial aceptado).
  Con ω_q = ω_p: FWHM = cG, c = 2.93 ± 0.25 (`figuras_finales`, Fig. 2 de validación). Tarea 47(d), con ω_q fijo: c = 2.87 ± 0.25.

### 4.3 Tasas efectivas (Reiter–Sørensen)
- Pares: κ₂ = 4G²/κ.
- Un fonón: Γ₁⁻ = g_x²κ/(ω² + κ²/4) (pérdida, vía ω) y Γ₁⁺ = g_x²κ/(9ω² + κ²/4) (ganancia, vía 3ω).
- Corrimiento con κ: δ₁ = −(4g_x²/3ω)[1 − (7/9)(κ/2ω)²] (Tarea 41).

### 4.4 Paridad (phase-flip)
D[a]†P = −2nP y D[a†]†P = −2(n+1)P ⇒ d⟨P⟩/dt = −2Γ₁⁻⟨nP⟩ − 2Γ₁⁺⟨(n+1)P⟩.

  **γ_pf = 2[Γ₁⁻|α|² + Γ₁⁺(|α|² + 1)] (+ 2γ|α|²)**, con |α|² = |⟨(a − g_z/ω)²⟩|.

- El "+1" faltaba en el trabajo original (E2). Verificado en V4b: con |α|² = 2, 4 y 6 da 0.996–0.999.
- Hay un piso de paridad del estado estacionario (~7e-4 con |α|² = 4). Por eso los ajustes temporales se hacen como A e^{−kt} + c.

### 4.5 Figura de mérito
κ₂ = 16g_x²g_z²/(ω²κ) ⇒ **κ₁/κ₂ = (5/72)(κ/g_z)²**.
- Es un cociente de tasas de Lindblad. El +1 pertenece a γ_pf, no a κ₁/κ₂.
- Precisión (corrección no adiabática): ≲ 0.4% para κ₂/κ ≲ 0.05, ~1% en 0.2 y ~2% en 1 (P4, sin explicación analítica).
- Umbral de Guillaud–Mirrahimi κ₂/κ₁ = 220 ⇒ g_z/κ ≥ √(220·5/72) = 3.909, solo por este canal.

### 4.6 Baño filtrado
κ_eff(δ) = κκ_f²/(4δ² + κ_f²), con 4J²/κ_f = κ ⇒ κ₁^filt = g_x²[κ_eff(ω)/ω² + κ_eff(3ω)/(9ω²)].
- Verificado canal por canal al 0.2% (V5, Fig. 4 de validación).
- **Mejora de γ_pf respecto al baño plano, en unidades de Ma (ω = 6, ω/κ = 200), para κ_f = 3, 1, 0.3 y 0.1:** 19.5, 166.5, 1838 y 16 528.
  Coincide con la predicción que incluye el +1 de C1 al ≤ 0.3%. Resuelve el [VERIFICAR] del HANDOFF del chat: ω = 6, sin "ω efectivo".
- **Umbral en g_z/κ:** baja de 3.94 a 0.094 (factor 0.0238; predicción √(f/(10/9)) = 0.0239 ≈ κ_f/2ω). El confinamiento con filtro, Δ_f/Δ_p, está en `figuras_finales/README.md`.

### 4.7 Confinamiento
- **Modelo mínimo con α = 0 (exacto, V6):** Δ = (κ/4)[1 − √(1 − 8Γ₂/κ)] si Γ₂ ≤ κ/8, y κ/4 por encima (punto excepcional). Límite adiabático Δ ≈ Γ₂ + 2Γ₂²/κ.
- **Con |α|² = 4 (tasa dinámica, C9):** c = lím Δ/κ₂ = 4.14. La saturación empieza en κ₂/κ ≈ 1/(4c) = 0.060. Satura en ≈ 0.28κ, que no es un techo estricto.
  κ₂^eff/κ₂ = 0.98, 0.84, 0.50, 0.24 y 0.058 para κ₂/κ = 1e-3, 0.01, 0.045, 0.16 y 1.
- La tasa espectral no converge en el régimen saturado (rama interior); la relevante es la **dinámica**.

### 4.8 P9: corrimiento del qubit dependiente de n
χn|e⟩⟨e| con χ = (8/3)g_x²/ω frena el confinamiento. Con κ₂/κ = 16(g_x/ω)²(g_z/κ)²:
  χ|α|²/κ = (|α|²/6)(ω/κ)(κ₂/κ)/(g_z/κ)².
- Con |α|² = 4, la frontera χ|α|²/κ = 0.3 es **κ₂/κ = 0.45(g_z/κ)²/(ω/κ)**. Esta es la frontera, no una fórmula para κ₂/κ; resuelve el [VERIFICAR] del chat.
- Pérdida de confinamiento del 5% en χ|α|²/κ ≈ 0.3. El modelo efectivo con χ reproduce el completo al 0.5–4%.
- Valores: Ma 2.7 (fuera), Naseem 0.19 (dentro). La frontera también vale con filtro: el punto (0.25, 6), con χ|α|²/κ = 0.93, confina 0.75 veces lo del mapa.

### 4.9 Base polarónica (C6)
Con el qubit en |g⟩ el oscilador se desplaza d = +g_z/ω (laboratorio, t = nT_p). P_c usa el **código fijo** D(g_z/ω)|±α_nom⟩; α_eff solo entra en γ_pf.

### 4.10 Piso intrínseco
κ₁ → κ₁^filt + γ. Como ε ≥ γ/κ₂^eff, llegar al umbral de 220 exige κ₂^eff/κ ≥ 220γ/κ.
- Con γ/κ = 2e-4: franja κ₂/κ ≈ 0.19–0.33, y solo con g_z/κ ≳ 9.3–11.9 dentro de la zona χ válida.
- Verificado con el modelo completo en (0.25, 14), N = 22: ε = 4.39e-3 < 1/220, y ε_completo/ε_mapa = 1.028.

---

## 5. Resultados consolidados (cifras y archivo de respaldo)
| Resultado | Valor | Respaldo |
|---|---|---|
| H_ef | §4.1 | V1 |
| ω_p* | 2(ω − 4g_x²/3ω) | V2; `figuras_finales` Fig. 2 |
| γ_pf con +1 | 0.996–0.999 de la fórmula | V4b |
| κ₁/κ₂ | (5/72)(κ/g_z)², 19 puntos, 0.979–0.999 | V4, `data/principal_fig2.csv` |
| Filtro | mejoras al ≤ 0.3%; umbral 3.94 → 0.094 | V5, `figura_central` |
| Brecha con α = 0 | exacta al ≤ 2e-13 | V6 |
| Confinamiento con \|α\|² = 4 | c = 4.14, saturación ≈ 0.28κ | `data/principal_fig3_minimo.csv` |
| Frontera χ | χ\|α\|²/κ ≈ 0.3 | `data/p9_diagnostico.csv` |
| Mapa con filtro | ±2–3% frente al completo (N = 22) | `data/figura_central_verificacion.csv`, README |

## 6. Errores de la literatura
- **Ma 2019, Ec. (4):** omite −(g_x²/ω)n y da 3g_x²/ω para el corrimiento del qubit (el correcto es 4g_x²/3ω). El término de pares es correcto (V1, E1).
- **Liu 2025, Ec. (11):** omite −(4g_x²/3ω)n. Hay una inconsistencia probable del factor del drive (ε_p frente a 2ε_p). Con su Ec. (9) completa el gato no se forma con κ/ω = 0.9 (Tareas 46 y 47b).
- **Naseem 2026:** no incluye δ₁ y usa g_eff = g en lugar de 2g (Tareas 7–8). Su receta de aumentar κ contradice κ₁/κ₂ ∝ κ².
- **Hou 2024:** pendiente de lectura.

## 7. Plataformas (tabla pendiente de rehacer, §12)
| Plataforma | κ/ω | g_z/κ | κ₂/κ | κ₁/κ₂ | χ\|α\|²/κ | notas |
|---|---|---|---|---|---|---|
| Ma | 0.005 | 7.1 | 1.0 | 1.4e-3 | 2.7 | fuera de la frontera de χ; γ/κ = 2e-5 (Q = 1e7) [VERIFICAR Q] |
| Naseem | 0.001 | 60 | 2.07 | 9.2e-5 | 0.19 | f_q = 200 MHz ⇒ x = 0.96 a 10 mK; γ/κ = 1.5e-4; térmico no calculado |
| Liu | 0.90 | 0.22 | 0.031 | — | — | fuera de validez; el gato no se forma |

## 8. Térmico (P7) — detalle en `figuras_finales/README.md`
- **Definiciones:** η = γ_pf/γ_bf (Tarea 37); x = hf_q/k_BT con f_q la frecuencia del **qubit**; n_q = 1/(eˣ−1), n_m = 1/(e^{x/2}−1); k_BT/h = 208.37 MHz a 10 mK.
- **Modelo efectivo** (qubit y filtro explícitos) en la isla: κ₂/κ = 0.25, g_z/κ = 14, |α|² = 4, ω/κ = 200.

| baño | γ/κ | x*(η=100) | x*(η=220) |
|---|---|---|---|
| filtrado | 2e-5 | 10.11 | 10.91 |
| filtrado | 2e-4 | 7.78 | 8.59 |
| plano | 2e-5 | 8.19 | 9.01 |
| plano | 2e-4 | 7.18 | 8.00 |

- **Convergencia:** N = 22 → 24 cambia x* ≤ 0.008.
- **El filtro endurece el requisito** porque reduce γ_pf; γ_bf/n_q solo sube un 28%.
- **Bit-flip térmico:** γ_bf = c_bf n_qκ, con c_bf ∝ (G/κ)^s: s = 3.96 ± 0.04 con filtro y 3.69 ± 0.05 en el plano. **No** es 0.05 constante.
- **Supresión con |α|²:** d ln γ_bf/d|α|² ≈ −0.55 a −0.64 en el plano (coincide con Gautier/Tarea 44 para g₂/κ ≈ 0.3) y ≈ −0.42 con filtro.
- η = 100 y 220 son **valores de referencia**: 220 es un umbral sobre κ₂/κ₁, no sobre η.
- **VALIDACIÓN CON EL MODELO COMPLETO — ABIERTA (bloquea la figura térmica):**
  - En (0.25, 14), x = 10.1, N = 22, el modelo completo en el laboratorio da γ_pf = 1.56e-4, plausible frente al efectivo.
  - Pero no muestra un modo lento de pozo: el modo con λ ≈ −1 decae a ≈ 4e-2 (N = 12) y 7e-2 (N = 22), frente a γ_bf ≈ 1e-6 del efectivo.
  - La selección del modo se corrigió en `calc_filtro_completo.py` y ahora se guardan los 24 modos lentos. La discrepancia persiste, así que no es de selección.
  - Diagnóstico en curso: T = 0, con y sin γ. Hasta resolverlo, los umbrales térmicos no están validados.
  - Además, la hermiticidad cruda de esa corrida fue 2.7e-10, por encima de la tolerancia de 1e-10.

## 9. Criterios numéricos (siempre)
- Validar traza, hermiticidad **antes** de hermitizar y positividad de ρ, con tolerancias 1e-10 / 1e-10 / −1e-9. Integrador con atol 1e-12 y rtol 1e-10.
- Convergencia en N en al menos un punto por resultado. Con |α|² = 4 y filtro, **N ≥ 20** (N = 16 dio el artefacto ×2.9 de P10). N = 24 con filtro no cabe en 14 GB.
- Modos espurios: peso de borde > 0.5 o autovalor inestable al 0.3% en tres N. Graficar tasas **dinámicas**, no brechas espectrales.
- No razonar mecanismos físicos con datos no convergidos.
- Marco de laboratorio con muestreo t = nT_p (desplazamiento polarónico +g_z/ω).

## 10. Inventario de archivos (repositorio `msc/Mechanical-Cat-State/`)
- `figuras_finales/figuras/`:
  - principales: `figura_central` y `figura_central_sinpiso` (dos versiones para revisión), `principal_fig2` (universalidad de κ₁/κ₂), `figura_termica` (borrador, sin validar);
  - apéndice: `apendice_resonancia_universal`, `apendice_fig2_estados`, `fig2`, `fig3` y `fig4` (validación), `p9_diagnostico`.
- `figuras_finales/codigo/`: scripts de cálculo (`calc_*`, `run_*`) y de figura. `figuras_finales/data/`: cachés y CSV.
- `verificacion_independiente/`: V1–V6 y V4b (reportes .md, código y datos).
- `validacion/` (QuTiP 4, Tareas 1–44, Naseem) y `validacion_ma/` (QuTiP 5, Tareas 39–47, Ma).
- PDFs: `msc/ma2019.pdf` (no subido al remoto por derechos) y `msc/2508.10500v2.pdf` (Naseem).
- **Fuera de este repositorio (en el entorno del chat):** `documento_completo.tex` v1, `brecha_punto_excepcional.tex`, `lamb_shift_ma2019.tex`, `figura_merito.tex`, `mapa_plataformas.tex` y las .bib (`refs_doc.bib`, etc.).

## 11. Superado y retirado (no usar)
- "hf_q/kT ≳ 9 para sesgo 100" (régimen Naseem, frecuencia mecánica) → usar §8.
- "Techo κ/4 del confinamiento" → ≈ 0.28κ con |α|² = 4 (κ/4 solo vale para α = 0).
- "γ_bf ≈ 0.05 n_qκ" como constante → c_bf(κ₂/κ).
- "≤ 0.4%" de precisión en todo el rango → función de κ₂/κ (§4.5).
- Brecha con máximo en Γ₂/κ ≈ 0.13 y rama con |Im λ| ∝ N: artefactos de borde de Fock.
- "5.º modo espectral" como tasa de confinamiento (Tarea 43): espurio. Con la tasa dinámica, el filtro cuesta ≤ 5% para κ_f ≥ 0.3.
- Exceso ×2.9 de P10: artefacto de N = 16.
- Código adaptado con α_eff para P_c: no sirve (daba P_c = 0.967 sin gato).
- Acuerdo del 0.05% de V3: era una compensación de errores (con la fórmula corregida: 0.982).

## 12. Pendientes
1. **Validación térmica con el modelo completo** (§8): diagnóstico en curso. Luego, segundo punto κ₂/κ = 0.05 y T = 0.
2. **Tabla de plataformas** para el texto: κ/ω, g_z/κ, κ₂/κ, χ|α|²/κ, κ₁/κ₂, κ₁/κ₂^eff, x a 10 mK y veredicto. Liu queda fuera del régimen.
3. Elegir entre `figura_central` y `figura_central_sinpiso`. Se recomienda la versión sin piso.
4. Fig. 1 (esquema en TikZ, la hace Jhon).
5. `documento_completo` v2: reescribir la sección térmica, el techo, γ_bf y la precisión de C3; añadir η, la isla, Liu 47b, FWHM = cG y nocounter.
6. Revisión de citas y lectura de Hou 2024.
7. P4: déficit no adiabático sin explicación analítica.
8. Térmico de Naseem con κ₂/κ = 2.07: no lanzar hasta decidirlo.
9. Revisión humana independiente; autoría con Gómez; política de la APS sobre IA.

## 13. Lo que falta que pase Jhon para organizar el escrito
Ver la respuesta del chat que acompaña a este archivo.

## 14. Convenciones
- LaTeX y Python: ver §1.
- Cifras: reportar siempre la convergencia en N.
- Figuras: variables adimensionales; las plataformas van en una tabla, no en las figuras principales.

## 15. Riesgos de credibilidad
- Autor único con verificación asistida por IA: pedir revisión humana y revisar la política de la APS.
- Las conclusiones sobre Liu son numéricas: contrastarlas.
- La figura térmica depende del modelo efectivo, y su validación con el modelo completo está abierta (§8).
- No sobregeneralizar el mapa fuera de la isla verificada.
