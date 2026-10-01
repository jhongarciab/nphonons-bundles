# HANDOFF — paper de generalización: qubits gato estabilizados por un qubit auxiliar con acoplamientos (g_x, g_z)

Documento de traspaso para organizar el escrito (PRA, autor único). Consolidado el 2026-09-29 a partir del HANDOFF del chat y del estado real del repositorio. Actualizado el 2026-10-01 con los resultados de las simulaciones térmicas (§5, §8, §9 a §12, §15, §16 y §17); las derivaciones analíticas (§4) no se tocaron.
Idioma: español. **[VERIFICAR]** marca lo que no se ha podido comprobar contra archivos. La sección 11 (superado) prevalece sobre cualquier documento viejo.

Fuentes vivas en este repositorio (mapa de carpetas en `msc/README.md`):
- `Mechanical-Cat-State/figuras_finales/README.md`: métodos, parámetros, cifras y validaciones de cada figura.
- `Mechanical-Cat-State/figuras_finales/PENDIENTES_Y_HALLAZGOS.md`: pendientes, discrepancias y correcciones adoptadas (P1–P10, E1–E8).
- `Mechanical-Cat-State/validacion/independiente/V1..V6_reporte.md`, `V4b_reporte.md`: verificación independiente de R1–R6.
- `Mechanical-Cat-State/figuras_finales/C6_polaron.md`: base polarónica.

---

## 0. Índice
1. Autor y preferencias · 2. Idea y mensaje · 3. Modelo y notación · 4. Derivaciones · 5. Resultados consolidados ·
6. Errores de la literatura · 7. Plataformas · 8. Térmico · 9. Criterios numéricos · 10. Inventario de archivos ·
11. Superado y retirado · 12. Pendientes · 13. Lo que falta para escribir · 14. Convenciones · 15. Riesgos ·
16. Verificaciones térmicas · 17. Historial de tareas de Opus

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

## 4. Derivaciones (paso a paso en `validacion/independiente/V1_derivacion.md`)

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
| Térmico: efectivo con filtro frente al completo | γ_pf 0.956–0.959; γ_bf 0.956 (κ₂/κ = 0.05) a 0.776 (0.4) [MEDIDO], §8.4 | `data/termico_validacion.csv`, `data/handoff_cifras.txt` |
| Térmico: x* del completo | [ESTIMADO], efectivo conservador 0 a 0.2, §8.5 | `data/termico_x_estrella_piso.csv` |

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

## 8. Térmico (P7). Detalle en `figuras_finales/README.md` y `PENDIENTES_Y_HALLAZGOS.md` (P7)
Estado a 2026-10-01. Cada afirmación lleva una marca: **[MEDIDO]** (sale de una corrida, con archivo), **[ESTIMADO]** (derivado de medidas con un supuesto que se indica), **[HIPÓTESIS]**, **[NO VALIDADO]** o **[VERIFICAR]**. Las cifras nuevas están en `figuras_finales/data/handoff_cifras.txt` (cada una con su archivo fuente) y en los archivos citados. Tasas en unidades de κ salvo que se diga lo contrario. El modelo completo se corre en unidades de Ma (ω = 6, κ = 0.03, g_z = 0.42, κ_f = 0.3, γ = 6e-7) y sus tasas se dividen entre κ.

### 8.1 Definiciones y modelo efectivo
- **Definiciones:** η = γ_pf/γ_bf (Tarea 37); x = hf_q/k_BT con f_q la frecuencia del **qubit**; n_q = 1/(eˣ−1), n_m = 1/(e^{x/2}−1); k_BT/h = 208.37 MHz a 10 mK.
- **Modelo efectivo** (qubit y filtro explícitos, `calc_termico.py`) en la isla: κ₂/κ = 0.25, g_z/κ = 14, |α|² = 4, ω/κ = 200.
- **Modelo completo** (`calc_filtro_completo.py`): H(t) en el laboratorio sin RWA, filtro explícito (N_f = 2), propagador de un período y diagonalización densa. Variante `--variante=plano` (sin filtro, N_f = 1).
- **Baños:** todos son ruido blanco de Lindblad. El baño del qubit o del filtro y los canales de un fonón Γ₁± usan n_q; la pérdida γ usa n_m. Ver §8.12 para el supuesto.

### 8.2 Resultados del efectivo (isla) [MEDIDO con el efectivo; ver en §8.3 qué está validado]
| baño | γ/κ | x*(η=100) | x*(η=220) |
|---|---|---|---|
| filtrado | 2e-5 | 10.11 | 10.91 |
| filtrado | 2e-4 | 7.78 | 8.59 |
| plano | 2e-5 | 8.19 | 9.01 |
| plano | 2e-4 | 7.18 | 8.00 |

- **Convergencia en N del efectivo:** N = 22 → 24 cambia x* ≤ 0.008 [MEDIDO].
- **Filtro y γ_bf:** el filtro endurece el requisito porque reduce γ_pf. El efectivo da γ_bf(filtro)/γ_bf(plano) = 1.277 en x = 6.86; el completo da 1.167 (ver §11, "+28%").
- **Bit-flip térmico:** γ_bf = c_bf n_qκ, con c_bf ∝ (G/κ)^s: s = 3.96 ± 0.04 con filtro y 3.69 ± 0.05 en el plano. No es 0.05 constante. [MEDIDO con el efectivo; el completo no se ha usado para s, y como el cociente γ_bf completo/efectivo depende de κ₂/κ (§8.4), el exponente del completo puede diferir: **[VERIFICAR]**.]
- **Supresión con |α|²:** d ln γ_bf/d|α|² ≈ −0.55 a −0.64 en el plano (coincide con Gautier/Tarea 44 para g₂/κ ≈ 0.3) y ≈ −0.42 con filtro [MEDIDO con el efectivo].
- η = 100 y 220 son **valores de referencia**: 220 es un umbral sobre κ₂/κ₁, no sobre η.
- **Resuelto el 2026-09-30:** el bloque "validación con el modelo completo abierta" de la versión anterior (modo de pozo no convergido, λ ≈ −1 con tasa 4e-2 a 7e-2) era un artefacto de correr el completo en unidades κ = 1 (ω = 200): el modo de pozo no converge ahí. En unidades de Ma (ω = 6, κ = 0.03, N = 22) el modo de pozo converge (§8.4). **[VERIFICAR]** la causa exacta de la falta de convergencia en unidades κ = 1: solo está documentado que no converge (README, sección de validación térmica).

### 8.3 Estado de validación
| afirmación | estado | archivo fuente |
|---|---|---|
| γ_pf del efectivo con filtro reproduce el completo: cociente completo/efectivo 0.9562 a 0.9588 en los ocho puntos | [MEDIDO] (el sesgo de ~4% lo explica en parte α_eff² ≈ 3.80 frente a |α|² = 4 del efectivo, **[HIPÓTESIS]**) | `data/handoff_cifras.txt`, `termico_validacion.csv` |
| γ_bf del efectivo con filtro dentro de ±5% del completo | [MEDIDO] solo en κ₂/κ = 0.05; en 0.1, 0.25 y 0.4 el efectivo lo sobreestima | `termico_validacion.csv`, `termico_validacion_fase3.txt` |
| el cociente γ_bf completo/efectivo con piso restado no depende de x (criterio 3%) | [MEDIDO] en 4 valores de κ₂/κ, entre x = 6.86 y x ≈ 9 | `termico_x_independencia.csv`, `.txt` |
| x* del completo (tabla §8.5) | [ESTIMADO]: supone cociente γ_bf con piso restado y cociente γ_pf constantes en x, y piso de T = 0 fijo | `termico_x_estrella_piso.csv` |
| el efectivo es conservador: x* real entre 0 y 0.2 menor | [ESTIMADO] (misma base) | idem |
| piso de T = 0 por κ₂/κ | [MEDIDO]; origen **[VERIFICAR]** | `handoff_cifras.txt`, logs en `data/filtro_completo/` |
| convergencia en N con filtro | **[VERIFICAR]** N = 20 en curso; N = 24 no cabe en memoria | `log_termico_ma5.txt` |
| baño plano, cualquier cantidad | **[NO VALIDADO]** | `data/filtro_completo/*plano*` |
| factor 1.52 de γ_pf en el plano es el canal n_q del qubit | [MEDIDO] (apagado por canales en el completo) | `*_nqoff.npz`, `*_nmoff.npz` |
| el desplazamiento dependiente del estado explica el canal n_q del plano | **[HIPÓTESIS]** (reproduce el 84% con 16% de exceso sin explicar) | `data/plano_kick.txt` |
| el desplazamiento aplicado al acoplamiento qubit-filtro explica γ_bf | **[HIPÓTESIS]** descartada solo en su parte secular | `data/filtro_kick.txt` |
| el modo vecino de γ_pf es el tercer modo lógico | [MEDIDO] en el plano; con filtro **[VERIFICAR]** | `log_plano_modos.txt` |
| positividad y hermiticidad | [MEDIDO]; dos violaciones de positividad con el criterio anterior; origen del residuo **[VERIFICAR]** | `log_hermiticidad_positividad.txt` |
| validez fuera del juego de parámetros usado | **[NO VALIDADO]** (§8.12) | |

### 8.4 Validación del efectivo contra el completo, con filtro [MEDIDO]
Un solo juego de parámetros (§8.12), N = 22, N_f = 2. Los x ≈ 9 son 9 para κ₂/κ = 0.05 y 0.1, y 9.5 para 0.4 (a x = 9 el piso esperado en 0.4 quedaba bajo 1%); la isla (0.25) está a x = 10.1 (fase 2).

Tasas (κ = 1):
| κ₂/κ | x | γ_pf completo | γ_pf efectivo | γ_bf completo | γ_bf efectivo | η completo | η efectivo |
|---|---|---|---|---|---|---|---|
| 0.05 | 6.86 | 1.650e-04 | 1.721e-04 | 1.647e-06 | 1.722e-06 | 100.2 | 99.95 |
| 0.05 | 9 | 1.570e-04 | 1.641e-04 | 1.953e-07 | 2.038e-07 | 803.8 | 805.2 |
| 0.1 | 6.86 | 1.651e-04 | 1.722e-04 | 6.420e-06 | 7.497e-06 | 25.72 | 22.97 |
| 0.1 | 9 | 1.571e-04 | 1.642e-04 | 7.550e-07 | 8.780e-07 | 208 | 187 |
| 0.25 | 6.86 | 1.653e-04 | 1.724e-04 | 3.349e-05 | 4.267e-05 | 4.935 | 4.039 |
| 0.25 | 10.1 | 1.556e-04 | 1.627e-04 | 1.316e-06 | 1.650e-06 | 118.2 | 98.63 |
| 0.4 | 6.86 | 1.654e-04 | 1.726e-04 | 6.908e-05 | 8.903e-05 | 2.395 | 1.938 |
| 0.4 | 9.5 | 1.566e-04 | 1.638e-04 | 4.850e-06 | 6.219e-06 | 32.29 | 26.33 |

Cocientes completo/efectivo:
| κ₂/κ | x | γ_pf c/e | γ_bf c/e crudo | γ_bf c/e con piso restado | η c/e | piso completo / γ_bf térmico | hermiticidad cruda | mín. autovalor |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 6.86 | 0.9588 | 0.9561 | 0.956 | 1.003 | 0.30% | 1.780e-09 | -1.5e-10 |
| 0.05 | 9 | 0.9566 | 0.9583 | 0.9574 | 0.9982 | 2.52% | 1.319e-09 | +4.8e-17 |
| 0.1 | 6.86 | 0.9588 | 0.8563 | 0.8558 | 1.12 | 0.17% | 2.117e-09 | +7.3e-15 |
| 0.1 | 9 | 0.9565 | 0.8599 | 0.856 | 1.112 | 1.44% | 2.797e-09 | -1.0e-16 |
| 0.25 | 6.86 | 0.9587 | 0.7848 | 0.7842 | 1.222 | 0.09% | 4.102e-10 | +1.7e-14 |
| 0.25 | 10.1 | 0.9562 | 0.7976 | 0.7817 | 1.199 | 2.35% | 1.155e-09 | -2.7e-16 |
| 0.4 | 6.86 | 0.9587 | 0.7759 | 0.7752 | 1.236 | 0.09% | 1.645e-12 | +2.3e-14 |
| 0.4 | 9.5 | 0.9562 | 0.7798 | 0.7697 | 1.226 | 1.35% | 1.983e-09 | -2.5e-16 |

- **Criterio ±5%:** γ_pf cumple en todos los puntos. γ_bf cumple solo en κ₂/κ = 0.05 (0.956 y 0.9583); en 0.1, 0.25 y 0.4 el cociente es 0.8563, 0.7848 y 0.7759 a x = 6.86.
- **Independencia de x (criterio fijado antes: cambio < 3% del cociente γ_bf con piso restado entre x = 6.86 y x ≈ 9):** cambios de 0.14%, 0.02%, -0.72% y -0.33% (κ₂/κ = 0.05, 0.1, 0.4 y 0.25). **Sostenida.** Archivo: `data/termico_x_independencia.csv`.
- **Dependencia de κ₂/κ:** el cociente γ_bf crudo baja de 0.9561 a 0.7759 entre κ₂/κ = 0.05 y 0.4 y se aplana entre 0.25 y 0.4. En κ₂/κ = 0.4 (χ|α|²/κ = 0.27, junto a la frontera 0.3) no se ve un desplome propio de la frontera. No se ajusta ninguna ley ni se aplica corrección. El origen de la discrepancia es **[VERIFICAR]** (abierto).
- **Tolerancia del integrador (fase 2, isla):** atol 1e-14 y rtol 1e-12 en lugar de 1e-12 y 1e-10 deja γ_pf igual en 7 cifras y mueve γ_bf −2.6e-5 relativo (x = 10.1) y el piso −0.10% [MEDIDO, `termico_validacion_fase2_tolE.txt`].

### 8.5 x* del completo frente al efectivo [ESTIMADO]
Modelo: γ_pf,c(x) = q_pf · γ_pf,e(x); γ_bf,c(x) = piso_c + r_s · (γ_bf,e(x) − piso_e), con q_pf el cociente crudo de γ_pf y r_s el cociente de γ_bf con piso restado, medidos en el x más alto de cada κ₂/κ; piso_c del completo a T = 0 y piso_e del efectivo a x = 60. "antes" es la estimación η_c(x) = q η_e(x) con q medido en ese x, que suponía el piso proporcional. La corrección por el crecimiento relativo del piso solo importa en κ₂/κ = 0.4 (+0.028 en x*(100), +0.081 en x*(220)); en los demás cambia ≤ 0.017.
| κ₂/κ | rango del completo medido en x | x*(100) efectivo | x*(100) completo | posición | x*(220) efectivo | x*(220) completo | posición |
|---|---|---|---|---|---|---|---|
| 0.05 | 6.86 a 9 | 6.86 | 6.861 (antes 6.862) | interpolado | 7.666 | 7.667 (antes 7.668) | interpolado |
| 0.1 | 6.86 a 9 | 8.361 | 8.25 (antes 8.253) | interpolado | 9.166 | 9.057 (antes 9.057) | extrapolado (+0.06 sobre el rango) |
| 0.25 | 6.86 a 10.1 | 10.11 | 9.927 (antes 9.931) | interpolado | 10.91 | 10.75 (antes 10.73) | extrapolado (+0.65 sobre el rango) |
| 0.4 | 6.86 a 9.5 | 10.84 | 10.67 (antes 10.64) | extrapolado (+1.17 sobre el rango) | 11.64 | 11.52 (antes 11.43) | extrapolado (+2.02 sobre el rango) |

- **Enunciado correcto:** el efectivo es conservador, pide un x* mayor que el completo por 0 a 0.2 unidades de x (diferencias efectivo menos completo: -0.001, 0.111, 0.187, 0.177 en x*(100) y -0.001, 0.109, 0.166, 0.125 en x*(220), para κ₂/κ = 0.05, 0.1, 0.25 y 0.4). [ESTIMADO]
- Los x* extrapolados dependen del supuesto de cociente constante más allá del rango medido. En κ₂/κ = 0.25 hubo una comprobación: r_s pasa de 0.7842 (x = 6.86) a 0.7817 (x = 10.1).

### 8.6 Piso de T = 0 [MEDIDO]
Piso = γ_bf a T = 0 (sin ocupación térmica) con γ = 2e-5κ. Completo: corrida de T = 0 con filtro; efectivo: x = 60.
| κ₂/κ | piso completo (κ) | piso efectivo (κ) | cociente c/e | η(T=0) completo | η(T=0) efectivo |
|---|---|---|---|---|---|
| 0.05 | 4.931e-09 | 4.959e-09 | 0.994 | 3.1e+04 | 3.23e+04 |
| 0.1 | 1.090e-08 | 8.751e-09 | 1.25 | 1.4e+04 | 1.83e+04 |
| 0.25 | 3.086e-08 | 5.819e-09 | 5.3 | 4.97e+03 | 2.76e+04 |
| 0.4 | 6.528e-08 | 3.152e-09 | 20.7 | 2.35e+03 | 5.1e+04 |

- **Fracción del γ_bf térmico del completo** (piso completo / γ_bf térmico): a x = 6.86, 0.30%, 0.17%, 0.09%, 0.09%; a x ≈ 9, 2.52%, 1.44%, 2.35%, 1.35% (κ₂/κ = 0.05, 0.1, 0.25, 0.4). Crece con x porque el γ_bf térmico cae como e^(−x) y el piso es fijo.
- **η(T=0)** es el máximo de η a esa κ₂/κ: va de 3.1e+04 (κ₂/κ = 0.05) a 2.35e+03 (0.4) en el completo. Por tanto **"η_max ≈ 5000" no vale para todo el mapa**: solo es cercano al valor de κ₂/κ = 0.25 (4.97e+03). Sigue siendo mucho mayor que 100 y 220 en los cuatro casos.
- **El piso crece con κ₂/κ en el completo y no en el efectivo** (completo 4.931e-09 a 6.528e-08; efectivo 4.959e-09, 8.751e-09, 5.819e-09, 3.152e-09). A κ₂/κ = 0.05 coinciden al 0.6%.
- **Origen del piso: [VERIFICAR].** Comprobado: no depende de la tolerancia del integrador (−0.10%). No comprobado: el mecanismo ni su dependencia de κ₂/κ. **No hay barrido en N con filtro todavía** (N = 20 en curso, N = 24 no cabe); en el plano el piso cambia 1.9% entre N = 22 y 26 (1.060% a 1.040%), y N = 18 no converge (20.9%).
- **En el plano el efectivo con desplazamiento empeora el piso:** completo 3.042e-07, efectivo sin desplazamiento 2.114e-07, con desplazamiento 1.667e-07 (§8.7).

### 8.7 Baño plano [NO VALIDADO]
Completo sin filtro (`--variante=plano`), κ₂/κ = 0.25, x = 6.86, N = 22. Archivos `data/filtro_completo/*Nf1_plano*`.
- **Cocientes completo/efectivo plano:** γ_pf 1.522 (térmico) y 0.94 (T = 0); γ_bf 0.8584. El efectivo plano a T = 0 reproduce la fórmula γ_pf = 2[Γ₁⁻|α|² + Γ₁⁺(|α|²+1)] + 2γ|α|² con |α|² = 4 (8.863e-04 frente a 8.863e-04); con α_eff² = 3.798 del completo la fórmula da 8.425e-04 y el completo plano la reproduce al 1.1% (en esa comparación el efectivo sin α_eff² es 6% mayor).
- **El factor 1.52 de γ_pf es térmico, no de T = 0** [MEDIDO]: a T = 0 el cociente es 0.94.
- **Apagado por canales en el completo plano** (γ_pf, κ = 1): ambos encendidos 1.369e-03; n_q apagado 8.445e-04; n_m apagado 1.358e-03; ambos apagados (T = 0) 8.331e-04. Aumento térmico total 5.362e-04; **5.248e-04 (98%) es el canal del qubit (n_q)**, igual a 0.5·n_qκ con n_q = 1.050e-03; el canal n_m aporta 1.139e-05, que el efectivo reproduce (1.272e-06). El efectivo sin desplazamiento aporta por n_q solo 1.272e-06.
- **Efectivo plano con desplazamiento dependiente del estado [HIPÓTESIS]** (`calc_termico.punto(..., kick=True)`, β = 2g_z/ω = 0.14; canales D[σ₋a], D[σ₋a†] con tasa β²κ(n_q+1) y D[σ₊a], D[σ₊a†] con tasa β²κn_q): incremento por n_q 6.101e-04 frente a 5.248e-04 del completo (**exceso de 16%, sin explicar**); tasa por evento 0.581·n_qκ frente a 0.5·n_qκ del completo (sin desplazamiento: 0.00121). Efecto en las tasas térmicas (con desplazamiento / sin / completo): γ_pf 1.506e-03 / 8.996e-04 / 1.369e-03; γ_bf 3.192e-05 / 3.342e-05 / 2.869e-05.
- **Aviso:** η plano con el desplazamiento (47.17) queda a 1.2% de η completo (47.73), pero γ_pf queda 10% y γ_bf 11% sobre el completo. **Ese acuerdo de η es compensación de errores, no validación.** Tampoco se reproduce el piso (arriba).
- **Cualquier comparación plano/filtrado hecha con el efectivo plano sin el desplazamiento no es fiable** (§11).
- **Barrido en N del plano** (N = 18, 22, 26; térmico y T = 0): γ_pf térmico 1.377e-03, 1.369e-03, 1.369e-03; γ_bf térmico 3.621e-05, 2.869e-05, 2.868e-05; γ_pf T = 0 8.404e-04, 8.331e-04, 8.331e-04; γ_bf T = 0 7.567e-06, 3.042e-07, 2.983e-07. **N = 18 no converge** (γ_bf térmico 26% mayor y γ_bf a T = 0 unas 25 veces mayor que con N = 22). De N = 22 a 26: γ_pf cambia 4.4e-06 (térmico) y 7.2e-06 (T = 0); γ_bf térmico 0.02%; γ_bf T = 0 1.9%; el piso 1.9% relativo.
- **Filtro frente a plano, γ_bf térmico en κ₂/κ = 0.25, x = 6.86:** completo +17%; el efectivo da +28% (el efectivo lo exagera). Es un solo punto.

### 8.8 Desplazamiento aplicado al acoplamiento con filtro [HIPÓTESIS, descartada solo en su parte secular]
Diagnóstico en el efectivo con filtro (`filtro_kick.py`, `data/filtro_kick.txt`). Canales D[σ∓a], D[σ∓a†] con β = 2g_z/ω = 0.14 y el factor lorentziano del filtro en δ = ω (ke(ω) = κ_f²/(4ω²+κ_f²), 6.2e-4), porque los términos σ₊a†b y similares del acoplamiento qubit-filtro son no seculares (giran a ±ω) y su efecto a segundo orden es ese factor.
- **Resultado:** a x = 6.86, γ_bf cambia -4.6e-05 (κ₂/κ = 0.25) y -3.9e-05 (0.4) relativo; γ_pf +0.27%. A T → 0 (x = 60) γ_bf cambia +6.8e-04 y +1.0e-03. Es despreciable y no acerca γ_bf al completo (el cociente sigue en ~0.78).
- **Alcance:** no es una exclusión general. Incluir los términos no seculares exige un cálculo de Floquet con tiempo, que no se hizo. **No está resuelto.** Es solo un diagnóstico y no se aplica al mapa ni como corrección.

### 8.9 Modos lentos
- **Plano, tercer modo lógico [MEDIDO]** (`plano_modos.py`, `data/filtro_completo/log_plano_modos.txt`, N = 22, κ₂/κ = 0.25). Base lógica del marco polarónico: P_L (paridad), Y_L = i(|C+⟩⟨C−| − h.c.) y Z_L (bit-flip). A T = 0, el modo de paridad tiene λ = +0.999987, tasa 2.49940e-05 (Ma), traslape con P_L 1.409 y con Y_L 0.000; el modo vecino tiene λ = -0.999987, tasa 2.49915e-05, traslape con Y_L 1.409 y con P_L 0.000. Las tasas difieren 1.0e-04 relativo. A x = 6.86, paridad λ = +0.999978, tasa 4.10809e-05, y vecino λ = -0.999978, tasa 4.13247e-05 (difieren 0.6%). Conclusión: **el vecino de γ_pf en el plano es el tercer modo lógico (la otra coherencia entre |α⟩ y |−α⟩)**; el valor 1.41 es √2, la norma del operador lógico. El bit-flip (Z_L, λ = −1) es el modo de tasa 9.13e-09 (T = 0) y 8.61e-07 (x = 6.86) en unidades de Ma.
- **Con filtro: solo observado, sin comprobar** [VERIFICAR]. A x = 6.86 hay un segundo modo con λ < 0 y tasa cercana a γ_pf. Cociente de su tasa a la de γ_pf: 1.010, 1.038, 1.193, 1.384 y |⟨a⟩| de 0.014, 0.045, 0.192, 0.358 (κ₂/κ = 0.05, 0.1, 0.25, 0.4). Con filtro no está degenerado con el de paridad y su traslape con a crece con κ₂/κ, lo que sugiere mezcla con el bit-flip; no se comprobó que sea Y_L. El peso de borde de los modos elegidos (paridad y bit-flip) es 6.6e-07 como máximo (criterio: < 0.5).

### 8.10 Hermiticidad y positividad [MEDIDO]
Hermiticidad cruda = ‖ρ − ρ†‖_F con Tr ρ = 1, antes de hermitizar; positividad = mínimo autovalor de (ρ + ρ†)/2. **Umbrales vigentes (desde 2026-10-01):** plano 1e-10 en hermiticidad y −1e-9 en positividad; completo con filtro 1e-8 en ambos. Log crudo con marca por umbral: `data/filtro_completo/log_hermiticidad_positividad.txt` (`codigo/log_herm_posit.py`).
- **Conteos con los umbrales vigentes:** completo con filtro, 0 de 14 corridas superan 1e-8 en hermiticidad y 0 superan −1e-8 en positividad. Plano: 1 de 8 supera 1e-10 (N = 18 a T = 0) y 0 violan la positividad.
- **Conteos con los umbrales anteriores (1e-10 y −1e-9 también para el filtro):** 13 de 14 corridas con filtro superan 1e-10 en hermiticidad y 2 violan −1e-9 en positividad. Hasta la fase 3 se informó "10 de 11" (11 corridas con filtro); con las tres de la fase 6 son 13 de 14.
- **Las dos violaciones de positividad (criterio anterior −1e-9):** κ₂/κ = 0.050 a T = 0 (N = 22, N_f = 2): mínimo autovalor -2.49e-09; κ₂/κ = 0.100 a T = 0 (N = 22, N_f = 2): -1.42e-09. Ambas con filtro; con el umbral vigente de −1e-8 no se marcan.
- **Justificación de que no afectan las tasas:** las tasas salen de autovalores del propagador y son estables (cambiar la tolerancia del integrador las movió 2.6e-5 relativo), y a κ₂/κ = 0.05 el piso del completo coincide con el del efectivo al 0.6% aunque es unas 10 veces menor que el residuo de hermiticidad.
- **El origen del residuo (ruido de la diagonalización densa) no está verificado [VERIFICAR].** La hermiticidad cruda del completo con filtro va de 1.6e-12 a 2.8e-09 entre puntos y no depende de la tolerancia del integrador.

### 8.11 Recursos y tiempos reales
- **Memoria máxima del completo con filtro, N = 22, N_f = 2: 14.0 GB** (toda la RAM del equipo), medida con `/usr/bin/time` en las tres corridas de la fase 6 (14015972, 14044332 y 14017232 kB). Las corridas anteriores no se midieron. **N = 24 no cabe** (~20 GB si escala como D⁴ con la dimensión de Liouville, **[ESTIMADO]**); no se intentó.
- **Tiempos reales con filtro** (N = 22, 6 hilos): fase 6, 5553.24 s, 5460.22 s y 5608.04 s (`/usr/bin/time`, `log_termico_ma4.txt`, κ₂/κ = 0.05 y 0.1 a x = 9, 0.4 a x = 9.5). Fases 2 y 3 (`time`, logs `log_termico_ma2.txt`, `log_termico_ma3.txt`): 93 min 45 s (0.25, x = 6.86), 135 min 48 s (0.25, x = 10.1, tolerancia estricta), 134 min 0 s (0.25, T = 0, tolerancia estricta), 89 min 46 s (0.1, T = 0), 93 min 11 s (0.4, x = 6.86), 92 min 15 s (0.4, T = 0). La corrida (0.1, x = 6.86) se hizo en paralelo con otra y tardó 7456 s solo en el propagador; su tiempo total no quedó en el log **[VERIFICAR]**. Las corridas de κ₂/κ = 0.05 (x = 6.86 y T = 0) y 0.25 (T = 0) son de la fase 1 y su tiempo total no está en los logs actuales (propagador 5262 s, 5188 s y 5245 s) **[VERIFICAR]**.
- **Dos corridas con filtro en paralelo agotaron la memoria** (13 de 14 GB) y el sistema cortó un vigilante; se pasó a corridas secuenciales.
- **Plano** (`log_plano_control.txt`; el tiempo total se asigna por el tiempo del propagador): N = 18, 4 min 5 s (térmico) y 3 min 57 s (T = 0); N = 22, n_q apagado 11 min 36 s, n_m apagado 11 min 27 s, T = 0 13 min 46 s (`time`); N = 26, 24 min 39 s (térmico) y 20 min 27 s (T = 0). El plano térmico N = 22 (propagador 416 s) se calculó de cero, pero su tiempo total no se registró **[VERIFICAR]**: la cifra de 7.6 s que se dio antes era de una lectura de caché y se retira (§11). `plano_modos.py`: unos 8 min con dos corridas en paralelo (marcas de archivo, 16:05:53 a 16:13:52 del 2026-09-30).
- **Efectivo:** segundos por punto (`plano_kick.py`, `filtro_kick.py`: ≤ 2 s por corrida, en `data/plano_kick.txt` y `data/filtro_kick.txt`).
- **N = 20 con filtro (κ₂/κ = 0.25, x = 6.86): en curso desde 2026-10-01 08:26, sin resultado [VERIFICAR].** Criterio fijado antes: si γ_pf y γ_bf cambian menos de 3% entre N = 20 y 22, N = 22 se da por convergido (apoyado en el plano, donde N = 18 falla y N = 22 a 26 cambia menos de 0.1% en los térmicos). Si cambian más, no se interpreta y se pasa al cálculo solo de los modos lentos, sin implementarlo hasta estimar su costo y su verificación (convergencia en N = 24 y 26 y un cálculo independiente de los autovalores frente a la diagonalización densa).

### 8.12 Parámetros cubiertos y supuesto del baño
- **Un solo juego de parámetros.** Toda la validación con el modelo completo usa g_z/κ = 14, ω/κ = 200 (ω = 6, κ = 0.03), |α|² = 4, γ/κ = 2e-5, κ_f/ω = 0.05, unidades de Ma y N = 22 (N_f = 2). **No cubierto:** γ/κ = 2e-4 (las curvas de γ/κ = 2e-4 de §8.2 no tienen punto del completo), otros κ_f, otros g_z/κ, y Naseem (x ≈ 0.96, fuera del rango validado de x ≥ 6.86). **[NO VALIDADO]**
- **Supuesto del baño.** Todos los baños son ruido blanco de Lindblad con n_q en el baño del qubit o del filtro y en los canales Γ₁±, y n_m en la pérdida γ. Un baño térmico físico daría n(ω) y n(3ω) en las transiciones de banda lateral. **Comprobado solo en el efectivo** (`calc_termico.punto(..., ocup='real')`, caches `*_ocreal.npz`): con filtro, γ_pf cambia +2.5e-04 (x = 6), +9.8e-05 (8), +3.7e-05 (10) y +1.4e-05 (12) relativo; en el plano cambia +7.8% (x = 6), +2.9% (8), +1.1% (10) y +0.4% (12). El modelo completo usa un único baño de Lindblad y no se ha probado con n(ω) y n(3ω). En el plano esa diferencia no es despreciable y el plano está **[NO VALIDADO]**.

## 9. Criterios numéricos (siempre)
- Validar traza, hermiticidad **antes** de hermitizar y positividad de ρ, con tolerancias 1e-10 / 1e-10 / −1e-9 (plano y modelo efectivo). **Completo con filtro (desde 2026-10-01): 1e-8 en hermiticidad y −1e-8 en positividad**; los valores crudos se registran siempre y se marca el umbral superado (§8.10). Integrador con atol 1e-12 y rtol 1e-10.
- Convergencia en N en al menos un punto por resultado. Con |α|² = 4 y filtro, **N ≥ 20** (N = 16 dio el artefacto ×2.9 de P10). N = 24 con filtro no cabe en 14 GB (N = 22 ya usa 14.0 GB, §8.11).
- Modos espurios: peso de borde > 0.5 o autovalor inestable al 0.3% en tres N. Graficar tasas **dinámicas**, no brechas espectrales.
- No razonar mecanismos físicos con datos no convergidos.
- Marco de laboratorio con muestreo t = nT_p (desplazamiento polarónico +g_z/ω).

## 10. Inventario de archivos
Ver `msc/README.md` (mapa completo tras la reorganización del 2026-09-29).
- Papers en `msc/papers/`. Escritos LaTeX del chat en `msc/docs/<nombre>/`: faltan `refs.bib` (brecha) y `refs_mapa.bib` (mapa), y **Hou 2024 no está** (`2407.17299` es Dubovitskii).
- Código y figuras oficiales en `Mechanical-Cat-State/figuras_finales/` (`figuras/`, `codigo/`, `data/`). Entorno: `Mechanical-Cat-State/.venv_qutip5`.
- Validación en `Mechanical-Cat-State/validacion/`: `independiente/` (V1–V6), `tareas_ma/` (Tareas 39–47) y `tareas_naseem/` (Tareas 1–44).
- Código del repositorio original del artículo en `Mechanical-Cat-State/codigo_original/`.
- **Archivos térmicos nuevos** (todos en `Mechanical-Cat-State/figuras_finales/`). Datos: `data/termico_validacion.csv` y `termico_validacion_fase3.txt` (cocientes completo/efectivo), `termico_validacion_fase2.txt`/`.csv`/`_tolE.txt` (fase 2), `termico_tabla_k2.csv`/`.txt` (x* por κ₂/κ, versión anterior), `termico_x_independencia.csv`/`.txt` (independencia de x), `termico_x_estrella_piso.csv`/`.txt` (x* con el piso medido), `plano_kick.txt`, `plano_efectivo_controles.txt`, `filtro_kick.txt` (hipótesis del desplazamiento), `handoff_cifras.txt` (cifras de §8 con su fuente), `filtro_completo/` (`.npz` del completo y logs `log_termico_ma2.txt`, `log_termico_ma3.txt`, `log_termico_ma4.txt`, `log_termico_ma5.txt`, `log_plano_control.txt`, `log_plano_modos.txt`, `log_hermiticidad_positividad.txt`). Scripts en `codigo/`: `calc_filtro_completo.py` (variante `plano`, controles `NQ_OFF`/`NM_OFF`, `TOL_ESTRICTA`), `calc_termico.py` (opción `kick`), `verif_termica.py`, `x_independencia.py`, `x_estrella_piso.py`, `p7_tabla.py`, `plano_modos.py`, `plano_kick.py`, `plano_analisis_efectivo.py`, `filtro_kick.py`, `log_herm_posit.py`, `handoff_cifras.py`, `run_validacion_termica2.sh` a `run_validacion_termica4.sh`, `run_plano_control.sh`, `run_plano_modos.sh`, `run_convergencia_N20.sh`.

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
- **"γ_bf/n_q solo sube un 28% con el filtro"** (§8 anterior, README): retirado el 2026-10-01. El +28% sale del efectivo; en el completo es +17% (un punto, κ₂/κ = 0.25 y x = 6.86) y el efectivo lo exagera (§8.7).
- **"η_max ≈ 5000" en todo el mapa**: retirado el 2026-10-01. η(T=0) del completo va de 3.1e4 (κ₂/κ = 0.05) a 2.35e3 (0.4) (§8.6). Está en PENDIENTES (P7) y en la conversación de la fase 3.
- **"N = 24 con filtro: ~13 GB solo en el propagador"** (README): retirado el 2026-10-01. N = 22 ya usa 14.0 GB de máximo medido; N = 24 no cabe (§8.11).
- **Tiempo de 7.6 s de la corrida plana térmica**: retirado el 2026-10-01. Era una lectura de caché; la corrida original tuvo un propagador de 417 s (§8.11).
- **"7 de 10 corridas sobre 1e-10 en hermiticidad"**: retirado el 2026-10-01. Eran 10 de 11 y, con las corridas de la fase 6, 13 de 14; con el umbral vigente de 1e-8 ninguna lo supera (§8.10).
- **"Positividad ≥ −4.4e-10 en las corridas completas"** (README): retirado el 2026-10-01. Dos corridas con filtro dan −2.49e-9 y −1.42e-9 (§8.10).
- **Comparaciones plano/filtrado hechas con el efectivo plano sin el desplazamiento dependiente del estado**: no fiables desde el 2026-09-30 (el efectivo plano omite el canal n_q que explica el 98% del aumento térmico de γ_pf del completo plano, §8.7).
- **"El efectivo con filtro sobreestima γ_bf en la isla por un efecto de x"**: retirado el 2026-09-30. La discrepancia depende de κ₂/κ y no de x (§8.4).

## 12. Pendientes
**Térmico, abierto:**
1. **Convergencia en N con filtro:** N = 20 en (κ₂/κ = 0.25, x = 6.86) en curso (§8.11). Si γ_pf o γ_bf cambian más de 3% frente a N = 22, pasar al cálculo solo de los modos lentos (estimar costo y verificación antes de implementarlo: convergencia en N = 24 y 26 y cálculo independiente de autovalores frente a la diagonalización densa).
2. **Validación de κ_f y de γ/κ = 2e-4** con el modelo completo (hoy solo κ_f/ω = 0.05 y γ/κ = 2e-5).
3. **Naseem** (x ≈ 0.96, fuera del rango validado): no lanzar hasta decidirlo; térmico con κ₂/κ = 2.07 no calculado.
4. **Exceso del 16% del canal n_q en el efectivo plano con desplazamiento** (§8.7): sin explicación.
5. **Origen de la discrepancia de γ_bf con κ₂/κ** (0.956 a 0.776, §8.4).
6. **Origen del piso de T = 0 con filtro** (crece con κ₂/κ en el completo y no en el efectivo, §8.6).
7. **Validación del baño plano** y del supuesto de ruido blanco (n(ω) y n(3ω)) en el completo (§8.12).
8. **Desplazamiento con filtro, parte no secular:** requiere un cálculo de Floquet con tiempo (§8.8).
9. **Modo vecino con filtro** (§8.9) y **origen del residuo de hermiticidad** (§8.10).
10. **Exponente s de c_bf en el completo** (§8.2).

**Escrito y figuras:**
11. **Tabla de plataformas** para el texto: κ/ω, g_z/κ, κ₂/κ, χ|α|²/κ, κ₁/κ₂, κ₁/κ₂^eff, x a 10 mK y veredicto. Liu queda fuera del régimen.
12. Elegir entre `figura_central` y `figura_central_sinpiso` (se recomienda la versión sin piso). Estilo de `figura_termica` (la nota se solapa con la leyenda): lo ajusta Jhon.
13. Fig. 1 (esquema en TikZ, la hace Jhon).
14. `documento_completo` v2: reescribir la sección térmica, el techo, γ_bf y la precisión de C3; añadir η, la isla, Liu 47b, FWHM = cG y nocounter.
15. Revisión de citas y lectura de Hou 2024. P4: déficit no adiabático sin explicación analítica.
16. Revisión humana independiente; autoría con Gómez; política de la APS sobre IA.

**Resuelto:** validación térmica con el modelo completo en unidades de Ma (2026-09-30, §8.4); segundo punto κ₂/κ = 0.05 y T = 0 (2026-09-29); separación de κ₂/κ y x (2026-09-30).

## 13. Lo que falta que pase Jhon para organizar el escrito
Ver la respuesta del chat que acompaña a este archivo.

## 14. Convenciones
- LaTeX y Python: ver §1.
- Cifras: reportar siempre la convergencia en N.
- Figuras: variables adimensionales; las plataformas van en una tabla, no en las figuras principales.

## 15. Riesgos de credibilidad
- Autor único con verificación asistida por IA: pedir revisión humana y revisar la política de la APS.
- Las conclusiones sobre Liu son numéricas: contrastarlas.
- La figura térmica depende del modelo efectivo. Validado con el completo en un solo juego de parámetros y solo con filtro; γ_bf del efectivo falla el ±5% salvo en κ₂/κ = 0.05 (§8.4).
- **Baño plano sin validar:** el efectivo plano omite el canal n_q del qubit (98% del aumento térmico de γ_pf en el completo plano). Las curvas planas de la figura térmica están marcadas "not validated".
- **El acuerdo de η del efectivo plano con desplazamiento es compensación de errores** (γ_pf +10%, γ_bf +11%), no validación (§8.7).
- **Dos violaciones de positividad** con el criterio −1e-9 (−2.49e-9 y −1.42e-9) y hermiticidad cruda sobre 1e-10 en 13 de 14 corridas con filtro; el origen del residuo no está verificado. El umbral del filtro se subió a 1e-8 el 2026-10-01 con la justificación de §8.10.
- **Un solo juego de parámetros** (g_z/κ = 14, ω/κ = 200, |α|² = 4, γ/κ = 2e-5, κ_f/ω = 0.05); nada validado en γ/κ = 2e-4, otros κ_f, otros g_z/κ ni Naseem (§8.12).
- **Supuesto de ruido blanco:** n_q en los canales del qubit y Γ₁±, n_m en γ; solo probado con n(ω) y n(3ω) en el efectivo (§8.12).
- **x\* del completo es estimado** y varios valores se extrapolan más allá del rango medido (§8.5).
- No hay barrido en N con filtro todavía (§8.11).
- No sobregeneralizar el mapa fuera de la isla verificada.

## 16. Verificaciones térmicas (fases 1 a 6)
| prueba | criterio de aceptación | resultado | dónde |
|---|---|---|---|
| efectivo con filtro frente al completo, γ_pf y γ_bf | ±5% en cada tasa | γ_pf cumple en los 8 puntos; γ_bf cumple solo en κ₂/κ = 0.05 | §8.4 |
| independencia de x del cociente γ_bf con piso restado | cambio < 3% entre x = 6.86 y x ≈ 9 (fijado antes) | cumple: 0.02% a 0.72% en 4 valores de κ₂/κ | §8.4 |
| tolerancia del integrador (isla) | reportar el cambio | γ_pf igual en 7 cifras; γ_bf −2.6e-5; piso −0.10% | §8.4 |
| piso γ_bf(T=0) frente al γ_bf térmico | < 1% (criterio original) | cumple a x = 6.86 (0.09% a 0.30%); no a x ≈ 9 (1.35% a 2.52%) ni en la isla a x = 10.1 (2.35%). El criterio depende de x por construcción; se propuso reportarlo y no usarlo como aceptación, **decisión de Jhon pendiente** | §8.6 |
| barrido en N, plano | convergencia | N = 18 no converge; N = 22 a 26: γ_pf y γ_bf térmicos < 0.1%, piso 1.9% relativo | §8.7 |
| apagado de n_q y n_m en el completo plano | atribuir el aumento térmico de γ_pf | 98% es el canal n_q (0.5·n_qκ), 2% es n_m | §8.7 |
| efectivo plano con desplazamiento | reproducir el canal n_q del completo | reproduce con 16% de exceso; η coincide por compensación de errores | §8.7 |
| desplazamiento en el acoplamiento qubit-filtro (efectivo con filtro) | ver si cambia γ_bf | cambio despreciable (≤ 1.1e-3 relativo); descartado solo en su parte secular | §8.8 |
| modo vecino de γ_pf (plano) | identificar | es el tercer modo lógico (traslape 1.409 con Y_L) | §8.9 |
| control con n(ω) y n(3ω) (efectivo) | cambio de γ_pf pequeño | con filtro ≤ 2.5e-4; en el plano hasta +7.8% (x = 6) | §8.12 |
| hermiticidad y positividad | 1e-10 y −1e-9 (plano); 1e-8 y −1e-8 (filtro) | filtro: 0 de 14 superan el umbral vigente; con el anterior, 13 y 2. Plano: 1 de 8 supera 1e-10 | §8.10 |
| convergencia en N con filtro (N = 20 frente a 22) | cambio < 3% en γ_pf y γ_bf (fijado antes) | **en curso** | §8.11 |

## 17. Historial de tareas de Opus (fases térmicas; la numeración de fases es la de `PENDIENTES_Y_HALLAZGOS.md`, P7)
| fase | fecha | commit | qué se pidió | qué se obtuvo |
|---|---|---|---|---|
| 1 | 2026-09-29 | `040eb0e`, `2652746` | validar el efectivo térmico con el modelo completo en unidades de Ma, N = 22 | κ₂/κ = 0.05 aceptado; en la isla γ_bf del efectivo 20% alto |
| 2 | 2026-09-29 a 09-30 | `faaabea` (lanzamiento), `bedbe5e` (datos), `effff72` (resultados) | separar κ₂/κ de x: punto (0.25, x = 6.86) y la isla con tolerancia estricta | la discrepancia de γ_bf depende de κ₂/κ y no de x; sin dependencia de la tolerancia |
| 3 | 2026-09-30 | `4d23ea4` | κ₂/κ = 0.1 y 0.4, baño plano y tabla de x* | γ_bf c/e 0.856 y 0.776; plano con γ_pf c/e 1.52; tabla de x* estimados |
| 4 | 2026-09-30 | `2b64d26` | origen del 1.52 y barrido en N del plano; puntos y notas en la figura térmica | canal n_q; N = 18 no converge; figura con puntos validados |
| 5 | 2026-09-30 | `1603317` | hipótesis del desplazamiento, modo vecino, log de hermiticidad | desplazamiento reproduce el canal con 16% de exceso; vecino = tercer modo lógico |
| 6 | 2026-09-30 a 10-01 | `efda948` (datos), `163e971` (documentación) | independencia de x (x ≈ 9), umbrales del filtro, desplazamiento con filtro; N = 24 si cabía | independencia sostenida; N = 24 no cabe; desplazamiento con filtro despreciable |
| N = 20 | 2026-10-01 | sin commit | convergencia en N con filtro (κ₂/κ = 0.25, x = 6.86, N = 20) | **en curso desde 08:26, sin resultado [VERIFICAR]** |
| x* con piso | 2026-10-01 | sin commit de datos aún | corregir x* por el crecimiento relativo del piso | solo cambia en κ₂/κ = 0.4 (+0.028 y +0.081); `termico_x_estrella_piso.csv` |
