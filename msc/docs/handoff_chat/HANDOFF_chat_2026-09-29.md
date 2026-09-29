# HANDOFF.md: paper de generalización de qubits gato estabilizados por un qubit auxiliar (g_x, g_z)

Documento de traspaso para continuar en Claude Code CLI con acceso a archivos. Escrito el 2026-09-29 a partir del chat (resumen de compactación + lectura de archivos). Idioma: español. Todo lo marcado con [VERIFICAR] no pude comprobarlo al escribir este archivo.

Regla de lectura: lo que aparece en la sección 12 ("Superado / retirado") NO debe usarse aunque aparezca en documento_completo v1 o en reportes viejos de Opus.

---

## 0. Índice

1. Contexto, autor y preferencias de Jhon
2. Idea del trabajo (dos frases) y alcance
3. Modelo y notación
4. Derivaciones paso a paso
5. Resultados numéricos consolidados (tablas)
6. Errores de la literatura
7. Plataformas
8. Térmico
9. Verificaciones independientes (V1 a V6, C1 a C9) y criterios de calidad numérica
10. Historial de tareas de Opus (1 a 47) con estado
11. Inventario de archivos y rutas
12. Superado / retirado / errores míos y del otro agente
13. Pendientes y abiertos
14. Prompts vigentes para Opus / Claude Code
15. Convenciones de código y de LaTeX
16. Riesgos de credibilidad y autoría

---

## 1. Contexto, autor y preferencias de Jhon

- Autor: Jhon Sebastián García Barrera, MSc Ciencias Físicas, UNAL Manizales. Director: Edgar Gómez. Email: jhons.garciab@uqvirtual.edu.co. Se comunica en español.
- Proyecto de Claude "Trabajo de Grado" (tesis: emisión de paquetes de n-fonones en molécula excitónica vía procesos de Stokes; archivos pureza_2qds.py, 2qds_trayectorias.py, forster_2qds.py, Bibli.bib, Kap3.tex, Kap4.tex, marco_teorico.tex, etc.). El paper actual es una línea distinta pero comparte convenciones.
- Objetivo actual: paper de generalización para Physical Review A (autor único) sobre qubits gato estabilizados por un qubit con acoplamientos transversal g_x y longitudinal g_z a un oscilador. Familia de referencia: Ma 2019, Hou 2024, Liu 2025, Naseem 2026.
- Exigencias: derivaciones correctas, verificación numérica independiente (Claude Code y Opus 5.5), "sin errores". Cero tolerancia a hechos inventados.
- Peticiones recurrentes: revisar cada reporte de Opus (solo la física; lo visual lo revisa Jhon), preparar prompts para Opus, un .tex amplio, figuras que muestren la generalización (no "líneas rectas con puntos").
- Preferencias de formato (LaTeX): clase report; sin \textbf, \textit, \emph; sin em-dashes en prosa; derivaciones paso a paso sin omitir; resultados en caja; .tex y .bib juntos; sin \newpage; babel/babelprovide + onehalfspacing.
- Preferencias de trabajo: respuestas en español; honestidad directa incluyendo riesgos de credibilidad; terminal un comando a la vez; prefiere dialogar cambios antes de editar.
- Instrucciones del proyecto: código Python/QuTiP completo, comentarios en español, validaciones de traza, hermiticidad y positividad de ρ; respetar γ ≪ κ y advertir si se viola; citar desde Bibli.bib cuando sea posible (para el paper se usa refs_doc.bib).
- Memoria: sesión ligada al Project 019c2136-aee8-758c-8ad1-e93d16757a3f; escrituras de memoria solo bajo /projects/019c2136-aee8-758c-8ad1-e93d16757a3f/. No mencionar la memoria en las respuestas.
- Recomendación hecha a Jhon (y que debe mantenerse): revisión humana independiente de las derivaciones y hablar con Gómez la autoría, además de revisar la política de APS sobre uso de IA.

---

## 2. Idea del trabajo

Dos frases: se estudia un qubit gato mecánico (oscilador) estabilizado por un qubit auxiliar con acoplamientos mixtos g_x σ_x + g_z σ_z, conducido cerca de la resonancia de dos fotones, que genera una pérdida efectiva de pares (κ₂) y una conducción cuadrática (G). Se deriva y se verifica numéricamente el hamiltoniano efectivo correcto (con corrimientos que la literatura omite), la tasa de phase-flip κ₁, la figura de mérito κ₁/κ₂ = (5/72)(κ/g_z)², el efecto de un baño filtrado, el confinamiento, y el límite térmico.

Mensaje central: la figura de mérito es independiente de g_x y de ω; contradice la receta de Naseem de aumentar κ; el gato solo se forma en la resonancia vestida ω_p* = 2(ω − 4g_x²/3ω), no en ω_p = 2ω.

---

## 3. Modelo y notación

H(t) = ω a†a + (ω_q/2) σ_z + (a + a†)(g_x σ_x + g_z σ_z) + Ω(σ₊ e^{−iω_p t} + h.c.), con ω_q ≈ 2ω.

Disipación (T=0): κ D[σ₋] (qubit) y γ D[a] (pérdida intrínseca del oscilador). Térmico: κ(n_q+1) D[σ₋] + κ n_q D[σ₊]; γ(n_m+1) D[a] + γ n_m D[a†].

Convención D[o]ρ = oρo† − ½{o†o, ρ}. Régimen: γ ≪ κ (débil) y κ, g ≪ ω para la eliminación adiabática (Liu viola κ/ω = 0.9).

Definiciones: G = 2 g_x g_z / ω (coeficiente de pares); κ₂ = 4G²/κ; |α|² ≈ 4 (valor típico); κ₁ = tasa de phase-flip del canal de un fonón; Π: P_c = población en el código; P_e = población del qubit excitado.

Cantidades: A = g_x σ₋ a† ; B = g_z σ_z a ; h₁ = A + B (frecuencia ω); h₃ = g_x σ₋ a (frecuencia 3ω).

---

## 4. Derivaciones paso a paso

### 4.1 Hamiltoniano efectivo (James–Jerke, segundo orden)

Imagen de interacción respecto a H₀ = ω a†a + ω σ_z (ω_q = 2ω). Los términos del acoplamiento (a + a†)(g_x σ_x + g_z σ_z) se separan por frecuencia:
- Frecuencia ω: h₁ = g_x σ₋ a† + g_z σ_z a (y su h.c.).
- Frecuencia 3ω: h₃ = g_x σ₋ a (contrarrotante).
- El término g_z σ_z a† es el h.c. de B.

James–Jerke: si H_I = Σ_k (h_k e^{−iω_k t} + h.c.), entonces H_ef = Σ_k (1/ω_k) [h_k†, h_k].

Conmutadores necesarios:
- A†A = g_x² σ₊σ₋ a a† = g_x² |e><e| (n+1); AA† = g_x² σ₋σ₊ a†a = g_x² |g><g| n. Luego [A†,A] = g_x² [(n+1)|e><e| − n|g><g|].
- [B†,B] = g_z² σ_z² [a†, a] = −g_z².
- [A†,B] = g_x g_z [σ₊ a, σ_z a] = g_x g_z (σ₊σ_z − σ_zσ₊) a² = −2 g_x g_z σ₊ a² (porque σ₊σ_z = −σ₊ y σ_zσ₊ = +σ₊). Idem [B†,A] = h.c.
- [h₃†,h₃] = g_x² (σ₊σ₋ a†a − σ₋σ₊ a a†) = g_x² [n|e><e| − (n+1)|g><g|].

Suma: (1/ω)[h₁†,h₁] + (1/3ω)[h₃†,h₃]:
- Sector |e>: (g_x²/ω)(n+1) + (g_x²/3ω) n = (g_x²/ω)(4n/3 + 1).
- Sector |g>: −(g_x²/ω) n − (g_x²/3ω)(n+1) = −(g_x²/ω)(4n/3 + 1/3).
- Pares: −(2 g_x g_z/ω)(σ₊ a² + h.c.) = −G(σ₊ a² + h.c.).
- Constante: −g_z²/ω.

Resultado (caja):

H_ef = (g_x²/ω)[(4n/3 + 1)|e><e| − (4n/3 + 1/3)|g><g|] − G(σ₊ a² + a†² σ₋) − g_z²/ω.

Corrimientos: oscilador en |g>: −(4g_x²/3ω) n; en |e>: +(4g_x²/3ω) n. Qubit (n=0): splitting = g_x²/ω + g_x²/(3ω) = 4g_x²/3ω (3/4 rotante y 1/4 contrarrotante; Bloch–Siegert). Verificado por diagonalización exacta: el error relativo escala como g_x² (cuarto orden en el hamiltoniano), y el signo de los pares se verificó en los 4 cuadrantes de signos de (g_x, g_z).

### 4.2 Estado oscuro y resonancia vestida

Marco rotante a ω_p/2 para oscilador y qubit; con el drive Ω σ₊ y los pares: H_rot = Δ_m a†a + Ω(σ₊ + σ₋) − G(σ₊a² + h.c.) + ...
- |ψ, g> es estacionario (estado oscuro) si a²|ψ> = (Ω/G)|ψ> (el drive cancela los pares) y la desintonía del oscilador es cero, Δ_m = 0.
- La desintonía efectiva en |g> es Δ_m = ω − ω_p/2 − 4g_x²/3ω. Se anula en ω_p* = 2(ω − 4g_x²/(3ω)).
- Los estados propios: |±α> con α² = Ω/G.
- Ma con ω_p = 12 (= 2ω, ω=6): P_c ≈ 0.81, no hay gato; con ω_p = 11.98: P_c = 0.9975. Máximo medido 11.9806 / 11.97993 ± 0.0005, predicho 11.9800.

Ancho de la resonancia: FWHM = c·G con c = 2.87 ± 0.25 (siete valores de κ₂/κ: c = 2.454, 2.666, 2.878, 3.050, 3.100, 3.116, 2.826 para κ₂/κ = 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1). Curva asimétrica (en κ₂/κ=1 semiancho izq. 0.285 κ₂, der. 1.082 κ₂). Ajuste libre FWHM ∝ κ₂^0.550 = G^1.101. Depende de |α|²: ≈ 3G para |α|²=4, ≈ 3.7 a 4.2 G para |α|²=2 (P1: colapso parcial aceptado). Con d=12 fijo el qubit no sigue al drive; con qubit siguiendo, c ≈ 2.94.

### 4.3 Reiter–Sørensen (tasas efectivas)

L_ef = L H_NH^{−1} V₊, con H_NH = H_g − i(κ/2)|e><e|.
- Canal de pares: acoplamiento G con decaimiento κ/2 en amplitud da κ₂ = 4G²/κ (salto √κ₂ a²).
- Canal de un fonón, mediado por h₁ (frecuencia ω): Γ₁⁻ = g_x² κ / (ω² + κ²/4) ≈ g_x² κ/ω² (salto a, pérdida).
- Canal de un fonón, mediado por h₃ (3ω): Γ₁⁺ = g_x² κ/(9ω² + κ²/4) ≈ g_x² κ/(9ω²) (salto a†, ganancia).
- κ₁ = Γ₁⁻ + Γ₁⁺ ≈ (10/9) g_x² κ/ω².
- Corrimiento del oscilador con dependencia en κ: δ₁ = −(4g_x²/3ω)[1 − (7/9)(κ/2ω)²] (Tarea 41).

### 4.4 Phase-flip y paridad

Con P = (−1)^n: D[a]†P = a†Pa − ½{n,P}; como Pa = −aP, a†Pa = −nP, luego D[a]†P = −2nP. Análogamente D[a†]†P = aPa† − ½{a a†, P} = −2(n+1)P. Entonces

d<P>/dt = −2Γ₁⁻ <nP> − 2Γ₁⁺ <(n+1)P>.

Para un gato con <nP> ≈ |α|² <P>: γ_pf = 2[Γ₁⁻ |α|² + Γ₁⁺(|α|² + 1)] (+ 2γ|α|² si hay pérdida intrínseca). El "+1" en el segundo término estaba faltando en una versión previa (corregido). |α|² = |<(a − g_z/ω)²>| (desplazamiento polarónico). Piso de paridad del gato finito: e^{−2|α|²} ≈ 3.4e-4 (por eso los ajustes son A e^{−kt} + c).

### 4.5 Figura de mérito

κ₂ = 4G²/κ con G² = 4 g_x² g_z²/ω² ⇒ κ₂ = 16 g_x² g_z²/(ω² κ).
κ₁/κ₂ = (10/9)(g_x² κ/ω²) · ω² κ/(16 g_x² g_z²) = (10/144) κ²/g_z² = (5/72)(κ/g_z)².

Independiente de g_x y de ω. Corrección [1 + 1/(10|α|²)] por el "+1" de Γ₁⁺. Precisión medida: ≲0.4% (κ₂/κ ≲ 0.05), ~1% (0.2), ~2% (1) (corrección no adiabática; P4 abierto).

Umbral Guillaud–Mirrahimi: κ₂/κ₁ = 220 ⇒ (72/5)(g_z/κ)² ≥ 220 ⇒ g_z/κ ≥ √(220·5/72) = 3.909 ≈ 3.9 (solo por este canal). Contradice la receta de Naseem (aumentar κ empeora κ₁/κ₂ ∝ κ²).

### 4.6 Baño filtrado

Densidad espectral Lorentziana con ancho κ_f y acoplamiento J: κ_eff(δ) = κ κ_f²/(4δ² + κ_f²), con 4J²/κ_f = κ (a δ = 0 recupera κ).
κ₁^filt = g_x² [κ_eff(ω)/ω² + κ_eff(3ω)/(9ω²)].
Mejora κ₁/κ₁^filt ≈ (4ω² + κ_f²)/κ_f²: 19.5, 166.5, 1838, 1.65e4 para κ_f = 3, 1, 0.3, 0.1 (en las unidades de la validación; concuerdan con ω_efectivo² ≈ 41.6 [VERIFICAR unidades: no reconstruí qué ω se usó]); coincidencia con la simulación ≤ 0.3%. Umbral g_z/κ baja de 3.94 a 0.094 (factor 0.0238). Δ_filtro/Δ_plano (brecha del confinamiento) = 1.002, 1.014, 1.037, 1.051, 0.955, 0.480 para κ₂/κ = 1e-3 … 1. Canales Γ₁± verificados por separado al 0.2% (V5).

### 4.7 Confinamiento: brecha exacta del modelo mínimo

Modelo mínimo H = G[(a² − α²)σ₊ + h.c.], Γ₂ = 4G²/κ.

Caso α = 0 (exacto): N̂ = n + 2σ₊σ₋ se conserva ([H, N̂] = 0 porque a²σ₊ baja n en 2 y sube σ₊σ₋ en 1). Bloque N: base {|N,g>, |N−2,e>}, con a²|N> = √(N(N−1))|N−2>:

H_N = [[0, G_N],[G_N, −iκ/2]], G_N = G√(N(N−1)).

Eigenvalores E_N^± = −iκ/4 ± √(G_N² − κ²/16). El salto σ₋ lleva el bloque N a N−2 ⇒ el superoperador es triangular por bloques y λ = −i(E_i^{(N)} − E_j^{(M)*}) con M el bloque del código (E = 0).

Bloque N = 2: G_2² = 2G² = κΓ₂/2 ⇒ G_2² − κ²/16 = −(κ²/16)(1 − 8Γ₂/κ). Brecha (modo lento):

Δ = (κ/4)[1 − √(1 − 8Γ₂/κ)] para Γ₂ ≤ κ/8; Δ = κ/4 para Γ₂ > κ/8 (punto excepcional, Γ₂ = κ/8).

Límite adiabático: √(1−x) ≈ 1 − x/2 − x²/8 con x = 8Γ₂/κ ⇒ Δ ≈ Γ₂ + 2Γ₂²/κ. Pendiente dΔ/dΓ₂ = 1/√(1 − 8Γ₂/κ). Punto excepcional del bloque N: G_N² = κ²/16 ⇒ Γ₂^PE(N) = κ/(4N(N−1)). Techo κ/4 vía Im E = −(κ/2) p_e (p_e = peso en el excitado ≤ 1/2 en el PE). Verificado numéricamente (≤ 2e-13 en 18 valores de Γ₂).

Caso α ≠ 0: sin PE agudo, cruce suave (Γ₂^× ≈ 0.18, 0.10, 0.08 κ para α² = 1, 2, 3). Numéricamente: c(1) ≈ 1.40, c(2) ≈ 2.57, c(3) ≈ 3.31 (Δ = c κ₂ para los primeros κ₂/κ); satura en ≈ 0.28κ con |α|² = 4 (no es techo estricto); c = lím Δ/κ₂ = 4.14; saturación desde κ₂/κ ≈ 1/(4c) = 0.06; κ₂^eff/κ₂ = 0.98, 0.84, 0.50, 0.24, 0.058 (κ₂/κ = 1e-3, 0.01, 0.045, 0.16, 1).

RETIRADO: la "brecha con máximo en Γ₂/κ ≈ 0.13" y la rama de Floquet con |Im λ| ∝ N fueron artefactos de modos de borde de Fock (Tareas 35, 38, 45, 47). La tasa dinámica es la relevante (C9).

### 4.8 P9: corrimiento del qubit dependiente de n

Con H_ef, el corrimiento del qubit depende de n: χ n |e><e| con χ = (8/3) g_x²/ω (diferencia entre el sector e y g: (4/3 + 4/3) = 8/3). Esto ralentiza el confinamiento. Frontera: χ|α|²/κ ≈ 0.3 (costo ~5%). κ₂/κ = 16 g_x² g_z²/(ω²κ²) (exacto; la forma 0.45 (g_z/κ)²/(ω/κ) del reporte [VERIFICAR]); χ|α|²/κ = (32/3)(κ₂/κ)(ω/κ)/(16 (g_z/κ)²) (expresión en términos de κ₂/κ, ω/κ y g_z/κ). Ma: 2.7 (fuera); Naseem: 0.19 (dentro). En la fila g_z/κ = 10 de la figura térmica χ|α|²/κ = 0.333 > 0.3 (fuera).

### 4.9 Base polarónica (C6)

Completar el cuadrado en el laboratorio: con el qubit en |g>, g_z σ_z (a + a†) → −g_z(a + a†) + ω a†a ⇒ oscilador desplazado d = +g_z/ω. Código fijo {D(g_z/ω)|±α_nom>} para P_c (1 − P_c baja de 2.5e-3 a 1.3e-3 en Ma). α_eff solo para γ_pf. d_opt es 2.4% menor que g_z/ω en Ma (P5, irrelevante). Medición estroboscópica en t = n T_p; propagador de Floquet en el laboratorio con T_p = 2π/ω_p; modo de paridad λ ≈ +1 con |Tr(PR)|/‖R‖ ≈ 1.41; modo espurio si peso de borde > 0.5 o inestable al 0.3% en tres N (N, N+6, N+12); lo no resuelto se reporta como "no resuelto".

### 4.10 Piso intrínseco por γ

κ₁ → κ₁^filt + γ (pérdida del oscilador). Error de bit-flip/phase-flip: ε ≥ γ/κ₂^eff ⇒ para el umbral 220 se necesita κ₂^eff/κ ≥ 220 γ/κ: 0.0044 para γ/κ = 2e-5 (Ma), 0.044 para 2e-4 (Naseem 1.5e-4). Con γ/κ = 2e-4, ε = 1/220 solo se alcanza en la franja κ₂/κ ≈ 0.19 a 0.33 y g_z/κ ≳ 10 (zona verificada). Punto (κ₂/κ, g_z/κ) = (0.25, 14), N = 22: ε_completo = 4.39e-3 < 1/220 (margen ~3.5%), ε_completo/ε_mapa = 1.028; sin filtro saldría ≈ 6.3e-3 (estimación mía; mejora del total solo 1.44×). Precisión con filtro ~2 a 3% (N=20 → 22).

### 4.11 Eliminación previa a Reiter–Sørensen: comprobación de coeficientes

Sanity: 10/144 = 5/72 = 0.069444; √(220·5/72) = 3.909 (hecho con Python en esta sesión). f_q a 10 mK con x = 10.11: 10.11 × 208.37 MHz = 2.107 GHz (coincide con 2.11 GHz del reporte).

---

## 5. Resultados numéricos consolidados

### 5.1 Lo esencial

| Resultado | Valor | Estado |
|---|---|---|
| H_ef correcto | ver 4.1 | verificado por diagonalización exacta |
| ω_p* | 2(ω − 4g_x²/3ω) | máx. medido 11.9806 vs predicho 11.9800 (Ma) |
| P_c Ma en ω_p = 12 | ≈ 0.81 (sin gato) | |
| P_c Ma en ω_p = 11.98 | 0.9975 | |
| κ₁/κ₂ | (5/72)(κ/g_z)² [1+1/(10|α|²)] | ≲0.4% a κ₂/κ ≲ 0.05; ~2% a κ₂/κ = 1 |
| Umbral GM | κ₂/κ₁ = 220 ⇒ g_z/κ ≥ 3.9 | solo canal de un fonón |
| Filtro | κ₁ mejora (4ω²+κ_f²)/κ_f² | ≤ 0.3% |
| Umbral con filtro | g_z/κ ≥ 0.094 | factor 0.0238 |
| Brecha (α=0) | (κ/4)[1−√(1−8Γ₂/κ)] | exacto |
| c = FWHM/G | 2.87 ± 0.25 | n = 7 |

### 5.2 Tarea 45/47: cifras clave del esquema de Ma (g_x = 0.05, ω = 6, N = 22, |α|²=4)

Tasas de paridad (ajuste / espectral) y κ₁/κ₂ vs predicción (5/72)(κ/g_z)²:
- g_z/κ = 5: full 1.873e-5 / 1.879e-5, κ₁/κ₂ = 2.810e-3 (previsto 2.778e-3), P_c máx 0.99930.
- g_z/κ = 12: full 1.801e-5 / 1.805e-5, κ₁/κ₂ = 4.690e-4 (previsto 4.823e-4), P_c máx 0.99517.
- Sin g_z σ_z a pero con pares explícito: P_c máx 0.99996 (g_z/κ=5) y 0.99998 (g_z/κ=12); κ₁/κ₂ sigue la predicción. Es decir, la caída de P_c a g_z/κ = 12 la causa el término g_z σ_z a (no los contrarrotantes).
- Quitar términos de 3ω sin re-sintonizar: κ₁ sube ~24% (drive desintonizado 2g_x²/(3w) = 2.8e-4 = 0.34 κ₂).
- Quitar 3ω y re-sintonizar (ω_p = 2(ω − g_x²/ω), d = ω_p): κ₁/full = 0.882 (g_z/κ=5) y 0.898 (g_z/κ=12), previsto 0.900 (1/(10/9)).

### 5.3 Filtro armónico N_f (Tarea 45b)

N_f = 3 y 4 idénticos ⇒ converge en N_f = 3; N_f = 2 alcanza con error ≤ 3% (κ₁ −1.5% para κ_f = 0.3, −0.1% para κ_f = 1; brecha +2.8% y +0.4%). Valores absolutos a N=10 no valen (κ₁ 2.5× el de N=16); solo comparación entre N_f. La pérdida de brecha con filtro (4.2e-3 plano a 9.8e-4 con κ_f = 0.1, factor ~4) es cualitativa, ~6% de incertidumbre en N. κ₁ converge (Δrel 2.9e-4 entre N=16 y 20).

### 5.4 Curvas de resonancia P_c(s), s = (ω_p − ω_p*)/κ₂ (Tarea 45e, t = 120/κ₂)

Picos en s = 0 (< 0.01 κ₂) para los siete casos; P_c pico = 0.9986, 0.9986, 0.9985, 0.9985, 0.9984, 0.9983, 0.9975 (κ₂/κ = 0.03…1). P_e en el pico: 1e-4 a 2.6e-3. FWHM/κ₂ = 7.084, 5.961, 4.551, 3.409, 2.830, 2.203, 1.413. Interpolación PCHIP (la cúbica daba P_c > 1).

### 5.5 Isla de parámetros (γ/κ = 2e-4)

Franja κ₂/κ ≈ 0.19 a 0.33 con g_z/κ ≳ 10 (ver 4.10). Fila g_z/κ = 10 fuera de la frontera de χ (0.333 > 0.3). |α|² = 4, ω/κ = 200.

---

## 6. Errores de la literatura (confirmados numéricamente)

- Ma 2019, Ec. (4): omite el corrimiento del oscilador −(g_x²/ω) n (correcto −4g_x²/3ω dentro de |g>) y da 3g_x²/ω para el corrimiento del qubit (correcto 4/3 g_x²/ω). El coeficiente de pares es correcto.
- Liu 2025, Ec. (11): omite −(4g_x²/3ω) n. Coeficiente de pares correcto. Además: en la Ec. (11) el g_eff = 4g_xg_z/ν con drive ε_p da α² = ε_p/g_eff = 2.5 (α = 1.58), pero la Ec. (9) con ε_p cos(ω_p t)σ̃_x da por RWA ε_p/2 (α² = 1.25). Para coherencia, el coeficiente del coseno debería ser 2ε_p. Inconsistencia interna probable (mi inferencia, no verificada contra su derivación).
- Naseem 2026: el código no incluye δ₁ en el H efectivo y su Ec. (11) usa g_eff = g en lugar de 2g (Tareas 7–8: Rabi de dos fotones y BCH polarónico confirman g_eff = 2g; las Tareas 2–3 que sugerían g_eff = g se retiraron como no concluyentes). Su receta de aumentar κ es contraria a κ₁/κ₂ ∝ κ².
- Hou 2024: no leído en detalle [pendiente].

Liu, Ec. (9) completa (Tarea 46 / 47b): el gato no se forma en κ/ω = 0.9 (γ=16, ν=35.4, 2π·MHz). F ≤ 0.35 vs 0.95 del efectivo; |<a²>| máx 0.4 a 1.1 (analítico 1.25 o 2.5); la fórmula de la paridad falla 95 a 99%. Convergencia en N y validaciones de traza/hermiticidad/positividad cumplidas (|Tr ρ − 1| ≤ 4.4e-16; ‖ρ−ρ†‖ ≤ 1.3e-15; autovalores ≥ −1.5e-10). Interpretación: la eliminación adiabática no vale en ese régimen. Conclusión numérica mía, no contrastada con otra fuente.

---

## 7. Plataformas (documento v1; tabla pendiente de rehacer)

| Plataforma | κ/ω | g_z/κ | κ₂/κ | κ₁/κ₂ | notas |
|---|---|---|---|---|---|
| Ma | 0.005 | 7.1 | 1.0 | 1.4e-3 | hf/kT(10 mK) = 58; χ|α|²/κ = 2.7 (fuera de la frontera) |
| Naseem | 0.001 | 60 | 2.07 | 9.2e-5 | x = 0.96 (f_q = 200 MHz) a 10 mK; falla térmica; χ|α|²/κ = 0.19 |
| Liu | 0.90 | 0.22 | 0.031 | fuera de validez | el gato no se forma |

No cubre iones (de Matos Filho 1996). Térmico para Naseem con κ₂/κ = 2.07 no calculado.

---

## 8. Térmico

- η = γ_pf/γ_bf (sesgo); x = h f_q/(k_B T) con f_q la frecuencia del QUBIT; n_q = 1/(eˣ − 1); n_m = 1/(e^{x/2} − 1) (oscilador a f_q/2). k_B T/h a 10 mK = 208.37 MHz.
- Modelo: efectivo estático con qubit y filtro explícitos, N = 22 o 24, en la isla (κ₂/κ = 0.25, g_z/κ = 14, |α|² = 4, ω/κ = 200).
- x*(η=100) / x*(η=220):
  - filtrado γ/κ = 2e-5: 10.11 / 10.91 (f_q 2.11 / 2.27 GHz a 10 mK)
  - filtrado γ/κ = 2e-4: 7.78 / 8.59
  - plano γ/κ = 2e-5: 8.19 / 9.01
  - plano γ/κ = 2e-4: 7.18 / 8.00
- γ_bf ∝ n_q^{1.00 a 1.02} con coeficiente 0.032 κ (plano) y 0.041 κ (filtro), NO 0.05 universal. c_bf(κ₂/κ) ∝ (G/κ)^{3 a 4}.
- Con filtro, κ₂/κ domina: x*(100) = 5.0, 6.9, 9.7, 10.8 para κ₂/κ = 0.02, 0.05, 0.2, 0.4; g_z apenas influye.
- Supresión con |α|²: d lnγ_bf/d|α|² ≈ −0.55 / −0.64 (plano), ≈ −0.42 (filtro); coincide con Tarea 44 (Gautier, g₂/κ ≈ 0.3); supresión fuerte −1.0…−1.4 solo para g₂/κ ≲ 0.1.
- 34 de 160 puntos no convergidos (x ≳ 13: piso T=0 dependiente del truncamiento). Región caliente n_q > 0.3 excluida. Mesetas de η a x ≳ 13 = piso de truncamiento. η = 220 no es un criterio de sesgo (origen de η = 100/220 sin documentar; pendiente).
- Hermiticidad cruda ≤ 8.2e-13 (E7: la de Tarea 37 se medía tras simetrizar por paridad).
- Reproducción mía de umbrales: dentro de 0.07 en x*.
- Mapas con el modelo efectivo (modelo completo ~90 h). Validación con el modelo completo en x = 10.1, N = 22 pendiente; prompt enviado a Opus (ver 14).
- Detalle abierto: elección de n_q vs ocupación a ω y 3ω en los canales de un fonón (~1% en γ_pf a x ≈ 10).

---

## 9. Verificaciones y criterios de calidad

### 9.1 Verificaciones independientes (V1 a V6, carpeta verificacion_independiente/)

V1 diagonalización del H_ef vs exacto; V2 ... V6 (brecha, filtro por canales al 0.2%, etc.). Lección: el acuerdo del 0.05% en V3 era compensación de errores. Ver v1_diag.py … v6_brecha.py y los reportes en /root/.claude/projects/-home-claude/e5642fee-.../tool-results/b344c1uz3.txt [VERIFICAR detalle de cada V].

### 9.2 Criterios de calidad numérica (adoptar siempre)

- Validar por cada corrida: |Tr ρ − 1|, ‖ρ − ρ†‖, autovalor mínimo de ρ ≥ −tol.
- Truncamiento: con |α|² = 4 y filtro usar N ≥ 20; N=16 dio artefactos (exceso ×2.9 en P10). N = 24 requiere ~13 GB (no cabe en 14 GB con filtro).
- Modos espurios: peso de borde > 0.5 o inestable al 0.3% en tres truncamientos ⇒ espurio; lo restante se reporta "no resuelto".
- No razonar mecanismos físicos con datos no convergidos.
- QuTiP: validacion/ usa 4.7.6; validacion_ma/ usa 5.3.1.
- En marco rotante el término de pares estático es +|G| = −(2 g_x g_z/ω) según convención del código: en ma2.py el signo es `H0 - p['G']*(sp*a*a + sm*a.dag()*a.dag())` (G negativo del H_ef); con el signo equivocado el gato se forma en ±2 real en lugar de ±2i (P_c = 0.0366).

---

## 10. Historial de tareas de Opus (resumen y estado)

Rondas 1–14 (Naseem, validacion/, muchas conclusiones superadas):
- Bug Re_S1_plus factor 9: despreciable.
- Tareas 16/19: calentamiento ∝ g_x² (Γ₁ estándar); modos ultralentos (paridad real 0.052 vs plateau 0.87).
- Tarea 21: brecha del efectivo diverge (superada por artefacto de borde).
- Tareas 22/25: supresión exponencial de bit-flip; resonancia vestida en δ_m = +0.048.
- Tareas 31–32: ningún ingrediente estático produce un máximo.
- Tarea 34: n_q = 0.6 destruye el gato.
- Tareas 39–44: esquema de Ma; Tarea 41 (δ₁), Tarea 42 (α²=4, N=22; umbral de peso de borde 0.1 vale, no cambia con ≥3e-2), Tarea 43 (filtro κ_f = 0.3, N_f = 2, N = 16 vs 20), Tarea 44 (Gautier, supresión con |α|²).
- Tareas 45 (a–e): ver 5.2 a 5.4. 45d: error mío del signo de pares corregido en ma2.py antes del commit. 45e: FWHM.
- Tarea 46: Liu Ec. (9) primera versión (convención ε_p): sin gato. Tarea 47(b): convención correcta 2ε_p; tampoco se forma el gato.
- Tarea 47(a): la rama interior no convergida del Floquet no afecta las condiciones iniciales probadas (pesos ~1e-14; solo tres estados iniciales).
- Tarea 47(c): "nocounter" re-sintonizado: κ₁ baja 10 a 12% (previsto 10%).
- Tarea 47(d): FWHM = c·G, c = 2.87 ± 0.25.
- Auditoría de la rama ambigua (N = 20, 26, 32, 38): Γ₂/κ = 3.0: Re 0.13097, 0.13132, 0.12486, 0.11608 y peso de borde 0.0050, 0.0008, 0.0002, 0.0001 (no converge, hibridación posible, no probada). Γ₂/κ ≲ 0.2 brecha estable (0.022 a ~0.19); Γ₂/κ ≳ 0.25 no determinada (grupo estable en 0.19–0.23 coincide con el modelo estático: 0.234 en Γ₂/κ = 1).
- P2: mapas 1D/2D con N=16 subestimaban P_max; resuelto con N=22. P10: exceso ×2.9 era artefacto. E7: hermiticidad corregida.

Commits en validacion_ma/: 4eb9c27 y 0810030.

Estado: todas las tareas 1–47 cerradas; abiertos: P4, P5 menor, validación térmica completa.

---

## 11. Inventario de archivos y rutas

### Documentos LaTeX (entorno de chat, /mnt/user-data/outputs/)
- documento_completo/documento_completo.tex (+ refs_doc.bib, fig2_val.pdf, fig3_val.pdf, fig2_wigner.pdf, .pdf 23 págs): v1 con 15 secciones. NECESITA v2 (ver sección 12).
- brecha_punto_excepcional.tex/.pdf: derivación paso a paso de la brecha exacta (α=0), techo κ/4, límite adiabático, pendiente, PE del bloque N, cruce suave con α≠0. Su veredicto sobre el "máximo en 0.13" debe leerse con la corrección de artefacto de borde.
- lamb_shift_ma2019.tex, figura_merito.tex, mapa_plataformas.tex (+ pdf) y refs.bib, refs_fm.bib, refs_ma.bib, refs_mapa.bib.
- HANDOFF.md (este archivo).

### Carpeta de Opus (rutas relativas)
- figuras_finales/ : README.md (versión más reciente 5de89b3d, con térmica), PENDIENTES_Y_HALLAZGOS.md (a8a021e0 más reciente), C6_polaron.md, figuras/, codigo/ (comun.py, estilo.py, calc_fig2/3.py, fig2/3/4.py, principal_fig2/3.py, calc_minimo*.py, calc_filtro_completo.py con --gam, verif_figura_central.py, figura_central.py, p9_diagnostico.py, p10_analisis.py, figura_termica.py, calc_termico.py, run_termico.py), data/ (termico_curvas.csv, termico_umbrales.csv, termico_umbral_parametro.csv, termico_ajuste_bf.csv, termico_mapas.npz).
- verificacion_independiente/ : V1–V6 (v1_diag.py … v6_brecha.py, .venv).
- validacion/ : Tareas 1–44, Naseem, QuTiP 4.7.6; audit_worker.py (perfil en Fock), tarea35_worker.py.
- validacion_ma/ : Tareas 39–47, ma_model.py, ma2.py (opciones counter, gz_direct, pair), tarea43_worker.py (N_f, noevo), tarea45a.py, tarea45_analisis.py, tarea45d_worker.py, tarea45e_worker.py, tarea46.py, tarea46_run.py, tarea47b_fig2.py, tarea47b_run.py, tarea47_analisis.py, run_t45*.sh, run_t47*.sh; resúmenes RESUMEN_FINAL_RONDA15.md y RESUMEN_FINAL_RONDA16.md; tablas tarea45a_resultados.md, tarea45_resultados_bcde.md, tarea46_resultados.md, tarea47_resultados.md, tarea47b_fig2_resultados.md, tarea47b_task46_resultados.md; QuTiP 5.3.1.
- Copias en /root/.claude/uploads/e5642fee-16f4-56ab-9207-a18e7e586ae7/ (220 archivos con prefijo hash) y /mnt/user-data/uploads/.
- Transcripción del chat (solo tramo posterior a la compactación): /root/.claude/projects/-home-claude/e5642fee-16f4-56ab-9207-a18e7e586ae7.jsonl (leer con json.loads(strict=False) y errors='replace').
- Scratch: /home/claude/ma/dump.txt y assistant.txt.

### Bibliografía (refs_doc.bib)
Ma2019 PRA 99 022302; Naseem2026 PRA 113 013732 (arXiv:2508.10500); Liu2025 PRA 112 023709 (arXiv:2501.08675); Hou2024 PRA 110 013711; deMatos1996 PRL 76 608; JamesJerke2007 Can. J. Phys. 85 625; ReiterSorensen2012 PRA 85 032111; Zueco2009 PRA 80 033846; GuillaudMirrahimi2021 PRA 103 042413; Gautier2022 PRX Quantum 3 020339; Chamberland2022 PRX Quantum 3 010329; Marquet2024 PRX 14 021019; Putterman2025 PRX 15 011070 (NO Hajr; error mío previo); Ferrari2026 arXiv:2605.24100; Carde2024 arXiv:2410.00975; Hillmann2026 arXiv:2607.08363; Lu2026 arXiv:2607.27771. Revisión de citas pendiente (verificar DOIs y autores antes de enviar).

---

## 12. Superado, retirado y errores

### 12.1 De documento_completo v1 (NO usar)
- "hf_q/kT ≳ 9 para sesgo 100 con |α|² = 2" y el umbral térmico calculado en régimen Naseem: reemplazar por la isla (sección 8).
- "Techo del confinamiento κ/4": ahora ≈ 0.28κ, no techo estricto (κ/4 solo para α=0).
- Coeficiente γ_bf ≈ 0.05 n_q κ: no es constante (0.032 plano, 0.041 filtro).
- La sección de temperatura debe reescribirse con los resultados de la isla.
- Curva del filtro en .tex solo incluye canal de pérdida (falta Γ₁⁺).

### 12.2 Resultados retirados
- Brecha con máximo en Γ₂/κ ≈ 0.13 y rama de Floquet con |Im λ| ∝ N: artefactos de borde de Fock.
- Tareas 2–3 (g_eff = g de Naseem): no concluyentes.
- "La brecha física es monótona y satura en ~0.23" (Tarea 38): hipótesis, no establecido.
- Exceso ×2.9 en P10 (N=16): artefacto de truncamiento.

### 12.3 Errores míos
- Criterio térmico hf_q/kT: hay que usar la frecuencia del QUBIT, no la mecánica.
- C3 generalizado desde un punto.
- Recomendé α_eff para P_c (correcto: código fijo con d = g_z/ω).
- Interpreté el exceso P10 como canal físico.
- Dije que N=24 pesaba 1.4 GB (necesita ~13 GB).
- Atribuí PRX 15 011070 a Hajr (es Putterman et al.).
- Arrastré "γ_bf ≈ 0.05 n_q κ" como constante.
- No diferencié versiones de figura_central al hablar del color (la más azul era otra versión; la última con piso en el color no está en mis archivos).

### 12.4 Errores del otro agente (Opus)
Signo del término de pares en ma2.py (corregido antes del commit); P_e estroboscópico vs promedio; hipótesis del canal polarónico no filtrado (descartada); +1 faltante en γ_pf; acuerdo 0.05% en V3 era compensación; Tarea 37 medía hermiticidad tras simetrizar; mapas N=16 subestimados; primer lanzamiento de 47(a) desde carpeta equivocada (relanzado sin pérdidas).

---

## 13. Pendientes

1. Revisar resultado de Opus para la validación térmica: criterio ±5% en γ_pf y γ_bf, γ_bf(T=0) < 1% del térmico, segundo punto κ₂/κ ≈ 0.05; c_bf(κ₂/κ); origen de η = 100/220; cambios de paneles (a), (b), (c).
2. Tabla de plataformas (rehacer con κ₂/κ, χ|α|²/κ, x, κ₁/κ₂).
3. Fig. 1 en TikZ (esquema del sistema).
4. documento_completo v2 (reemplazar térmica, techo, γ_bf; añadir η, isla, Liu 47b, FWHM = cG, nocounter).
5. Revisión de citas (Bibli/refs_doc.bib) y lectura de Hou et al.
6. P4: déficit no adiabático 1–2% sin explicación analítica.
7. P5 (menor).
8. Brecha física del Floquet a Γ₂/κ ≳ 0.25 (rama interior no convergida hasta N=38).
9. Dubovitskii/bit-flip con desintonía.
10. Naseem térmico a κ₂/κ = 2.07 (f_q = 200 MHz, x = 0.96 a 10 mK).
11. Revisión humana independiente, hablar autoría con Gómez, política APS sobre IA.
12. Elección n_q vs ocupación a ω/3ω en canales de un fonón.

---

## 14. Prompts vigentes para Opus

Prompt enviado (validación térmica): correr segundo punto completo κ₂/κ ≈ 0.05; reportar γ_pf y γ_bf por separado; T=0 completo; ajuste directo de c_bf; documentar origen de η = 100/220 y del uso de n_q; cambios en paneles (a), (b), (c); NO lanzar Naseem. Punto de validación en x = 10.1, N = 22 (modelo completo ~12:10 de terminación).

Siguiente prompt (cuando llegue el reporte): comparar con criterio ±5%; si pasa, congelar la figura térmica y pasar a tabla de plataformas.

---

## 15. Convenciones

- LaTeX: clase report; sin \textbf, \textit, \emph; sin em-dashes; derivaciones paso a paso; resultados en caja; \begin{document} con babel/babelprovide y onehalfspacing; sin \newpage; .tex y .bib juntos.
- Python/QuTiP: comentarios en español, código completo, validar traza, hermiticidad y positividad; advertir si se viola γ ≪ κ; usar tolerancias 1e-12 / 1e-10 en mesolve.
- Cifras: reportar siempre convergencia en N, y N ≥ 20 con |α|² = 4.
- Hacer un comando de terminal por vez; dialogar antes de editar.

---

## 16. Riesgos de credibilidad

- Autor único con verificación asistida por IA: pedir revisión humana; consultar política de APS.
- Las conclusiones sobre Liu (no se forma el gato) son numéricas mías; contrastar con los autores o con una lectura más cuidadosa.
- Precisión de κ₁/κ₂ ~2% a κ₂/κ = 1 (P4 abierto).
- Resultados térmicos con modelo efectivo; validación con modelo completo en un punto.
- No sobregeneralizar: el mapa vale en la isla (κ₂/κ ≈ 0.19–0.33, g_z/κ ≳ 10, γ/κ = 2e-4).
