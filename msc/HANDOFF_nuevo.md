# HANDOFF — paper de generalización: qubits gato estabilizados por un qubit auxiliar (g_x, g_z)

Documento único de traspaso de `msc/`. Reemplaza y absorbe el contenido de todos los demás `.md` del repositorio (archivados en `msc/_archivo/` solo tras aprobación; nada se borra). Verificado contra datos en `msc/consolidacion/` (`afirmaciones.csv`, `por_decidir.csv`, `tareas_historicas.csv`, `clasificacion.csv`, `cobertura_headings.csv`). Tag `pre-consolidacion` = commit `1eae004`. Idioma: español. Sin simulaciones nuevas: todo lo que sigue sale de cachés, logs y scripts ya existentes.

**Índice**
- Parte I: 1. Mapa de archivos · 2. Estado y resultados clave · 3. Discrepancias y decisiones de Jhon · 4. Limitaciones y pendientes · 5. Cómo reproducir
- Parte II: a. Derivaciones · b. Métodos y parámetros · c. Térmico completo · d. Figuras · e. Pendientes y literatura · f. Plataformas y tareas históricas · g. Literatura (fuera de alcance)

---

# Parte I — Estado actual

## 1. Mapa de archivos importantes

### Núcleo de registro (7 archivos, 0.2 MB)
| Archivo | Qué es |
|---|---|
| `HANDOFF.md` (este archivo) | Traspaso único |
| `README.md` | Mapa de carpetas de `msc/` |
| `docs/Informe/informe.tex` | Borrador del paper |
| `docs/Informe/refs_informe.bib` | Bibliografía del informe |

Los otros tres núcleos originales (`Mechanical-Cat-State/figuras_finales/README.md`, `PENDIENTES_Y_HALLAZGOS.md`, `C6_polaron.md`) y `data/preregistro_kf0.2.md` quedan **fusionados en la Parte II** de este archivo (ver índice de coberturas, §1 más abajo) y se archivan junto con los demás `.md`.

### Entregables (29 archivos, 4.2 MB)
`docs/Informe/figuras/{fig_resonancia,fig_merito,fig_filtro,fig_chi,fig_mapa,figura_termica}.pdf`, `docs/Informe/informe.pdf`, y en `Mechanical-Cat-State/figuras_finales/figuras/`: `principal_fig2.{pdf,png}`, `principal_fig3.{pdf,png}`, `figura_central{,_sinpiso}.{pdf,png}`, `figura_termica.{pdf,png}`, `fig2.{pdf,png}`…`fig4.{pdf,png}`, `apendice_*.{pdf,png}`, `p9_diagnostico.{pdf,png}`. Detalle por figura en Parte II.d.

### Cadena de dependencia (7072 archivos, 64.6 MB)
Todo `Mechanical-Cat-State/figuras_finales/codigo/` y `Mechanical-Cat-State/figuras_finales/data/` — código oficial del paper. Inventario completo en `consolidacion/clasificacion.csv` (categoría `cadena`).

### Histórico (2339 archivos, 14.6 MB) — a archivar, nada se borra
`Mechanical-Cat-State/validacion/tareas_naseem/` (1765: Tareas 1–44, Naseem), `tareas_ma/` (398: Tareas 39–47, Ma), `independiente/` (102: V1–V6), `codigo_original/` (41), `docs/` salvo `Informe/` (21: `documento_completo`, `brecha_punto_excepcional`, `figura_merito`, `lamb_shift_ma2019`, `mapa_plataformas`, `handoff_chat`), `papers/` (2).

### Cobertura de los `.md` archivados
500 encabezados (H1–H4) de los 75 `.md` que se archivan, más los 4 fusionados aquí, clasificados uno por uno en `consolidacion/cobertura_headings.csv` (mantenido/fusionado/descartado, destino, motivo). Ninguno quedó sin destino. Resumen:
- **Fusionado** (111 encabezados): el contenido pasó a esta Parte II, principalmente de `figuras_finales/README.md`, `PENDIENTES_Y_HALLAZGOS.md`, `C6_polaron.md`, `preregistro_kf0.2.md` y los `V*_reporte.md` de `validacion/independiente/` (vía ids de `afirmaciones.csv`).
- **Descartado** (≈385, con motivo): casi todos de `validacion/tareas_naseem/` y `validacion/tareas_ma/` — resumidos sin perder el resultado clave en la tabla de tareas históricas (Parte II.f); y `docs/handoff_chat/HANDOFF_chat_2026-09-29.md`, superado por este documento.
- **Mantenido** (4): encabezados de `README.md` raíz, ya reflejados en el §1 de esta Parte I.

### Citas a los 4 `.md` fusionados en scripts (verificado; solo comentarios, ningún `open()`)
| Archivo:línea | Cita | Acción |
|---|---|---|
| `Mechanical-Cat-State/figuras_finales/codigo/run_kf02_N20.sh:2` | `# ... Ver data/preregistro_kf0.2.md.` | actualizada a "Ver HANDOFF.md, Parte II.c — preregistro κ_f/ω=0.2." |
| `Mechanical-Cat-State/figuras_finales/codigo/resultado_kf02.py:1` | docstring con la misma cita | idem |
| `Mechanical-Cat-State/figuras_finales/codigo/fig2.py:72` | `# código fijo (ver README)` | actualizada a "ver HANDOFF.md, Parte II.d — Fig. 2" |
| `Mechanical-Cat-State/figuras_finales/codigo/handoff_cifras.py:1` | `"""... de msc/HANDOFF.md ..."""` | sin cambio (HANDOFF.md sigue existiendo) |
| `Mechanical-Cat-State/figuras_finales/data/handoff_segunda_pasada.txt` | menciones genéricas a HANDOFF.md | sin cambio (dato histórico, no referencia de ejecución) |

Las tres primeras ediciones (comentario/docstring, sin efecto en ejecución) quedan pendientes de tu aprobación final junto con los `git mv`.

---

## 2. Estado y resultados clave

Marca: **V** recalculado y coincide · **R** reportado, no recomputable sin simulación nueva · **H** hipótesis declarada · **S** superado por resultado posterior · **D** discrepancia (ver §3). Cada fila cita id(s) exactos de `afirmaciones.csv`.

### Derivaciones (simbólico, sympy) — detalle en Parte II.a
| Resultado | Valor | Marca | ids |
|---|---|---|---|
| H_ef (James–Jerke) | (g_x²/ω)[(4n/3+1)|e⟩⟨e|−(4n/3+1/3)|g⟩⟨g|] − G(σ₊a²+h.c.) | V | SIM-01, SIM-03, SIM-04, SIM-06 |
| G, κ₂, Γ₁⁻, Γ₁⁺ | G=2g_xg_z/ω; κ₂=4G²/κ; Γ₁⁻=g_x²κ/(ω²+κ²/4); Γ₁⁺=g_x²κ/(9ω²+κ²/4) | V | SIM-02, SIM-08, SIM-09 |
| ω_p* (resonancia vestida) | 2(ω−4g_x²/3ω) | V | SIM-07 |
| γ_pf (phase-flip, con +1) | 2[Γ₁⁻|α|²+Γ₁⁺(|α|²+1)] | V | SIM-13, SIM-14, SIM-15 |
| κ₁/κ₂ (figura de mérito) | (5/72)(κ/g_z)² | V | SIM-10 |
| Umbral κ₂/κ₁=220 ⇒ g_z/κ | ≥3.909 | V | SIM-11 |
| Baño filtrado κ_eff(δ), κ₁^filt | κκ_f²/(4δ²+κ_f²); g_x²[κ_eff(ω)/ω²+κ_eff(3ω)/(9ω²)] | V | SIM-19, SIM-20, SIM-21, SIM-22 |
| Brecha exacta (α=0) | (κ/4)[1−√(1−8Γ₂/κ)] / κ/4 a partir de Γ₂=κ/8 | V | SIM-23, SIM-24, SIM-25, SIM-26 |
| δ₁ con corrección de κ/ω | −(4g_x²/3ω)[1−(7/9)(κ/2ω)²] | V | SIM-16 |
| C6: desplazamiento polarónico d=+g_z/ω | verificado algebraicamente y numéricamente | V | SIM-28, C6-01…C6-14 |

### Efectivo vs. modelo completo (verificación independiente V1–V6)
| Resultado | Valor medido | Marca | ids |
|---|---|---|---|
| H_ef por diagonalización exacta | error ~g⁴ | V | ED-0.005-|g>, ED-0.01-|g>, ED-0.02-|g>, ED-0.04-|g>, ED-esc |
| Resonancia de Ma (ω_p=11.98) | P_c=0.99873 (fijo), 0.99754 (sin desplazar) | V | MA-12, MA-13 |
| Máximo del barrido fino | 11.9799±0.0005 vs. predicho 11.9800 | V | V2-01 |
| γ_pf con +1 (|α|²=2,4,6) | medido/predicho 0.9988, 0.9972, 0.9961 | V | V4b-2-con1, V4b-4-con1, V4b-6-con1 |
| κ₁/κ₂ (15 puntos de la Tabla V4) | 0.979–0.999 | V | V4-RANGO, V4-MAX |
| Baño filtrado, canal por canal | 0.1–0.25% | V | F5a-3-m, F5a-3-p, F5a-5-m, F5a-5-p |
| Brecha exacta (18 valores de Γ₂/κ) | ≤2e-13 salvo punto excepcional (−9.5e-8) | V | V6-0.005, V6-0.02, V6-0.05, V6-0.1, V6-0.12, V6-PE, V6-PE2, V6-sat, V6-N |

### Confinamiento y mapa de diseño
| Resultado | Valor | Marca | ids |
|---|---|---|---|
| c = lím Δ/κ₂ (|α|²=4) | 4.14 | V | MN-01 |
| Saturación de la tasa dinámica | ≈0.28κ (no κ/4; κ/4 es solo para α=0) | V | MN-02 |
| Frontera χ\|α\|²/κ≈0.3 (5% de pérdida) | V | P9-01, P9-02, P9-03, P9-04 |
| Umbral g_z/κ: plano 3.94 → filtro 0.094, factor 0.0238 | V | MU-01, MU-02, MU-03 |
| Costo del filtro en saturación (κ₂/κ=1) | 0.957 (N=26, no comprobado a N=30); se retira "0.480" (N=20, no convergido) | D (corregido en §3) | MF-SAT |

### Modelo completo con filtro, N=16/20/22 (verificación central, panel b)
| Punto | γ_pf/predicho | ε_completo/ε_mapa | Marca | ids |
|---|---|---|---|---|
| (κ₂/κ,g_z/κ)=(0.03,12), N=16/20/22 | 2.880 / 0.986 / 0.978 | 2.831 / 0.987 / 0.979 | V/S | FC-0.03-16-pf, FC-0.03-20-pf, FC-0.03-22-pf, FC-0.03-16-eps, FC-0.03-20-eps, FC-0.03-22-eps |
| (0.3,12), N=16/20/22 | 2.261 / 0.980 / 0.977 | 2.288 / 0.975 / 0.972 | V/S | FC-0.3-16-pf, FC-0.3-20-pf, FC-0.3-22-pf, FC-0.3-16-eps, FC-0.3-20-eps, FC-0.3-22-eps |
| Convergencia N=20→22 | Δγ_pf 0.8% y 0.3%; Δconfinamiento <0.05% | V | FC-conv1, FC-conv2, FC-conv3, FC-prec |

### Térmico (isla κ₂/κ=0.25, g_z/κ=14, ω/κ=200, |α|²=4) — detalle en Parte II.c
| Cociente completo/efectivo (x=6.86) | κ₂/κ=0.05 | 0.1 | 0.25 | 0.4 | Marca | ids |
|---|---|---|---|---|---|---|
| γ_pf | 0.959 | 0.959 | 0.959 | 0.959 | V | T84a-0.05-6.86-gpf_c, T84a-0.1-6.86-gpf_c, T84a-0.25-6.86-gpf_c, T84a-0.4-6.86-gpf_c |
| γ_bf crudo | 0.956 | 0.856 | 0.785 | 0.776 | V | T84b-0.05-6.86-rbf, T84b-0.1-6.86-rbf, T84b-0.25-6.86-rbf, T84b-0.4-6.86-rbf |
| Criterio ±5% en γ_bf | cumple | no | no | no | V | T84-c1, T84-c2 |
| x*(η=100) efectivo/completo [ESTIMADO] | 6.86/6.86 | 8.36/8.25 | 10.11/9.93 | 10.84/10.67 | V | T85-0.05-100-ef, T85-0.1-100-ef, T85-0.25-100-ef, T85-0.4-100-ef (y -co) |
| Prueba κ_f/ω=0.2: E_obs=+6.41e-6, regla cumplida (canal de tamaño ansatz) | — | — | — | — | V | K2-Eobs, K2-Eratio, K2-regla |
| Desplazamiento reproduce el canal n_q del plano | completo/efectivo-kick=0.860 (16% de exceso) | — | — | — | V (redacción "84%" retirada, §3) | PL7-84, PL7-16 |
| Origen de la discrepancia γ_bf(κ₂/κ) | no identificado | — | — | — | H | (pendiente, Parte II.e) |

### Canal polarónico (C6)
| Resultado | Valor | Marca | ids |
|---|---|---|---|
| Desplazamiento d=+g_z/ω en |g⟩ | P_c sube ×1.9 (Ma) a ×3.8 (T42) al desplazar | V | C6-04, C6-05, C6-06, C6-07, C6-10, C6-11, C6-12, C6-13, C6-14 |
| d_opt/d_pred (Ma) | 0.976 | V | C6-02, C6-03 |
| Campo medio explica parte de d_pred−d_opt | 0.22 (no "un tercio") | D (corregido en §3) | C6-18 |

**Resumen por nivel** (1069 filas de `afirmaciones.csv`): nivel 1 — 914 V, 26 D, 29 S, 22 R, 17 H, 8 L. Nivel 2 (`docs/*.tex`) — 31 V, 8 D, 12 S. Nivel 3 (histórico) — ver Parte II.f.

---

## 3. Discrepancias corregidas y decisiones de Jhon

### 3.1 Tabla única de mínimo autovalor por grupo de datos (ninguna celda abreviada)
Criterio original: mínimo autovalor de (ρ+ρ†)/2 ≥ −1e-9. **5 violaciones en todo `Mechanical-Cat-State/figuras_finales/data/`** (id Q-a1, Q-a2, Q-a3 en `afirmaciones.csv`):

| Archivo completo | Grupo | Mínimo autovalor |
|---|---|---|
| `Mechanical-Cat-State/figuras_finales/data/filtro_completo/gx0.0216506_w6_gz0.36_kf0.3_al4_N22_Nf2.npz` | verificación central (κ₂/κ=0.03, g_z/κ=12) | −5.5816553153963716e-09 |
| `Mechanical-Cat-State/figuras_finales/data/filtro_completo/gx0.0239579_w6_gz0.42_kf0.3_al4_N22_Nf2_gam6e-07.npz` | térmico (κ₂/κ=0.05, T=0) | −2.4905620589798748e-09 |
| `Mechanical-Cat-State/figuras_finales/data/filtro_completo/gx0.0338816_w6_gz0.42_kf0.3_al4_N22_Nf2_gam6e-07.npz` | térmico (κ₂/κ=0.1, T=0) | −1.4216126709142994e-09 |
| `Mechanical-Cat-State/figuras_finales/data/filtro_completo/gx0.0684653_w6_gz0.36_kf0.3_al4_N22_Nf2.npz` | verificación central (κ₂/κ=0.3, g_z/κ=12) | −1.3076847945747511e-09 |
| `Mechanical-Cat-State/figuras_finales/data/fig4/est_kf0.3_N8_Nf3.npz` | Fig. 4, método estático (κ_f=0.3, N=8, N_f=3) | −1.1176477435774536e-09 |

Las dos reportadas antes en HANDOFF (−2.49e-9 y −1.42e-9, filas 2 y 3) eran las únicas documentadas en `log_hermiticidad_positividad.txt`; las otras tres (filas 1, 4, 5) están en `data/` pero fuera de ese log. **Decisión tomada (PD-umbral, abajo): se declaran las 5.**

### 3.2 Las 8 decisiones de `por_decidir.csv`
1. **PD-umbral — A.** Los umbrales de hermiticidad/positividad del filtro completo se relajaron (1e-10/−1e-9 → 1e-8/−1e-8) **después** de ver las violaciones (código cambiado en `efda948`, 2026-09-30 23:17 −0500 = 10-01 UTC; datos que violan desde `2652746`, `4d23ea4`, `970edfa`, del 09-29/09-30). Se declara así explícitamente: el cambio de criterio fue posterior a los datos, no una elección a priori.
2. **PD-piso — A.** Se mantiene γ_bf(T=0)<1% del térmico como criterio de aceptación (falla a x≈9 y en la isla a x=10.1: 1.35%–2.52%). No se adopta η_max≥10×nivel de sesgo.
3. **PD-c.** Cifra fijada: **c=2.94±0.27** (ajuste por el origen, serie ω_q=ω_p; recalculado 2.9365, media 2.916±0.273 sobre 5 puntos — id FW-01…FW-06, FW-09). Como verificación cruzada se citan, sin usarse como cifra principal: la serie con ω_q fijo da c=2.87±0.25 (media sobre 7 puntos, Tarea 47d — histórico, no recomputable sin caché) y el ajuste libre sobre esa misma serie da FWHM∝κ₂^0.55.
4. **PD-sat.** Se retira "Δ_filtro/Δ_plano=0.480" en κ₂/κ=1 (N=20,N_f=2, no convergido: P_c estacionario 0.99934). Se reporta **0.957 (N=26,N_f=3), no comprobado a N=30** (id MF-SAT).
5. **PD-9.2 — A.** κ₁/κ₂ de Naseem: 9.2e-5 rotulado "con pérdida intrínseca incluida" (κ₁+γ)/κ₂; se agrega 1.9e-5 como valor nominal (5/72)(κ/g_z)² (ids PL-08, PL-09).
6. **PD-eta — A.** η=100 y η=220 se mantienen como "valores de referencia" sin justificación propia derivada aquí.
7. **PD-central — sinpiso.** Se recomienda `figura_central_sinpiso` para el paper.
8. **PD-naseem — no lanzar.** El térmico de Naseem (x≈0.96, κ₂/κ=2.07, fuera del rango validado x≥6.86) no se lanza; queda "no calculado".

### 3.3 Otras correcciones de redacción (no numéricas)
- El "84%" de §8.7 (desplazamiento reproduce el canal n_q del plano) era ambiguo: el cociente completo/efectivo-con-kick es 0.860 y su inverso 1.163. Se retira "84%"; la cifra correcta y ya reportada es el **16% de exceso** (id PL7-16, PL7-84).
- El −0.55 de la Tarea 44 (supresión de γ_bf con |α|²) se rotula explícitamente como **simulación propia del modelo de buffer de dos niveles de Gautier et al.**, no una medida de Naseem o Ma (id AL-0-9; fila "44" de Parte II.f).

---

## 4. Limitaciones y pendientes abiertas

- Un solo juego de parámetros valida el térmico completo (g_z/κ=14, ω/κ=200, |α|²=4, γ/κ=2e-5, κ_f/ω=0.05); no validado γ/κ=2e-4, otros κ_f, otros g_z/κ.
- Origen de la discrepancia de γ_bf con κ₂/κ (0.956 a 0.776 crudo): abierto.
- Origen del piso de T=0 con filtro (crece con κ₂/κ en el completo, no en el efectivo): abierto.
- Modo vecino de γ_pf con filtro: solo observado, sin comprobar que sea Y_L.
- Origen del residuo de hermiticidad (diagonalización densa): no verificado.
- N_f del completo: solo N_f=2; falta correr con N_f=3.
- Baño plano y supuesto de ruido blanco (n(ω), n(3ω)): no validado, solo diagnósticos.
- Desplazamiento con filtro, parte no secular: requiere Floquet con tiempo, no hecho.
- Naseem térmico (κ₂/κ=2.07): no calculado (PD-naseem).
- Hou 2024: no leído; no está en `papers/`.
- Revisión humana independiente y política de la APS sobre IA: pendiente.
- 117 líneas numéricas de nivel 1 sin verificar línea por línea: `consolidacion/lineas_sin_verificar.csv`.

---

## 5. Cómo reproducir

El entorno `.venv_qutip5` **no existe** en esta copia (está en `.gitignore`). Recrearlo con la versión de los logs de `figuras_finales/data/`:
```bash
cd Mechanical-Cat-State
python3.11 -m venv .venv_qutip5
.venv_qutip5/bin/pip install qutip==5.3.1 numpy scipy matplotlib
```
Reproducir una figura desde caché: `cd Mechanical-Cat-State/figuras_finales && ../.venv_qutip5/bin/python codigo/<script>.py`. Forzar recálculo de un punto: `--rerun`. Barridos: `codigo/run_*.sh`.

Reproducir la verificación de este HANDOFF (numpy/scipy/sympy, sin QuTiP):
```bash
cd msc && python3 consolidacion/verificar_todo.py
```
Compilar el informe: `cd docs/Informe && pdflatex informe && bibtex informe && pdflatex informe && pdflatex informe`.

---

# Parte II — Contenido completo de los documentos fusionados

## II.a — Derivaciones paso a paso

**Modelo.** H(t) = ω a†a + (ω_q/2)σ_z + (a+a†)(g_xσ_x+g_zσ_z) + Ω(σ₊e^{−iω_pt}+h.c.), con ω_q≈2ω, σ_z=|e⟩⟨e|−|g⟩⟨g|. Disipación a T=0: κD[σ₋] y γD[a]. Térmica: κ[(n_q+1)D[σ₋]+n_qD[σ₊]] y γ[(n_m+1)D[a]+n_mD[a†]].

**H_ef (James–Jerke, segundo orden).** A frecuencia ω: g_xσ₋a† (rotante) y g_zσ_za; a 3ω: g_xσ₋a (contrarrotante). Con H_ef=Σ(1/ω_k)[h_k†,h_k]:

  **H_ef = (g_x²/ω)[(4n/3+1)|e⟩⟨e| − (4n/3+1/3)|g⟩⟨g|] − G(σ₊a²+h.c.) − g_z²/ω**, G=2g_xg_z/ω.

Corrimiento del oscilador en |g⟩: −4g_x²/3ω (3/4 rama rotante, 1/4 contrarrotante); en |e⟩: +4g_x²/3ω; transición del qubit con n=0: +4g_x²/3ω. Verificado por diagonalización exacta (id ED-*): error relativo ~g_x² (cuarto orden). Signo del término de pares comprobado en los cuatro cuadrantes de signo (id SG-1, SG-2, SG-3).

**Resonancia vestida.** El estado oscuro |ψ,g⟩ con a²|ψ⟩=(Ω/G)|ψ⟩ existe si la desintonía del oscilador en |g⟩ se anula: ω_p*=2(ω−4g_x²/3ω) (id SIM-07, MA-03). Regla universal: en x=(ω_p−ω_p*)/G el pico cae en x≈0 (|x|≤0.09, salvo un caso con 0.16 — id RU-01, RU-02). FWHM≈cG: ver §3.2 (PD-c) para la cifra fijada.

**Tasas efectivas (Reiter–Sørensen).** Pares: κ₂=4G²/κ. Un fonón: Γ₁⁻=g_x²κ/(ω²+κ²/4) (pérdida, vía ω), Γ₁⁺=g_x²κ/(9ω²+κ²/4) (ganancia, vía 3ω). Corrimiento con κ: δ₁=−(4g_x²/3ω)[1−(7/9)(κ/2ω)²] (id SIM-16).

**Paridad (phase-flip).** D[a]†P=−2nP, D[a†]†P=−2(n+1)P (id SIM-13) ⇒ d⟨P⟩/dt=−2Γ₁⁻⟨nP⟩−2Γ₁⁺⟨(n+1)P⟩ ⇒

  **γ_pf = 2[Γ₁⁻|α|² + Γ₁⁺(|α|²+1)]** (+2γ|α|²), con |α|²=|⟨(a−g_z/ω)²⟩| (id SIM-14).

El "+1" faltaba en el trabajo original (E2, Parte II.e). Verificado en V4b: con |α|²=2,4,6 da 0.9988, 0.9972, 0.9961 (id V4b-2-con1, V4b-4-con1, V4b-6-con1). Hay un piso de paridad del estado estacionario (~7e-4 con |α|²=4), por eso los ajustes temporales son A e^{−kt}+c.

**Figura de mérito.** κ₂=16g_x²g_z²/(ω²κ) ⇒ **κ₁/κ₂=(5/72)(κ/g_z)²** (id SIM-10). Precisión (corrección no adiabática): ≲0.69% para κ₂/κ≲0.05, ~1% en 0.2, ~2% en 1 (id P4-a, P4-b, P4-c — el "≤0.4%" original no se sostiene en el punto outlier (0.1,5,2), ver E4/P4-a). Umbral de Guillaud–Mirrahimi κ₂/κ₁=220 ⇒ g_z/κ≥3.909 (id SIM-11).

**Baño filtrado.** κ_eff(δ)=κκ_f²/(4δ²+κ_f²), 4J²/κ_f=κ ⇒ κ₁^filt=g_x²[κ_eff(ω)/ω²+κ_eff(3ω)/(9ω²)] (id SIM-19, SIM-20). Verificado canal por canal al 0.1–0.25% (id F5a-3-m, F5a-3-p, F5a-5-m, F5a-5-p). Mejora de γ_pf respecto al baño plano, unidades de Ma, para κ_f=3,1,0.3,0.1: 19.5, 166.5, 1838, 16528 (id F5b-3.0-med, F5b-1.0-med, F5b-0.3-med, F5b-0.1-med). Umbral en g_z/κ: baja de 3.94 a 0.094 (id MU-01, MU-02, MU-03).

**Confinamiento.** Modelo mínimo con α=0 (exacto, id V6-*): Δ=(κ/4)[1−√(1−8Γ₂/κ)] si Γ₂≤κ/8, κ/4 por encima. Con |α|²=4 (tasa dinámica, id MN-01, MN-02): c=lím Δ/κ₂=4.14, saturación en ≈0.28κ.

**P9 — corrimiento del qubit dependiente de n.** χn|e⟩⟨e| con χ=(8/3)g_x²/ω frena el confinamiento. Frontera χ|α|²/κ≈0.3 (5%, id P9-01…04). Valores: Ma 2.7 (fuera), Naseem 0.19 (dentro).

**C6 — Base polarónica del espacio del código** (contenido íntegro del antiguo `C6_polaron.md`).

*Derivación.* Con el qubit en |g⟩ (σ_z|g⟩=−|g⟩), la parte del oscilador de H es ωa†a−g_z(a+a†)=ω(a−g_z/ω)†(a−g_z/ω)−g_z²/ω: el oscilador está desplazado en **d=+g_z/ω, real, en el marco de laboratorio** (con el qubit en |e⟩, −g_z/ω). En el marco rotante a ω_p/2 el desplazamiento rota como (g_z/ω)e^{iω_pt/2}; en t=nT_p vale ±g_z/ω, y como el código es simétrico, **el código es {D(g_z/ω)|±α⟩}**. El signo solo depende del signo de g_z, con σ_z=|e⟩⟨e|−|g⟩⟨g| (id SIM-28).

*Verificación numérica* (estado estacionario de Floquet, maximizando P_c sobre un desplazamiento complejo d con Nelder–Mead):

| caso | N | α | d predicho (g_z/ω) | d óptimo | d_opt/d_pred | P_c(d=0) | P_c(d=g_z/ω) | P_c(d=−g_z/ω) | P_c(d_opt) | ids |
|---|---|---|---|---|---|---|---|---|---|---|
| Ma, ω_p=11.98 | 22 | 2i | 0.03536 | 0.03451+2e-5i | 0.976 | 0.997544 | **0.998725** | 0.993890 | 0.998726 | C6-01…C6-07 |
| Ma, ω_p=11.98 | 28 | 2i | 0.03536 | 0.03451+2e-5i | 0.976 | 0.997551 | **0.998732** | — | — | C6-08, C6-09 |
| g_x=0.05,ω=6,g_z/κ=12 (T42) | 22 | 2 | 0.06000 | 0.05992 | 0.999 | 0.995044 | **0.998682** | 0.984188 | 0.998682 | C6-10…C6-12 |

Cuánto sube P_c con d=g_z/ω: Ma de 0.99754 a 0.99873 (factor 1.9, id C6-13); T42 de 0.99504 a 0.99868 (factor 3.8, id C6-14). El signo queda confirmado: con −g_z/ω, P_c baja respecto a d=0.

Desplazamiento óptimo = ⟨a⟩ estroboscópico: 0.03449 en Ma (id C6-15), un 2.4% menor que g_z/ω. La corrección de campo medio −g_z⟨σ_z⟩/ω, con ⟨σ_z⟩=−0.9947 (id C6-16), da 0.03517 (id C6-17) y explica **0.22 de la diferencia d_pred−d_opt** (id C6-18; no "un tercio" como decía el `C6_polaron.md` original — corregido aquí). La diferencia de P_c entre d=g_z/ω y d_opt es ≤1e-6, así que en todas las figuras se usa d=g_z/ω analítico.

Tamaño del gato: liberando |α| también, el óptimo da |α|²=3.98 (Ma, id C6-19) y 3.87 (T42, id C6-20); en T42, P_c sube a 0.99979 (id C6-21). Coherente con que γ_pf use |α|²=|⟨a²⟩|.

*Validaciones y convergencia.* ρ estacionarias: |Tr ρ−1|≤2.2e-16, ‖ρ−ρ†‖=0, mínimo autovalor ≥4.2e-10 — recalculado: ≥8.6e-13 (id C6-23, D: el valor declarado era más optimista que el real; la tolerancia 1e-10 sigue cumplida). Convergencia N=22→28 (Ma): P_c(g_z/ω) cambia 7e-6 (id C6-09), d_opt idéntico a 5 cifras.

---

## II.b — Métodos y parámetros

**Convenciones comunes (de `figuras_finales/README.md`).** Marco de laboratorio, muestreo estroboscópico t=nT_p (T_p=2π/ω_p, fase 0 del drive). El estado estacionario es el autovector λ=1 del propagador de Floquet de un período. Código fijo (decisión C6): P_c y todas las curvas usan {D(d)|±α_nom⟩}, d=g_z/ω analítico, α_nom²=Ω/G. α_eff²=⟨(a−d)²⟩ del estado estacionario (medido, no optimizado) se usa **solo** en la tasa de phase-flip. Regla de truncamiento con filtro (lección de P10): con |α|²=4 usar **N≥20** (N=16 da exceso espurio de γ_pf hasta ×2.9). Validaciones en toda ρ: |Tr ρ−1|<1e-10, ‖ρ−ρ†‖<1e-10, mínimo autovalor >−1e-9 (plano y efectivo); completo con filtro desde 2026-10-01: 1e-8 y −1e-8 (ver §3.1, 5 violaciones conocidas). Integrador: atol 1e-12, rtol 1e-10 (id CR-1).

**Juegos de parámetros de referencia.**
- *Ma et al.*: ω=6, g=0.3, θ=π/4 (g_x=−0.2121, g_z=0.2121), κ=0.03 (κ=2Γ), Ω=0.06, ω_q=12, γ=ω/Q con Q=1e7 (γ=6e-7, sin verificar). Unidades 2π·GHz.
- *Naseem*: ω_m/2π=100 MHz, g_z/2π=6 MHz, g_x/2π=0.6 MHz, κ/2π=100 kHz, γ_m/2π=15 Hz.
- *Liu et al.*: ν/2π=35.4 MHz (ω=ν/2), g_x=g_z=3.54 MHz, κ=16 MHz, ε_p=3.53 MHz.
- *Isla térmica* (región de referencia): κ₂/κ=0.25, g_z/κ=14, ω/κ=200, |α|²=4, κ_f/ω=0.05 (filtro) o baño plano; γ/κ=2e-5 o 2e-4.

**Unidades.** "Unidades de Ma" = ω=6, κ=0.03 (2π·GHz); el modelo completo con filtro corre así y sus tasas se dividen entre κ para expresarlas "en κ=1". El modelo efectivo corre directamente en κ=1.

**N y N_f usados.** Fig. 2(a): N=20→28 (convergencia); Fig. 2(b)/regla universal: N=14–22 según |α|²; Fig. 3: N=22 (hasta 28,32 para convergencia); Fig. 4 (filtro): N=8–28; modelo mínimo: N=24 (30,36 para C9); modelo mínimo con filtro: N=20 (26 para control); verificación central con filtro: N=16,20,22; térmico efectivo: N=22,24; térmico completo con filtro: N=18,20,22 (N=24 no cabe en 14 GB).

**Floquet y extracción de tasas.** H(t) periódico con T_p=2π/ω_p; se calcula el propagador del superoperador sobre un período (QuTiP `propagator`, atol 1e-12, rtol 1e-10) y el estado estacionario es su autovector λ=1. Tasas r_k=−ln|λ_k|/T_p. **γ_pf**: modo con λ real cerca de +1 y mayor |Tr(PR_k)|/‖R_k‖ (~1.41, id PO-1), peso de borde de Fock <0.5. **γ_bf**: modo con mayor traslape con a (modo de pozo). **η**=γ_pf/γ_bf. **Tasa dinámica de confinamiento** (C9, en vez de la brecha espectral, espuria en régimen saturado): retorno de P_c(t) desde |0⟩|g⟩ y D(d)|1.3α⟩|g⟩, ajuste de la cola exponencial de |P_c(t)−P_c(∞)| por encima del piso de paridad/ruido numérico.

**Modos espurios.** Peso de borde >0.5 o autovalor inestable al 0.3% en tres truncamientos N, N+6, N+12.

---

## II.c — Térmico completo

**Definiciones.** η=γ_pf/γ_bf; x=hf_q/k_BT con f_q la frecuencia del qubit; n_q=1/(eˣ−1), n_m=1/(e^{x/2}−1); k_BT/h=208.37 MHz a 10 mK.

**Convergencia en N con filtro** (κ₂/κ=0.25, x=6.86), memoria y tiempos reales (`/usr/bin/time`):

| N | γ_pf (κ=1) | γ_bf (κ=1) | Memoria máx. | Tiempo real | id |
|---|---|---|---|---|---|
| 18 | 1.65382e-04 | 4.64921e-05 | 7.51 GB | 2116.63 s (35.3 min) | CN-18-pf, CN-18-bf, RC-N18, RC-t18 |
| 20 | 1.65261e-04 | 3.38515e-05 | 11.36 GB | 3515.15 s (58.6 min) | CN-20-pf, CN-20-bf, RC-N20, RC-t20 |
| 22 | 1.65256e-04 | 3.34895e-05 | 14.0 GB (toda la RAM; en tres corridas: 14015972, 14044332, 14017232 kB) | 5553.24, 5460.22, 5608.04 s | CN-22-pf, CN-22-bf, RC-N22, RC-t22 |

N=18 no converge (γ_bf +38.8% respecto de N=22). N=20→22: γ_pf −3.2e-5 relativo, γ_bf −1.07%. Residual geométrico estimado de N=22 en γ_bf: −0.03% (peor caso −1.08%, id CN-q, CN-resid, CN-worst). N=24 no cabe (N=22 ya usa 14.0 GB).

**Preregistro κ_f/ω=0.2 y su resultado** (contenido íntegro de `preregistro_kf0.2.md`, escrito antes de lanzar las corridas, commit `cfa8f01`).

*Corridas.* Modelo completo con filtro, N=20, N_f=2, (κ₂/κ,x)=(0.25,6.86): g_x=0.0535714, ω=6, g_z=0.42, κ=0.03, **κ_f=1.2 (κ_f/ω=0.2)**, |α|²=4, γ=6e-7. Pareja térmica (x=6.86) y T=0.

*Predicciones del efectivo* (κ=1; N=22, N_f=2, κ_f/ω=0.2 — no cacheadas, id PR-2 R): γ_pf(T=0)=1.66411e-4, γ_pf(térmico)=1.78395e-4, Δ_e=1.1984e-5, γ_pf(térmico, ansatz n_q)=1.85736e-4, **E_ansatz=+7.341e-6** (+4.12%), γ_bf térmico/T=0=3.4148e-5/8.1797e-9, E con ocupación física (referencia)=+5.607e-5.

*Estadístico y regla (fijados antes, id PR-3):* E_obs=(r_x−r_0)·γ_e(x), r_0=γ_pf(completo,T=0)/γ_pf(efectivo,x=60), r_x=idem a x. Regla: E_obs>3e-6 ⇒ canal de tamaño ansatz; |E_obs|<1e-6 ⇒ no hay canal; entre 1e-6 y 3e-6 ⇒ indeterminado. Esperado con canal: E_obs≈+7.3e-6; sin canal: E_obs≈0.

*Comprobaciones previas* (id PR-4, PR-5, PR-6, PR-7): N_f=2 basta (γ_pf cambia 2.7e-6 entre N_f=2,3,4); 4J²/κ_f=κ se mantiene (J=0.09487); γ≪κ (γ/κ=2e-5, γ/κ_f=5e-7); ke(ω)=9.9e-3 (16× el de κ_f/ω=0.05); el residual de N=20 en γ_pf (~5e-9κ) es mucho menor que los umbrales de la regla.

*Resultado* (commit `ae2cc52`, id K2-*): completo κ=1: γ_pf T=0 1.58888e-04, térmico 1.76739e-04; γ_bf T=0 3.449e-07, térmico 2.911e-05; η térmico 6.072, η(T=0) 460.6. r_0=0.95479, r_x=0.99072. Δ_c=1.7850e-05, D=Δ_c−Δ_e=+5.866e-06. **E_obs=+6.408e-06, E_obs/E_ansatz=0.873 ⇒ regla cumplida: canal de tamaño ansatz.** α_eff² medido: 3.79744 (T=0), 3.79408 (térmico). Hermiticidad cruda 1.55e-9 (T=0), 7.5e-11 (térmico); |Tr−1|≤2e-16; mínimo autovalor −1.0e-14 y +3.3e-13. Tiempo real 3529.95 s y 3488.54 s, memoria 11.36 GB.

**Dependencia del modelo de baño (n_q frente a n_m).** Supuesto base: canales de un fonón mediados por el qubit (Γ₁±) y el baño del qubit/filtro usan n_q; la pérdida intrínseca γ usa n_m. Control con ocupación real n(ω),n(3ω) en Γ₁± (id OC-1): con filtro, γ_pf cambia ≤1e-4; en el plano, hasta +7.8% (x=6). Barrido de la densidad espectral del baño s=J(ω)/J(2ω) (id BS-1, BS-tab): γ_bf plano/filtrado cruza 1 en s=0.132 (variante A) y s=0.209 (B); con ruido blanco (s=1) el completo da 0.857. **No validado fuera del ruido blanco.**

**Ansatz del desplazamiento dependiente del estado** (canales σ∓a, σ∓a† con β=2g_z/ω=0.14). En el plano reproduce el 98% del aumento térmico de γ_pf vía el canal n_q, con 16% de exceso sobre el completo (id PL7-kick, PL7-16, PL7-nq, PL7-half). Con filtro (diagnóstico, factor ke(ω)=6.2e-4): efecto despreciable en γ_bf (≤1.1e-3 relativo, id FK-1…FK-6); no acerca γ_bf del efectivo al completo.

**Diagnóstico de α_eff² entre T=0 y térmico.** Cambio relativo de α_eff² similar en completo y efectivo (−3.2e-4 a −1.2e-3 en x=6.86, id AD-*); no explica E_obs (corrección ≤5% de E_obs).

**Sensibilidad a N_f** (efectivo, κ_f/ω=0.05, x=6.86, id NF-1, NF-2): N_f=2→3 mueve γ_bf −2.1% a −3.6%; si solo el efectivo cambiara, el cociente completo/efectivo pasaría de (0.9561,0.8563,0.7848,0.7759) a (0.9764,0.8807,0.8132,0.8045) — ninguno cruza ±5% salvo acercarse en κ₂/κ=0.05. El completo no se corrió con N_f=3 (pendiente, Parte I §4).

---

## II.d — Figuras

Convenciones comunes en Parte II.b. Validaciones declaradas por figura (|Tr ρ−1|, ‖ρ−ρ†‖, mínimo autovalor) en `afirmaciones.csv` (ids Q-f2, Q-f2b, Q-f3, Q-f4, Q-t1…Q-t4); **5 violaciones del criterio −1e-9 en todo `data/`**, tabla completa en Parte I §3.1.

**Fig. 2 — Resonancia vestida** (`fig2.py`, `calc_fig2.py`, `run_fig2.sh`). (a) H completo de Ma, ω=6, g=0.3, θ=π/4, κ=0.03, N=22: P_c con código fijo vs. ω_p∈[11.955,12.015]; línea punteada ω_p*=11.980, trazo-punto 2ω. (b) ω_q=ω_p, |α|²=4, κ₂/κ∈{0.03,0.1,0.3,1,2}: FWHM por interpolación PCHIP. Ajuste por el origen FWHM=cG → ver PD-c (Parte I §3.2). Convergencia ω_p=11.98, N=20→26: ΔP_c=1.4e-4 (id CV-f2a…CV-f2d).

**Fig. 3 — Figura de mérito** (`fig3.py`, `calc_fig3.py`, `run_fig3.sh`). H completo, ω_q=ω_p=ω_p*, κ=0.03. (a) 15 puntos con |α|²_nom=4, N=22: κ₁/κ₂ medido/predicho (Tabla V4, ids V4-01-k2…V4-15-con1, por fila y columna). (b) punto (0.05,6,4) con |α|²_nom=2,4,6: γ_pf medido/predicho con y sin +1 (id V4b-*). Convergencia (0.05,6,12) N=22→28: γ_pf 1.6e-5 relativo (id CV-f3, CV-f3b).

**Figuras principales 2 y 2 (nueva, regla universal)** (`principal_fig2.py`, `calc_principal_fig2.py`, `calc_principal_fig2d.py`). (a)-(c): ρ de Fig. 2(a); Wigner y P_c(ω_p). (d): gato par transitorio, F=0.741 en Γt=16.9 (id P3-01…P3-05). Regla universal: 157 estados estacionarios, pico en x≈0 (id RU-*), FWHM 2.5–3.25 (|α|²=4) a 3.7–4.2 (|α|²=2) (id RU-05…RU-08). P2: 49 puntos con N=16 recalculados con N=22 en |x|≤1 (id RM-48, nota: el reporte original decía 48).

**Fig. 4 — Baño filtrado** (`fig4.py`, `calc_fig4.py`, `run_fig4.sh`). (a) estático sin drive: Γ₁± vs. κ_eff, acuerdo 0.1–0.25% (id F5a-*). (b) con drive: mejora de γ_pf (id F5b-*); confinamiento relativo al plano 0.965 (κ_f=3) a 0.53 (κ_f=0.1) por tasa dinámica (id F4c-*).

**Figura principal 3 — espacio de diseño** (`principal_fig3.py`, `calc_minimo.py`, `calc_pfig3.py`). Mapa ε=κ₁/κ₂^eff(g_z/κ,κ₂/κ); κ₂^eff=Δ/c, c=4.14 (id MN-01). Curvas χ|α|²/κ=0.3 (id P9-*). 15 puntos del modelo completo, ε_completo/ε_mapa 0.97–1.55 (id MP-01, MP-02).

**Figura central** (`codigo/figura_central.py`, `calc_minimo_filtro.py`, `calc_filtro_completo.py`). (a) baño plano, (b) filtro κ_f/ω=0.05 con costo en confinamiento incluido, (c) colapso 19 puntos. Umbral g_z/κ: 3.94→0.094 (id MU-*). Verificación con filtro N=16/20/22 (id FC-*, tabla en Parte I §2). Piso intrínseco γ/κ=2e-5,2e-4 (id PI-*, PS-*). **Dos versiones**: `figura_central` (color con piso γ/κ=2e-5) y `figura_central_sinpiso` (recomendada, PD-central).

**Figura térmica** (`codigo/figura_termica.py`, `calc_termico.py`, `run_termico.py`, `ajuste_cbf.py`). (a) η(x) en la isla; (b)-(c) mapas η(γ/κ,x) y η(κ₂/κ,x); (d) γ_bf/κ vs. n_q. Umbrales x*(η=100,220) en la tabla de Parte I §2. c_bf(κ₂/κ)∝(G/κ)^s, s=3.96±0.04 (filtro) y 3.69±0.05 (plano) (id CB-s0, CB-s1). Validación con el modelo completo: tabla de Parte I §2 (T84-*). Detalle de convergencia y bano en Parte II.c.

---

## II.e — Pendientes y hallazgos (de `PENDIENTES_Y_HALLAZGOS.md`)

### Pendientes de decisión o revisión (P1–P10)
| # | Tema | Estado actual |
|---|---|---|
| P1 | Regla universal: colapso parcial (no total; FWHM depende de |α|²) | Aceptado, figura terminada |
| P2 | Curvas con N=16 subestimaban P_max | Resuelto: recalculadas con N=22 (id RM-48) |
| P3 | Panel (d), gato transitorio | En apéndice (id P3-*) |
| P4 | Déficit no adiabático de C1/C2 | Abierto, sin explicación analítica (id P4-a,b,c) |
| P5 | Desplazamiento polarónico óptimo en Ma, 2.4% menor que g_z/ω | Campo medio explica 0.22, no un tercio (id C6-18, corregido §3.3) |
| P6 | Exceso +1.6% en V4 a g_z/κ pequeño, explicado por el +1 | Menor, punto (0.1,5,2) con déficit residual 0.7% |
| P7 | Figura térmica | Validación parcial (±5% solo en κ₂/κ=0.05); ver Parte II.c |
| P8 | Apagado automático del PC sin permiso | Acción del usuario, no aplica a datos |
| P9 | Corrimiento del qubit dependiente de n | Resuelto: χ=(8/3)g_x²/ω, frontera 0.3 (id P9-*) |
| P10 | Exceso de κ₁ con filtro en régimen saturado | Resuelto: artefacto de N=16 (id FC-0.03-16-pf, P10-*) |

### Errores de la literatura (E1–E8)
| # | Qué | Consecuencia |
|---|---|---|
| E1 | Ma, Ec.(4): omite −(g_x²/ω)n, da 3g_x²/ω para el qubit | Resonancia en 11.98, no 12 (id ED-*, MA-03) — **ver literatura, Parte II.g** |
| E2 | Faltaba el +1 del canal de ganancia en γ_pf | +1.6% en V4 (id V4b-rango) |
| E3 | Acuerdo del 0.05% en V3 era compensación casual | Razón corregida 0.982 (id V3-07) |
| E4 | "≤0.4%" de C3 solo vale para κ₂/κ≲0.05 | Reemplazado por precisión continua (id P4-a,b,c) |
| E5 | "5.º modo espectral" (Tarea 43) es artefacto de truncamiento | Se usa la tasa dinámica (id F4m-*, F4c-*) |
| E6 | Código con α_eff no sirve para P_c | Código fijo D(g_z/ω)|±α_nom⟩ (id S-01) |
| E7 | Hermiticidad de la Tarea 37 se medía tras simetrizar por paridad | La cruda del térmico nuevo ≤1.40e-11 (id Q-t1…Q-t4) |
| E8 | Tolerancias de traza en T42/T43 excedidas por potencias del propagador | No aplica a los cálculos actuales |

### Correcciones adoptadas (C1–C9)
C1: γ_pf con +1 y |α|²=|α_eff²|. C2: κ₁/κ₂=(5/72)(κ/g_z)², κ₁=Γ₁⁻+Γ₁⁺. C3 (revisado): precisión continua en κ₂/κ. C6: código fijo polarónico, marco de laboratorio t=nT_p, desplazamiento +g_z/ω. C9: confinamiento dinámico en vez de la brecha espectral. Interpolación: PCHIP para los FWHM.

### Criterio del piso de T=0 (1b)
η_max=η(T=0) del completo va de 3.1e4 (κ₂/κ=0.05) a 2.35e3 (0.4). Propuesta de sustituir el criterio por η_max≥10×nivel de sesgo: **no adoptada (PD-piso A, Parte I §3.2)**.

### Limitaciones de los cálculos propios
Convergencia con filtro ahora cubre N=16→22 (antes solo 18); la regla universal usa N=14–22; Fig. 2(a) de validación con N=22 (N=20 subestimaba P_c en 1.4e-4).

---

## II.f — Plataformas y tareas históricas

### Tabla de plataformas
| Plataforma | κ/ω | g_z/κ | κ₂/κ | κ₁/κ₂ | notas | ids |
|---|---|---|---|---|---|---|
| Ma | 0.005 | 7.1 | 1.0 | 1.4e-3 | χ\|α\|²/κ=2.7 (fuera de frontera); γ/κ=2e-5 (Q=1e7, sin verificar) | PL-01…PL-05, PL-14 |
| Naseem | 0.001 | 60 | 2.07 | 9.2e-5 (con γ) / 1.9e-5 (nominal) | x=0.96 a 10 mK; térmico no calculado (PD-naseem) | PL-06…PL-10, PL-15 |
| Liu | 0.90 | 0.22 | 0.031 | — | fuera de validez (κ≪ω no se cumple); el gato no se forma | PL-L |

### Tabla de tareas históricas (23 vigentes, 12 parciales, 9 superadas, 5 contradichas, 1 incompleta)
Tabla completa en `consolidacion/tareas_historicas.csv` (50 filas: Tareas 1–44 de Naseem, 39–47b de Ma, y 6 resúmenes de ronda/documentos). Muestra:

| Tarea | Serie | Resultado clave | Estado |
|---|---|---|---|
| 7, 8 | Naseem | g_eff=2g confirmado (Rabi de dos fotones <5%; álgebra polarónica) | vigente |
| 12 | Naseem | Estacionario ≈ estado oscuro (F~0.97) | **contradicha** por Tarea 18 (paridad verdadera 0.052) |
| 18 | Naseem | Floquet exacto: modo de phase-flip ultra lento | vigente (método adoptado en todo el proyecto) |
| 21 | Naseem | Brecha del efectivo diverge del completo (53.5×) | **contradicha** por Tarea 38/HANDOFF 11 (artefacto de borde de Fock) |
| 23 | Naseem | Γ₂/κ óptimo≈0.126 | **contradicha** (maximo = artefacto, Tareas 35,38,45) |
| 38 | Naseem | Brecha física monótona, satura en ~0.23κ | parcial (rama interior no determinada; superado por tasa dinámica, satura en ~0.28κ con \|α\|²=4) |
| 42 | Ma | κ₁/κ₂ del completo re-sintonizado; brecha por 5.º modo | parcial (κ₁/κ₂ vigente; brecha por 5.º modo superada por E5) |
| 44 | Naseem | Supresión de bit-flip con \|α\|² (Gautier): pendiente −0.55 | vigente — **simulación propia del modelo de buffer de Gautier et al.**, no de Naseem/Ma (id AL-0-9) |
| 45b-e | Ma | N_f=2 basta; ancho ~κ₂^0.54; brecha a Γ₂/κ>~0.25 no determinada | vigente |
| 47 | Ma | FWHM=cG, c=2.87±0.25 (7 puntos) | vigente — verificación cruzada de PD-c |
| verificacion_signo_pair | Ma | Signo de pares en `ma2.py`: correcto, nunca se commiteó mal | vigente |

(Resto de las 50 filas, incluidas las de 29bis-incompleta y los 6 resúmenes de ronda, en el CSV.)

---

## II.g — Afirmaciones sobre la literatura: fuera de alcance, sin verificar

Copiadas tal cual de los documentos originales; **no se verificó el contenido de los artículos citados**, solo las simulaciones propias que las acompañan (ids en `afirmaciones.csv`, tipo `literatura`, estado `L`):

- **LIT-1.** Ma et al., Ec.(4): omite −(g_x²/ω)n, da 3g_x²/ω para el corrimiento del qubit. El término de pares es correcto.
- **LIT-2.** Liu et al., Ec.(11): omite −(4g_x²/3ω)n; posible inconsistencia del factor del drive (ε_p vs. 2ε_p).
- **LIT-3.** Naseem 2026: su código no incluye δ₁; su Ec.(11) usa g_eff=g en vez de 2g; su receta de aumentar κ contradice κ₁/κ₂∝κ². (Parte verificable con datos propios: g_eff=2g, Tareas 7–8 históricas, y κ₁/κ₂∝κ² id SIM-10, ambas V.)
- **LIT-4.** Ma et al., Ec.(5): su modelo efectivo conserva la paridad; Liu y Hou no se comparan en este punto.
- **LIT-5.** Hou 2024: pendiente de lectura; no está en `papers/` (2407.17299 es Dubovitskii, no Hou — hecho de archivo verificado).
- **LIT-6.** Liu et al.: la Ec.(9) completa no forma el gato (F≤0.35 frente a 0.95 del efectivo) — simulación propia (Tareas 46/47b, históricas); la afirmación sobre lo que Liu dice en su artículo queda fuera de alcance.
- **LIT-7.** Naseem: g_eff=2g confirmado por dos métodos independientes (Tareas 7–8).
