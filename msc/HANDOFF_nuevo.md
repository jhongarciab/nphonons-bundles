# HANDOFF — paper de generalización: qubits gato estabilizados por un qubit auxiliar (g_x, g_z)

Documento único de traspaso. Reemplaza todos los demás `.md` de `msc/` (archivados en `msc/_archivo/` tras aprobación). Verificado contra datos en `msc/consolidacion/` (ver `afirmaciones.csv`, `por_decidir.csv`, `tareas_historicas.csv`, `clasificacion.csv`). Tag `pre-consolidacion` = commit `1eae004`. Idioma: español.

---

## 1. Mapa de archivos importantes

### Núcleo de registro (7 archivos, 0.2 MB)
| Archivo | Qué es |
|---|---|
| `HANDOFF.md` (este archivo) | Traspaso único |
| `README.md` | Mapa de carpetas de `msc/` |
| `Mechanical-Cat-State/figuras_finales/README.md` | Métodos, parámetros y validaciones de cada figura |
| `Mechanical-Cat-State/figuras_finales/PENDIENTES_Y_HALLAZGOS.md` | Pendientes P1–P10, errores de la literatura E1–E8 |
| `Mechanical-Cat-State/figuras_finales/C6_polaron.md` | Derivación y verificación de la base polarónica |
| `docs/Informe/informe.tex` | Borrador del paper |
| `docs/Informe/refs_informe.bib` | Bibliografía del informe |

### Entregables (29 archivos, 4.2 MB)
| Ruta | Alimenta |
|---|---|
| `docs/Informe/figuras/fig_resonancia.pdf` | Fig. resonancia vestida del informe |
| `docs/Informe/figuras/fig_merito.pdf` | Fig. figura de mérito del informe |
| `docs/Informe/figuras/fig_filtro.pdf` | Fig. baño filtrado del informe |
| `docs/Informe/figuras/fig_chi.pdf` | Fig. frontera χ del informe |
| `docs/Informe/figuras/fig_mapa.pdf` | Fig. mapa de diseño del informe |
| `docs/Informe/figuras/figura_termica.pdf` | Fig. térmica del informe (copia de figuras_finales) |
| `docs/Informe/informe.pdf` | PDF compilado del informe |
| `Mechanical-Cat-State/figuras_finales/figuras/principal_fig2.{pdf,png}` | Figura principal 2 (resonancia vestida con estados) |
| `Mechanical-Cat-State/figuras_finales/figuras/principal_fig3.{pdf,png}` | Figura principal 3 (mapa de diseño) |
| `Mechanical-Cat-State/figuras_finales/figuras/figura_central{,_sinpiso}.{pdf,png}` | Mapa de diseño con baño plano/filtrado y piso |
| `Mechanical-Cat-State/figuras_finales/figuras/figura_termica.{pdf,png}` | Figura térmica (η vs x) |
| `Mechanical-Cat-State/figuras_finales/figuras/fig2.{pdf,png}` … `fig4.{pdf,png}` | Figuras de validación (apéndice) |
| `Mechanical-Cat-State/figuras_finales/figuras/apendice_*.{pdf,png}` | Apéndice (Wigner, colapso de resonancia) |
| `Mechanical-Cat-State/figuras_finales/figuras/p9_diagnostico.{pdf,png}` | Diagnóstico de P9 (corrimiento χ) |

### Cadena de dependencia (7072 archivos, 64.6 MB)
Todo `Mechanical-Cat-State/figuras_finales/codigo/` (scripts `calc_*`, `run_*`, `*fig*`, `comun.py`, `estilo.py`) y `Mechanical-Cat-State/figuras_finales/data/` (cachés `.npz` con ρ, CSV de cada figura) — es el código oficial del paper (README raíz). Sin él no se reproduce ningún entregable. Se detalla por bloque en §2; el inventario completo está en `consolidacion/clasificacion.csv` (categoría `cadena`).

### Histórico (2339 archivos, 14.6 MB)
No alimenta ningún entregable actual. Se archivará (git mv) solo tras tu aprobación, sin borrar nada:
- `Mechanical-Cat-State/validacion/tareas_naseem/` (1765 archivos: Tareas 1–44, esquema Naseem) — ver tabla de tareas históricas, §4.
- `Mechanical-Cat-State/validacion/tareas_ma/` (398: Tareas 39–47, esquema Ma)
- `Mechanical-Cat-State/validacion/independiente/` (102: verificación V1–V6 — conclusiones ya incorporadas al núcleo)
- `Mechanical-Cat-State/codigo_original/` (41: repo original del artículo, sin modificar)
- `docs/` salvo `Informe/` (21: borradores previos — `documento_completo`, `brecha_punto_excepcional`, `figura_merito`, `lamb_shift_ma2019`, `mapa_plataformas`, `handoff_chat`; todos compilan, todos superados por este HANDOFF, ver §3)
- `papers/` (2 PDF de referencia en arXiv)

---

## 2. Estado y resultados clave

Marca: **V** recalculado y coincide · **R** reportado en log, no recomputable sin simulación nueva · **H** hipótesis declarada · **S** superado por resultado posterior. Cada fila cita el id de `afirmaciones.csv` para el detalle (tolerancia exacta, archivo de dato, script).

### Modelo y derivaciones (simbólico, sympy)
| Resultado | Valor | Marca | id |
|---|---|---|---|
| H_ef (James–Jerke) | (g_x²/ω)[(4n/3+1)|e⟩⟨e|−(4n/3+1/3)|g⟩⟨g|] − G(σ₊a²+h.c.) | V | SIM-01 |
| G, κ₂, Γ₁± | G=2g_xg_z/ω; κ₂=4G²/κ; Γ₁⁻,Γ₁⁺ (Reiter–Sørensen) | V | SIM-02,08,09 |
| ω_p* (resonancia vestida) | 2(ω−4g_x²/3ω) | V | SIM-07 |
| γ_pf (phase-flip, con +1) | 2[Γ₁⁻|α|²+Γ₁⁺(|α|²+1)] | V | SIM-14 |
| κ₁/κ₂ (figura de mérito) | (5/72)(κ/g_z)² | V | SIM-10 |
| Umbral κ₂/κ₁=220 ⇒ g_z/κ | ≥3.909 | V | SIM-11 |
| Baño filtrado κ_eff(δ) | κκ_f²/(4δ²+κ_f²) | V | SIM-19 |
| Brecha exacta (α=0) | (κ/4)[1−√(1−8Γ₂/κ)] / κ/4 | V | SIM-24,26 |

### Efectivo vs. modelo completo (verificación independiente V1–V6)
| Resultado | Valor medido | Tol. | Marca | id |
|---|---|---|---|---|
| H_ef por diagonalización exacta | error ~g⁴ | — | V | ED-* |
| Resonancia de Ma (ω_p=11.98) | P_c=0.99873 (fijo), 0.99754 (sin desplazar) | ≤2e-5 | V | MA-12,13 |
| Máximo del barrido fino | 11.9799±0.0005 vs. predicho 11.9800 | 0.0005 | V | V2-01 |
| γ_pf con +1 (|α|²=2,4,6) | medido/predicho 0.9988, 0.9972, 0.9961 | 0.004 | V | V4b-* |
| κ₁/κ₂ (19 puntos del completo) | 0.979–0.999 | — | V | V4-RANGO |
| Baño filtrado, canal por canal | 0.1–0.25% | 0.3% | V | F5a-* |
| Brecha exacta (18 Γ₂/κ) | ≤2e-13 salvo punto excepcional | 3e-13 | V | V6-* |

### Confinamiento y mapa de diseño
| Resultado | Valor | Marca | id |
|---|---|---|---|
| c = lím Δ/κ₂ (|α|²=4) | 4.14 | V | MN-01 |
| Saturación de la tasa dinámica | ≈0.28κ (no κ/4) | V | MN-02 |
| Frontera χ\|α\|²/κ≈0.3 (5% de pérdida) | V | P9-01..04 |
| Umbral g_z/κ: plano 3.94 → filtro 0.094 | factor 0.0238 | V | MU-01..03 |
| Costo del filtro en saturación (κ₂/κ=1) | 0.957 (N=26) — **no** 0.480 (N=20, no convergido) | V/D | MF-SAT (ver PD-sat §3) |

### Modelo completo con filtro, N=16/20/22
| Punto | γ_pf/predicho | ε_completo/ε_mapa | Marca | id |
|---|---|---|---|---|
| (κ₂/κ,g_z/κ)=(0.03,12), N=22 | 0.978 | 0.979 | V | FC-0.03-22-* |
| (0.3,12), N=22 | 0.977 | 0.972 | V | FC-0.3-22-* |
| N=16 (ambos) | ×2.26–2.88 | artefacto de truncamiento | S | FC-0.03-16-pf |

### Térmico (isla κ₂/κ=0.25, g_z/κ=14, ω/κ=200, |α|²=4)
| Cociente completo/efectivo | κ₂/κ=0.05 | 0.1 | 0.25 | 0.4 | Marca | id |
|---|---|---|---|---|---|---|
| γ_pf (x=6.86) | 0.959 | 0.959 | 0.959 | 0.959 | V | T84a-* |
| γ_bf crudo (x=6.86) | 0.956 | 0.856 | 0.785 | 0.776 | V | T84b-*-rbf |
| Criterio ±5% en γ_bf | cumple solo en 0.05 | — | — | — | V | T84-c2 |
| x*(η=100) efectivo vs completo [ESTIMADO] | 6.86/6.86 | 8.36/8.25 | 10.11/9.93 | 10.84/10.67 | V | T85-* |
| Origen de la discrepancia γ_bf(κ₂/κ) | no identificado | — | — | — | H | HANDOFF 8.4 §4 pendiente |
| Canal n_q (84%): desplazamiento reproduce el canal | 0.86× (completo/kick) | — | — | — | D (redacción ambigua, no error numérico) | PL7-84 |
| Prueba κ_f/ω=0.2: canal de tamaño ansatz | E_obs=+6.41e-6, regla cumplida | — | — | — | V | K2-Eobs |

### Canal polarónico (C6)
| Resultado | Valor | Marca | id |
|---|---|---|---|
| Desplazamiento d=+g_z/ω en |g⟩ | verificado (P_c sube ×1.9 a ×3.8 al desplazar) | V | C6-04..14 |
| d_opt/d_pred (Ma) | 0.976 | V | C6-03 |
| Campo medio explica parte de d_pred−d_opt | 0.22 (no "un tercio") | D | C6-18 |

**Resumen por nivel:** nivel 1 (verificación línea por línea): 914 V, 26 D, 29 S, 22 R, 17 H, 8 L (literatura, fuera de alcance). Nivel 2 (docs/*.tex): 31 V, 8 D, 12 S. Nivel 3 (histórico, tabla por tarea): 23 vigentes, 12 parciales, 9 superadas, 5 contradichas, 1 incompleta — ver `tareas_historicas.csv`.

---

## 3. Discrepancias corregidas y decisiones de Jhon

Las 8 de `por_decidir.csv`, con la opción elegida:

1. **PD-umbral (elegida: A).** Los umbrales de hermiticidad/positividad del filtro completo se relajaron (1e-10/−1e-9 → 1e-8/−1e-8) **después** de observar violaciones en los datos (commit del código `efda948`, 2026-09-30 23:17 −0500 = 10-01 UTC; datos que violan desde `2652746`, `4d23ea4`, `970edfa`, todos del 09-29/09-30). **Se declara así**: el cambio de criterio fue posterior a ver los datos, no una elección a priori.

   **Tabla única de mínimo autovalor por grupo, con las 5 violaciones de −1e-9 en `data/`:**
   | Archivo | Grupo | Mín. autovalor | id |
   |---|---|---|---|
   | `filtro_completo/gx0.0216506_..._N22_Nf2.npz` | verificación central | −5.58e-9 | Q-a1 |
   | `filtro_completo/gx0.0239579_..._gam6e-07.npz` | térmico (κ₂/κ=0.05, T=0) | −2.49e-9 | Q-b7 |
   | `filtro_completo/gx0.0338816_..._gam6e-07.npz` | térmico (κ₂/κ=0.1, T=0) | −1.42e-9 | Q-b8 |
   | `filtro_completo/gx0.0684653_..._N22_Nf2.npz` | verificación central | −1.31e-9 | Q-a2 (nota) |
   | (una quinta, −1.12e-9) | verificación central | −1.12e-9 | Q-a2 (nota) |

2. **PD-piso (elegida: A).** Mantener el criterio original γ_bf(T=0)<1% del térmico como criterio de aceptación (falla a x≈9 y en la isla a x=10.1: 1.35%–2.52%). No se adopta la propuesta de sustituirlo por η_max≥10×nivel de sesgo.

3. **PD-c (elegida).** Citar **c=2.94±0.27** (ajuste por el origen, serie ω_q=ω_p, recalculado 2.936; media 2.916±0.27 sobre 5 puntos) para la serie con el qubit siguiendo al drive, y **c=2.87±0.25** (media sobre 7 puntos, Tarea 47d) para la serie con ω_q fijo. Se retira el "2.93" de HANDOFF/README anterior, que no reproducía ninguna de las dos series tal cual. id: FW-06..FW-10.

4. **PD-sat (elegida).** Se retira "Δ_filtro/Δ_plano=0.480" en κ₂/κ=1 de la tabla de costo del filtro: el punto no está convergido en N (N=20,N_f=2 → 0.480; N=26,N_f=3 → 0.957; el P_c estacionario del modelo mínimo con filtro pasa de 0.99934 a 0.99999993 entre esos N). Se reporta **0.957 (N=26), no comprobado a N=30**. id: MF-SAT.

5. **PD-9.2 (elegida: A).** El 9.2e-5 de κ₁/κ₂ de Naseem en la tabla de plataformas se rotula **"con pérdida intrínseca incluida" (κ₁+γ)/κ₂**, y se agrega 1.9e-5 como valor nominal (5/72)(κ/g_z)² junto a él. id: PL-08,09.

6. **PD-eta (elegida: A).** η=100 y η=220 se mantienen como "valores de referencia" sin justificación propia derivada en este proyecto (η=100: convención de trabajo heredada de Naseem; 220: umbral de Guillaud–Mirrahimi sobre κ₂/κ₁, no sobre γ_pf/γ_bf).

7. **PD-central (elegida: sinpiso).** Se recomienda `figura_central_sinpiso` para el paper (no ata el mapa a un γ/κ de plataforma); `figura_central` queda como alternativa con el piso γ/κ=2e-5 incluido en el color.

8. **PD-naseem (elegida: no lanzar).** El térmico de Naseem (x≈0.96, κ₂/κ=2.07, fuera del rango validado x≥6.86) no se lanza. Queda como "no calculado" en la tabla de plataformas.

**Otras correcciones de redacción (no numéricas):**
- El "84%" de HANDOFF §8.7 (desplazamiento dependiente del estado reproduce el canal n_q del plano) es ambiguo: el cociente completo/efectivo-con-kick es 0.860 y el inverso 1.163; ninguno es literalmente "84%" salvo como 1−0.16 (el 16% de exceso ya reportado). Se mantiene la cifra del exceso (16%) como la correcta y se retira "84%" como redundante.
- El −0.55 de la Tarea 44 (supresión de γ_bf con |α|², régimen de Gautier) se rotula explícitamente como **simulación propia del modelo de buffer de dos niveles de Gautier et al.**, no una medida del esquema de Naseem o Ma. id: AL-0-9 (tareas_historicas: Tarea 44).

---

## 4. Limitaciones y pendientes abiertas

- **Un solo juego de parámetros** valida el modelo completo térmico (g_z/κ=14, ω/κ=200, |α|²=4, γ/κ=2e-5, κ_f/ω=0.05); no está validado γ/κ=2e-4 ni otros κ_f ni otros g_z/κ.
- **Origen de la discrepancia de γ_bf con κ₂/κ** (0.956 a 0.776 crudo, x=6.86): abierto.
- **Origen del piso de T=0 con filtro** (crece con κ₂/κ en el completo, no en el efectivo): abierto.
- **Modo vecino de γ_pf con filtro**: solo observado, sin comprobar que sea Y_L.
- **Origen del residuo de hermiticidad** (diagonalización densa, ruido ~1e-9 a 1e-12): no verificado.
- **N_f del completo**: solo N_f=2; el efectivo con N_f=3 mueve γ_bf −2% a −3.6%; falta correr el completo con N_f=3.
- **Validación del baño plano** y del supuesto de ruido blanco (n(ω), n(3ω)): no validado, solo diagnósticos.
- **Desplazamiento con filtro, parte no secular**: requiere Floquet con tiempo, no hecho.
- **Naseem térmico** (κ₂/κ=2.07): no calculado (PD-naseem).
- **Hou 2024**: no leído; no está en `papers/` (2407.17299 es Dubovitskii, no Hou).
- **Revisión humana independiente** de las derivaciones y política de la APS sobre IA: pendiente.
- 117 líneas numéricas de nivel 1 (pies de figura, encabezados de tabla, descripciones de método) no fueron verificadas línea por línea; lista en `consolidacion/lineas_sin_verificar.csv`.

---

## 5. Cómo reproducir

El entorno `.venv_qutip5` **no existe** en esta copia (está en `.gitignore`). Recrearlo:
```bash
cd Mechanical-Cat-State
python3.11 -m venv .venv_qutip5
.venv_qutip5/bin/pip install qutip==5.3.1 numpy scipy matplotlib
```
(Versión tomada de los logs de `figuras_finales/data/`; QuTiP 4.7.6 solo se necesita para `validacion/tareas_naseem/.venv`, ahora histórico.)

Reproducir una figura desde caché (sin recalcular):
```bash
cd Mechanical-Cat-State/figuras_finales
../.venv_qutip5/bin/python codigo/<script>.py
```
Forzar un punto nuevo: `codigo/calc_*.py ... --rerun`. Barridos masivos: `codigo/run_*.sh`.

Reproducir la verificación de este HANDOFF (sin QuTiP, solo numpy/scipy/sympy):
```bash
cd msc
python3 consolidacion/verificar_todo.py   # regenera consolidacion/afirmaciones.csv
```

Compilar el informe:
```bash
cd docs/Informe && pdflatex informe && bibtex informe && pdflatex informe && pdflatex informe
```
