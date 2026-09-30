# Figuras finales (PRA)

Entorno: `../.venv_qutip5` (Python 3, QuTiP 5.3.1). Todo se ejecuta **desde esta carpeta** (`figuras_finales/`).

## Estructura
```
figuras_finales/
├── README.md                    este archivo (métodos, parámetros, validaciones por figura)
├── PENDIENTES_Y_HALLAZGOS.md    pendientes, discrepancias con el trabajo original, correcciones adoptadas
├── C6_polaron.md                derivación y verificación de la base polarónica
├── figuras/                     PDF (vectorial) y PNG (300 dpi) de cada figura
├── codigo/                      scripts
│   ├── comun.py, estilo.py      modelo/utilidades y estilo común (serif 8.5 pt, Okabe–Ito, 3.375 in)
│   ├── calc_*.py, run_*.sh      cálculos pesados → caché .npz en data/
│   └── *fig*.py, p9_*.py        solo leen la caché, escriben CSV en data/ y la figura en figuras/
└── data/                        cachés .npz (con ρ) y CSV de cada figura
```

## Índice de figuras
| figura (`figuras/`) | script (`codigo/`) | papel |
|---|---|---|
| `principal_fig2` | `principal_fig2.py` | **principal**: universalidad de la figura de mérito (borrador) |
| `principal_fig3` | `principal_fig3.py` | **principal**: mapa de diseño (pendiente de limpieza) |
| `apendice_resonancia_universal` | `apendice_resonancia_universal.py` | apéndice: colapso de la resonancia vestida |
| `apendice_fig2_estados` | `apendice_fig2_estados.py` | apéndice: Wigner de Ma y gato transitorio |
| `fig2`, `fig3`, `fig4` | `fig2.py`, `fig3.py`, `fig4.py` | apéndice/validación: resonancia de Ma, figura de mérito, baño filtrado |
| `p9_diagnostico` | `p9_diagnostico.py` | diagnóstico de P9 (corrimiento χ) |

**Rerun (solo estilo, sin recalcular):** `python codigo/<script>.py`.
Con `--rerun`, los `fig*.py` calculan los puntos que falten en la caché. `python codigo/calc_*.py ... --rerun` fuerza el recálculo de un punto.
Los cálculos masivos se lanzan con `codigo/run_*.sh` (se pueden llamar desde cualquier carpeta).

## Convenciones comunes (todas las figuras)
- **Marco de laboratorio, muestreo estroboscópico t = nT_p** (T_p = 2π/ω_p, fase 0 del drive). El estado estacionario es el autovector con λ = 1 del propagador de Floquet de un período.
  En este marco y en estos instantes el desplazamiento polarónico del oscilador vale +g_z/ω, con el qubit en |g⟩ y σ_z|g⟩ = −|g⟩.
  En el marco rotante a ω_p/2 ese mismo desplazamiento aparecería con signo alternante, (g_z/ω)e^{iω_pt/2}. Ver `C6_polaron.md`.
- **Código (C6, decisión final):** P_c y todas las curvas de confinamiento o resonancia usan el **código fijo {D(d)|±α_nom⟩}**, con d = g_z/ω analítico y α_nom² = Ω/G.
- **α_eff² = ⟨(a − d)²⟩** del estado estacionario (medido, no optimizado) se usa **solo** en la tasa de phase-flip (C1), evaluada en la resonancia.
  No sirve para P_c: el código adaptado sigue al estado, incluso lejos de la resonancia (en ω_p = 12 daría P_c = 0.967 con α_eff² = −2.48, sin gato).
- **Precisión de C1/C2 en función de κ₂/κ (C3 revisado, Fig. 3):** las fórmulas de orden dominante son exactas cuando κ₂/κ → 0.
  El déficit medido/predicho es la corrección no adiabática, que crece de forma suave con κ₂/κ:
  ≲ 0.4% para κ₂/κ ≲ 0.05, ~1% en κ₂/κ ≈ 0.2 y ~2% en κ₂/κ ≈ 1.
  Correlaciona con κ₂/κ (r = 0.89) más que con P_e (r = 0.69); a igual κ₂/κ = 0.16, P_e = 3e-4 y 3e-3 dan déficits de 1.3% y 1.0%.
- **Regla de truncamiento con filtro (lección de P10):** en puntos filtrados con |α|² = 4 usar **N ≥ 20** (con N = 16 aparece un exceso espurio de γ_pf de hasta ×2.9).
  En **cualquier** punto filtrado, reportar la convergencia en N junto al valor.
- **Validaciones:** en toda ρ, |Tr ρ − 1| < 1e-10, ‖ρ − ρ†‖ < 1e-10 y mínimo autovalor > −1e-9.
- Tolerancias del integrador: atol 1e-12, rtol 1e-10.

---

## Fig. 2 — Resonancia vestida (`fig2.py`, `calc_fig2.py`, `run_fig2.sh`)
**Modelo:** H completo de Ma en el laboratorio (sin RWA), ω = 6, g = 0.3, θ = π/4 (g_x = −0.2121, g_z = 0.2121), κ = 0.03 (κD[σ₋]). Unidades 2π·GHz.

**(a)** Ω = 0.06, ω_q = 12 fijo, ω_p ∈ [11.955, 12.015] con paso 0.0025 (25 puntos), **N = 22**.
- Curva principal: P_c con el código fijo polarónico. Curva discontinua: código viejo sin desplazar (α nominal = 2i).
- La línea punteada es 2(ω − 4g_x²/3ω) = 11.980 y la de trazo y punto es 2ω. La franja es la ventana con P_c > 0.99 (código fijo). Recuadro: 1 − P_c.
- Interpolación PCHIP.
- El CSV incluye además P_c con α_eff, α_eff², y P_e y paridad promediados sobre un período (C5; 40 muestras con mesolve desde el estado estroboscópico), además de P_e estroboscópico.
- **Datos:** `data/fig2a.csv`. Caché: `data/fig2/gx-0.212132_wp*_wq12.000000_Om0.060000_N22.npz` (los N20 de la primera versión se conservan).

**(b)** ω_q = ω_p (el qubit sigue al drive), |α|² = 4 (Ω = 4|G|, α = 2i), κ₂/κ ∈ {0.03, 0.1, 0.3, 1, 2}, obtenidos variando g_x = −√(κ₂/κ)·ωκ/(4g_z) con g_z fijo.
- 17 valores de ω_p por serie, en ω_p* + |G|·{−4, −2.5, −1.5, −1.2, −1.0, −0.8, −0.6, −0.3, 0, 0.3, 0.8, 1.1, 1.5, 2.0, 2.5, 4, 6}. La serie κ₂/κ = 1 tiene además el punto ω_p = 12 de (a).
- FWHM por interpolación **PCHIP** (monótona por tramos) a P_c = P_max/2 (misma definición que la Tarea 40), con el **código fijo** D(g_z/ω)|±α_nom⟩.
- Ajuste por el origen: **FWHM = cG, c = 2.93** (media 2.92 ± 0.25).
- **Datos:** `data/fig2b.csv` (FWHM y c por serie) y `data/fig2b_curvas.csv` (curvas crudas). Caché: `data/fig2/*wq==wp*.npz`.

**Método:** propagador de Floquet de un período en el laboratorio y estado estacionario (λ = 1). La ρ completa queda guardada en cada `.npz`, así que se pueden recalcular otras cantidades sin propagar.

**Convergencia:** ω_p = 11.98, N = 20 → 26: P_c(α_eff) 0.998621 → 0.998765 y P_c(viejo) 0.997407 → 0.997551, es decir **Δ = 1.4e-4**.
N = 20 subestima P_c en 1.4e-4 en el pico. V2 mostró que N = 22 ya coincide con N = 28 a <1e-5. No altera la forma de la curva ni el FWHM.

**Validaciones (111 ρ):** |Tr ρ − 1| ≤ 4.4e-16, ‖ρ − ρ†‖ = 0, mínimo autovalor ≥ 1.2e-13.

---

## Fig. 3 — Figura de mérito (`fig3.py`, `calc_fig3.py`, `run_fig3.sh`)
**Modelo:** H completo en el laboratorio, con la convención de la Tarea 42: ω_q = ω_p = 2(ω − 4g_x²/3ω), κ = 0.03, Ω = |α|²_nom·G, G = 2g_xg_z/ω (gato real).
**Método:** espectral. Propagador de un período T_p y tasa del modo de paridad γ_pf (λ real ≈ +1, mayor |Tr(PR)|, peso de borde < 0.5; C4 y C9).
El estado estacionario es el autovector λ ≈ 1. α_eff² = ⟨(a − g_z/ω)²⟩, y P_c con el código fijo.

**(a)** 15 puntos con |α|²_nom = 4 y N = 22:
- ω ∈ {4, 5, 6, 7, 8}, g_x ∈ {0.03, …, 0.15}, g_z/κ ∈ {2, …, 12}. Son los 10 puntos de V4 más 5 nuevos: (0.04, 8, 3), (0.08, 4, 6), (0.06, 8, 8), (0.1, 5, 2) y (0.05, 6, 4).
- κ₁ se obtiene invirtiendo C1 con |α|² = |α_eff²| y r = Γ₁⁺/Γ₁⁻: κ₁ = γ_pf(1+r)/(2[|α|²(1+r) + r]). κ₂ = 4G²/κ.
- Marcadores por ω; relleno si κ₂/κ ≤ 0.25, hueco si es mayor.
- Subpanel: cociente medido/predicho frente a **κ₂/κ** (corrección no adiabática).
- El CSV incluye P_e promediado en un período (`calc_fig3_pe.py`, caché `data/fig3/pe_*.npz`).
- **Datos:** `data/fig3a.csv`.

**(b)** Punto (g_x, ω, g_z/κ) = (0.05, 6, 4) con |α|²_nom = 2, 4 y 6 (N = 22, 22 y 26).
- γ_pf medido/predicho, con C1 y sin el +1.
- La curva punteada es la predicción analítica del cociente sin el +1: [Γ₁⁻x + Γ₁⁺(x+1)]/[x(Γ₁⁻+Γ₁⁺)].
- **Datos:** `data/fig3b.csv`.

**Convergencia:** (0.05, 6, 12), N = 22 → 28: γ_pf cambia 1.6e-5 relativo y P_c 1e-6.
**Validaciones (18 ρ):** |Tr ρ − 1| ≤ 2.2e-16, ‖ρ − ρ†‖ = 0, mínimo autovalor ≥ −7.2e-13. Peso de borde del modo de paridad ≤ 2.6e-6.
Caché: `data/fig3/*.npz`, con la ρ estacionaria y los 12 modos más lentos.

---

# Figuras principales (las Figs. 2–4 anteriores quedan como figuras de validación del apéndice)

## Figura principal 2 — Resonancia vestida con estados (`principal_fig2.py`, `calc_principal_fig2d.py`)
- **(a)–(c):** sin propagación nueva. Se usan las ρ estacionarias de `data/fig2/` (Ma, ω_q = 12, Ω = 0.06, N = 22, t = nT_p).
- **(a), (b):** Wigner del oscilador con el qubit trazado (QuTiP `wigner(..., g=2)`, ejes en β). (a) ω_p = 2ω = 12; (b) ω_p* = 11.98.
  Misma escala de color (±máximo común). Las cruces marcan ±2i + g_z/ω.
- **(c):** P_c(ω_p) con el código fijo polarónico (`data/principal_fig2c.csv`). Los cuadrados son los puntos de (a) y (b).
- **(d) (opcional):** gato par transitorio desde |0⟩|g⟩ en ω_p*. Propagador de un período (N = 22), evolución estroboscópica hasta Γt = 30 (Γ = κ/2).
  Se toma el máximo de la fidelidad con el gato par polarónico D(g_z/ω)(|2i⟩ + |−2i⟩): **F = 0.741 en Γt = 16.9**, con P_c = 0.965 y paridad 0.52.
  Tiene escala de color propia, de amplitud parecida (±0.31). Caché: `data/principal_fig2d.npz`, con la serie completa F(t), P_c(t) y paridad(t).
- Rejillas de Wigner: `data/principal_fig2_wigner.npz`.

| estado | P_c (código fijo) | P_e (promedio en el período) | α_eff² | paridad (promedio) |
|---|---|---|---|---|
| (a) ω_p = 12.00 | 0.8140 | 0.1149 | −2.477 − 0.206i | +0.067 |
| (b) ω_p = 11.98 | 0.99873 | 0.0060 | −3.975 + 0.012i | +0.0006 |

**Validaciones:** (a) y (b) cumplen con |Tr ρ − 1| = 0 y mínimo autovalor ≥ 4e-15. (d): |Tr ρ − 1| = 2.6e-12, ‖ρ − ρ†‖ = 1.1e-13, mínimo autovalor 5e-9.
**Convergencia:** la de (b) está en la Fig. 2 de validación (N = 22 coincide con N = 28 a <1e-5 en V2).

---

## Apéndice / validación — Fig. 4: baño filtrado (`fig4.py`, `calc_fig4.py`, `run_fig4.sh`)
- **Modelo:** Ma con ω_q = ω_p = 2(ω − 4g_x²/3ω), |α|² = 2, filtro a 2ω con 4J²/κ_f = κ.
- **(a) Método estático** (sin drive, g_z = 0, N = 8, N_f = 3): Γ₁⁻ y Γ₁⁺ coinciden con g_x²κ_eff(ω)/ω² y g_x²κ_eff(3ω)/(9ω²) al 0.1–0.25% para κ_f ∈ {0.1, 0.3, 1, 3} y en el baño plano.
  N_f = 2 frente a 3: ≤ 4e-6. N = 8 frente a 12: ≤ 2e-6.
- **(b) Floquet** (N = 16, N_f = 2): mejora de γ_pf medida frente a la predicha con C1: 19.5/19.5, 166.5/166.1, 1838/1833 y 16528/16487 (≤ 0.3%).
- **Tasa de confinamiento (C9):** el 5.º modo espectral es espurio. En el baño plano su |Im λ| crece con N (0.040, 0.054 y 0.061 para N = 16, 22 y 28) y su tasa cambia un 2.2% entre N = 16 y 22.
  Se usa la tasa **dinámica**: retorno de P_c(t) desde |0⟩|g⟩ y D(d)|1.3α⟩|g⟩, ajustando la cola exponencial por encima del piso de paridad.
  Baño plano: 6.98e-3 / 7.13e-3 (los dos estados iniciales), idéntico para N = 22 y 28.
  Relativa al baño plano: 0.965 (κ_f = 3), 0.967 (1), 0.90 (0.3) y 0.53 (0.1). Es una pérdida menor que la ×0.23 de la Tarea 43, que usaba el modo espectral espurio.
- **Convergencia con filtro** (κ_f = 1, N = 16 → 18; N = 20 no cabe en memoria): γ_pf cambia −1.5e-4 y el confinamiento 0.2–0.5%.
- **Validaciones:** todas las ρ con |Tr ρ − 1| ≤ 2e-16 y mínimo autovalor ≥ −2e-11.
- **Datos:** `data/fig4a.csv`, `data/fig4b.csv`, caché `data/fig4/`.

---

## Figura principal 2 (nueva) — regla de operación universal (`principal_fig2.py`, `calc_principal_fig2.py`, `run_principal_fig2.sh`)
**Estado: figura terminada** (`principal_fig2.pdf/png`). Análisis en `data/principal_fig2_resumen.csv` y todas las curvas en `data/principal_fig2.csv`.
- **Presentación (decisión P1):** colores por |α|² (azul = 4, naranja = 2) y estilos por plataforma (ver leyenda). Ma con ω_q fijo en punteado negro.
  Recuadro: Wigner de Ma en x = 0 y en ω_p = 2ω (x = 1.33, marcado con una flecha).
- **P2 (hecho):** los puntos con |x| ≤ 1 de las curvas con |α|² = 4 que tenían N = 16 se recalcularon con N = 22 (48 puntos). Si un ω_p está repetido, se usa el N mayor.
  P_max pasa a 0.9983–0.9999 en todas las curvas salvo ω = 6, g_z/κ = 20 (0.9905). Allí g_z/ω = 0.1, en el límite de validez del polarón a primer orden.
- **Texto (P1 aceptado):** pico universal en x = 0 y ancho ≈ 3G (|α|² = 4) a ≈ 4G (|α|² = 2), con asimetría que crece con κ₂/κ.
  Con N = 22, el pico parabólico queda en |x| ≤ 0.09 salvo ω = 8, κ₂/κ = 1: x = +0.16, con semianchos 0.68/2.18, la curva más asimétrica.
- **Variable:** x = (ω_p − ω_p*)/G, con ω_p* = 2(ω − 4g_x²/3ω) y G = |2g_xg_z/ω|. P_c con el código fijo polarónico; ω_q = ω_p salvo en Ma (ω_q = 12).
- **Curvas:** serie 1 (ω = 6, cinco κ₂/κ, N = 20, caché de la Fig. 2 de validación); ω = 4 y 8 con g_z/κ = 7 y κ₂/κ = 0.1 y 1 (N = 16); ω = 6 con g_z/κ = 4 y 20 y κ₂/κ = 0.1 (N = 16);
  |α|² = 2 (N = 14); Naseem en unidades de κ (ω = 1000, g_z = 60, g_x = 6, κ₂/κ = 2.07, y g_x reducido hasta κ₂/κ = 0.1; N = 14); Ma con ω_q fijo (N = 22).
  ω/κ = 1000 no encarece el propagador en el marco de laboratorio, porque cada integración cubre un período del drive.
- **Máximo:** parábola sobre |x| ≤ 0.6. Queda en |x| < 0.1 en todas las curvas salvo ω = 8, κ₂/κ = 1 (+0.15). Las curvas son asimétricas, lo que sesga la parábola, y la rejilla tiene paso 0.3 cerca del pico.
- **FWHM en x:** 2.5–3.25 con |α|² = 4 y 3.7–4.2 con |α|² = 2 (incluido Naseem). El ancho depende de |α|².
- **Asimetría:** crece con κ₂/κ. El semiancho izquierdo baja de ~1.3 (κ₂/κ = 0.03–0.1) a ~0.6–0.75 (κ₂/κ = 1–2) y el derecho sube a ~2.2.
- **Convergencia:**
  - ω = 4, κ₂/κ = 1, N = 16 → 22: P_c en el pico sube 3.7e-3 (0.9945 → 0.9983); en el flanco x = 1.5 cambia 9e-5.
  - Naseem κ₂/κ = 2.07, N = 14 → 20: 1.5e-4 en el pico y −3.0e-3 en x = 1.5.
  - **N = 16 subestima el P_max de las curvas con |α|² = 4** (ω = 4, 8 y g_z/κ = 4, 20), pero apenas mueve los flancos ni el FWHM.
- **Validaciones (157 ρ):** |Tr ρ − 1| ≤ 4.4e-16, ‖ρ − ρ†‖ = 0, mínimo autovalor ≥ 6.5e-15.
- La versión anterior (Wigner + transitorio) se movió a `apendice_fig2_estados.py` (sección de apéndice).

---

## Figura principal 3 — espacio de diseño (`principal_fig3.py`, `calc_minimo.py`, `calc_pfig3.py`, `run_pfig3.sh`)
- **Mapa:** ε = κ₁/κ₂^eff en el plano (g_z/κ, κ₂/κ), con κ₁ = (10/9)g_x²κ/ω² y (g_x/ω)² = (κ₂/κ)/(16(g_z/κ)²).
  κ₂^eff = Δ(κ₂)/c, donde Δ es la tasa de confinamiento **dinámica** del modelo mínimo (|α|² = 4, retorno de P_c desde |0⟩|g⟩).
  c = lim Δ/κ₂ = **4.14** (extrapolación lineal de los tres κ₂/κ más pequeños).
  - Curvas de nivel: ε = 1/220 (continua) y ε = 1e-3 (discontinua).
  - **Validez (P9, sin rayado):** curvas naranjas χ|α|²/κ = 0.3 con χ = (8/3)g_x²/ω, para ω/κ = 200 (continua) y 1000 (punteada), es decir κ₂/κ = 0.45(g_z/κ)²/(ω/κ) con |α|² = 4.
    Vertical g_z/ω = 0.1 para ω/κ = 200 (g_z/κ = 20); para ω/κ = 1000 cae en el borde (g_z/κ = 100) y va en el pie. Línea blanca de trazo y punto: g_x/ω = 0.1, que no depende de ω/κ.
    Por encima de la curva χ el mapa sobreestima el confinamiento (más del 5%).
- **Modelo mínimo:** 30 valores de κ₂/κ entre 1e-3 y 10 con N = 24 (`data/minimo/`, `data/principal_fig3_minimo.csv`).
  - Dinámica exacta por descomposición espectral completa del Liouvilliano estático.
  - La tasa dinámica desde |0⟩|g⟩ coincide con la espectral al 1e-3 hasta κ₂/κ ≈ 0.2.
  - **C9:** en 8 valores con N = 24, 30 y 36, la tasa dinámica es estable (≤ 2e-4). La espectral no converge en el régimen saturado (0.216 → 0.231 → 0.238 en κ₂/κ = 0.57): es la rama interior no convergida.
  - κ₂^eff/κ₂ = 0.98 (1e-3), 0.84 (0.01), 0.50 (0.045), 0.24 (0.16) y 0.058 (1).
  - **Saturación:** la tasa dinámica satura en ≈ 0.28κ (0.2784κ en κ₂/κ = 10). **No es un techo estricto en κ/4**: el valor exacto κ/4 corresponde a α = 0 (R6). Con |α|² = 4 la meseta queda algo por encima.
  - **Inicio de la saturación:** cκ₂ ≈ κ/4, es decir **κ₂/κ ≈ 1/(4c) = 0.060** para c = 4.14 (|α|² = 4). Allí κ₂^eff/κ₂ ≈ 0.43.
- **Verificación adiabática:** ε·(g_z/κ)² = 0.0707 en κ₂/κ = 1e-3, frente a 5/72 = 0.0694 (κ₂^eff/κ₂ = 0.983). Es exacta en el límite κ₂ → 0.
- **Puntos del modelo completo** (15, N = 22, `data/principal_fig3_puntos.csv`):
  - γ_pf espectral y confinamiento dinámico reconstruido exactamente a tiempos estroboscópicos, P_c(n) = Σ λⁿ c p.
  - κ₁ se obtiene de γ_pf invirtiendo C1 con |α_eff²|. Los puntos van coloreados con la misma escala.
  - ε_completo/ε_mapa va de 0.97 a 1.55. **Causa identificada (P9):** el corrimiento del qubit dependiente de n, χn|e⟩⟨e| con χ = (8/3)g_x²/ω.
    - Diagnóstico en `p9_diagnostico.py`, `data/p9_diagnostico.csv` y `p9_diagnostico.pdf`.
    - El modelo efectivo estático **con** χ reproduce el confinamiento del modelo completo al 0.5–4% (0.632 frente a 0.640 en el peor punto).
    - **Sin** χ reproduce el modelo mínimo (≥ 0.99).
    - Con ω/κ = 500 (mismos g_x/ω, g_z/κ y κ₂/κ) la desviación crece: conf/Δ = 0.175 (el efectivo con χ da 0.176) y 0.13.
    - Con ω/κ = 1000 (χ|α|²/κ = 4.3 y 6.7) el gato no se estabiliza: P_c estacionario 0.49 y 0.95, dinámica no monótona, sin ajuste exponencial posible.
    - Umbral empírico: conf/Δ ≥ 0.98 para χ|α|²/κ ≲ 0.2; 0.91 en 0.45; 0.64 en 1.3. **5% de pérdida en χ|α|²/κ ≈ 0.3.**
    - **Ma** tiene χ|α|²/κ = 2.7: fuera de la región cuantitativa del mapa, aunque el gato se estabiliza (P_c = 0.9987).
    - **Naseem** tiene χ|α|²/κ = 0.19 (|α|² = 2): dentro.
- **Validaciones (15 ρ):** |Tr ρ − 1| ≤ 2.2e-16, ‖ρ − ρ†‖ = 0, mínimo autovalor ≥ −7.2e-13.
- **Pie de figura:** el mapa usa la fórmula de orden dominante para κ₁. La corrección no adiabática de γ_pf es ~1% en κ₂/κ ≈ 0.2 y ~2% en κ₂/κ ≈ 1.
  Naseem (ω/κ = 1000) cae en la zona rayada, que se calculó con ω/κ = 200; con su ω/κ real, g_z/ω = 0.06 < 0.1.

---

## Figura central — mapa de diseño con baño plano y filtrado (`codigo/figura_central.py`, `calc_minimo_filtro.py`, `calc_filtro_completo.py`)
**Pie (borrador):** ε = κ₁/κ₂^eff en el plano (g_z/κ, κ₂/κ), |α|² = 4, κ₂^eff = Δ/c con Δ la tasa de confinamiento dinámica del modelo mínimo y c = 4.14 (pendiente adiabática del baño plano).
- Curva blanca: umbral del código de repetición κ₂/κ₁ = 220 (ε = 1/220).
- Región aclarada: χ|α|² > 0.3κ, con χ = (8/3)g_x²/ω y ω/κ = 200.
- **(a)** Baño plano: κ₁ = (10/9)g_x²κ/ω².
- **(b)** Qubit acoplado a un filtro centrado en 2ω con κ_f/ω = 0.05 (κ_f/κ = 10 con ω/κ = 200): κ₁ = g_x²[κ_eff(ω)/ω² + κ_eff(3ω)/(9ω²)], con κ_eff(δ) = κκ_f²/(4δ² + κ_f²).
  **El costo en confinamiento del filtro está incluido:** Δ_filtro se calcula con el modelo mínimo más el modo filtro, J(σ₊b + h.c.), κ_f D[b], 4J²/κ_f = κ, sin decaimiento directo del qubit, N = 20, N_f = 3.
  Por encima de κ₂/κ = 1.5 (gris) el retorno de P_c no es monótono (sobrepaso) y la tasa no está definida.
- **(c)** Colapso del modelo completo: (κ₁/κ₂)(g_z/κ)²/(5/72) frente a κ₂/κ (19 puntos con gato estable).

**Resultados:**
- **Desplazamiento del umbral en g_z/κ:** en el régimen adiabático pasa de 3.94 (plano) a 0.094 (filtro), un factor 0.0238. La predicción es √(f/(10/9)) = 0.0239, con f = κ_eff(ω)/κ + κ_eff(3ω)/(9κ) = 6.32e-4; κ_f/(2ω) = 0.025.
  El factor se mantiene en 0.023–0.024 hasta κ₂/κ ≈ 0.3 y sube a 0.034 en κ₂/κ = 1, donde el filtro frena el confinamiento.

  | κ₂/κ | 1e-3 | 0.01 | 0.03 | 0.1 | 0.3 | 1 |
  |---|---|---|---|---|---|---|
  | Δ_filtro/Δ_plano | 1.002 | 1.014 | 1.037 | 1.051 | 0.955 | 0.480 |

- **Δ_filtro/Δ_plano:** con κ₂ pequeño el filtro confina algo **más rápido** (hasta +5%); en saturación, la mitad.
- **Controles del modelo mínimo con filtro:**
  - Con κ_f = 1000κ se recupera el plano (1.0015).
  - N = 20 → 26 cambia < 0.1%.
  - N_f = 2 → 3 cambia 1.2%; se usa N_f = 3.
- **Verificación con el modelo completo con filtro** (`verif_figura_central.py`, N = 16, N_f = 2, ω/κ = 200, κ_f = 0.3):
  - (κ₂/κ, g_z/κ) = (0.03, 4): ε_completo/ε_mapa = **1.03**; κ₁ = 1.03 veces el predicho y confinamiento 1.04 veces el del plano.
  - (0.3, 12): con N = 16 salía 2.29, pero era un **artefacto de truncamiento** (P10).
  - **Con N = 20:** (0.03, 12) da ε_completo/ε_mapa = 0.987 y (0.3, 12) da 0.975 (γ_pf/predicho = 0.986 y 0.980). El panel (b) queda verificado en los dos regímenes.
  - **Convergencia en N** (N_f = 2; N = 24 no cabe en 14 GB, ~13 GB solo en el propagador):

    | punto (κ₂/κ, g_z/κ) | N = 16 | N = 20 | N = 22 |
    |---|---|---|---|
    | (0.03, 12): γ_pf/predicho | 2.880 | 0.986 | 0.978 |
    | (0.03, 12): ε_completo/ε_mapa | 2.831 | 0.987 | 0.979 |
    | (0.3, 12): γ_pf/predicho | 2.261 | 0.980 | 0.977 |
    | (0.3, 12): ε_completo/ε_mapa | 2.288 | 0.975 | 0.972 |

    Entre N = 20 y 22, γ_pf cambia 0.8% y 0.3%, y el confinamiento < 0.05%. Validaciones con N = 22: |Tr ρ − 1| = 0 y mínimo autovalor ≥ −5.6e-9.
- **Precisión con filtro (declarada):** ε_completo/ε_mapa = 0.979 en (κ₂/κ, g_z/κ) = (0.03, 12) y 0.972 en (0.3, 12), con N = 22. Es decir, **~2–3%**, convergido (N = 20 → 22: < 1%).
- **Piso intrínseco en (b):** κ₁ → κ₁^filt + γ, rotulado por **γ/κ** (el piso depende solo de γ/κ = (ω/κ)/Q).
  Curvas: naranja γ/κ = 2e-4, celeste γ/κ = 2e-5; continua para ε = 1/220, discontinua para ε = 1e-3; líneas blancas finas sin piso.
  - **Pie:** Ma (Q = 1e7, ω/κ = 200) tiene γ/κ = 2e-5. Naseem (γ/2π = 15 Hz, κ/2π = 100 kHz) tiene γ/κ = 1.5e-4.
  - Alcanzar ε = 1/220 exige κ₂^eff/κ ≥ 220γ/κ: 0.044 con γ/κ = 2e-4 y 0.0044 con 2e-5. Para ε = 1e-3, ≥ 0.2 y ≥ 0.02.
  - **γ/κ = 2e-4:** ε = 1/220 solo en la franja κ₂/κ ≈ 0.19–0.33, con g_z/κ ≳ 0.8–1.9 según κ₂; ε = 1e-3 no se alcanza.
    **Dentro de la zona verificada (χ|α|² ≤ 0.3κ, ω/κ = 200) solo queda la parte con g_z/κ ≥ 9.3 (κ₂/κ = 0.19) a 11.9 (0.32), es decir g_z/κ ≳ 10.**
  - **γ/κ = 2e-5:** el umbral de 1/220 se desplaza +44% en κ₂/κ = 0.01, +14% en 0.03, +7% en 0.1, +5% en 0.3 y +8% en 1; el piso nunca baja del 10% de κ₁^filt.
  - **Dos versiones para revisión (sin cómputo nuevo, mismos mapas):**
    - `figura_central`: el color de (b) incluye el piso γ/κ = 2e-5.
    - `figura_central_sinpiso` (`python codigo/figura_central.py --sin-piso`): el color de (b) es ε **sin** pérdida intrínseca (resultado universal del esquema), el piso solo aparece como curvas, y un punteado marca donde γ/κ = 2e-5 supera a κ₁^filt (donde filtrar más ya no aporta).
      Recomendada para el paper, porque no ata el mapa a un γ/κ de plataforma. Pie: "el color muestra ε sin pérdida intrínseca; las curvas incluyen γ/κ = 2e-5 y 2e-4".
  - **Color de (b) en `figura_central`:** ε **con** el piso γ/κ = 2e-5. Escala limitada a ε ≥ 1e-5; por debajo mandan otros canales (pérdida intrínseca, temperatura, desfase del qubit).
  - La región χ|α|² > 0.3κ (ω/κ = 200) se superpone sombreada sobre las curvas del piso.
  - **Verificación con piso** (modelo completo con filtro y γ = 2e-4κ como γD[a]; opción `--gam` de `calc_filtro_completo.py`):
    - (κ₂/κ, g_z/κ) = (0.25, 14), χ|α|²/κ = 0.17, N = 22: γ_pf/predicho = 1.007, confinamiento/mapa = 0.979, **ε_completo/ε_mapa = 1.028**.
      ε_completo = 4.39e-3 < 1/220: el diseño de la franja verificada funciona.
    - (0.25, 6), χ|α|²/κ = 0.93 (fuera de la zona), N = 20: γ_pf/predicho = 1.000, pero **confinamiento/mapa = 0.751** y ε_completo/ε_mapa = 1.33.
      **La frontera de χ vale con filtro**, con una pérdida coherente con P9 (0.91 en 0.45, 0.64 en 1.3).
  - Validaciones: |Tr ρ − 1| ≤ 2e-16 y mínimo autovalor ≥ −1e-12.

---

## Figura térmica (P7) — `codigo/figura_termica.py`, `calc_termico.py`, `run_termico.py`, `run_termico_gamma.py`, `ajuste_cbf.py`
**Definición (la de la Tarea 37, `validacion/tareas_naseem/tarea37_worker.py`):** η = γ_pf/γ_bf.
- γ_pf: tasa del modo, entre los modos lentos 1–3, con mayor traslape con P = e^{iπa†a}.
- γ_bf: tasa del modo con mayor traslape con a (modo de pozo).
- x = hf_q/(k_BT), con f_q la frecuencia del **qubit**; n_q = 1/(eˣ − 1).
- Eje superior: f_q a 10 mK = x · 208.37 MHz (k_B·10 mK/h, calculado con `scipy.constants`).

**Umbrales η = 100 y η = 220: valores de referencia, no cotas derivadas aquí.**
- η = 100 es el criterio de sesgo que se usó en la Tarea 37 y en el plan de figuras (C8). Es una convención de trabajo; no tiene justificación propia en este proyecto.
- 220 es el umbral de Guillaud–Mirrahimi, que se define sobre κ₂/κ₁ y **no** es una cota sobre γ_pf/γ_bf. Aquí se usa solo como segundo nivel de referencia.
- La figura los llama "valores de referencia" (pie).

**Supuestos de los baños** (convención de la Tarea 37 y del modelo completo con un solo baño de Lindblad):
- El qubit, o el filtro, y los canales de un fonón mediados por el qubit (Γ₁±) usan n_q, la ocupación a f_q.
- La pérdida intrínseca del oscilador usa n_m a f_q/2.
- **Control** (`calc_termico.punto(..., ocup='real')`): si Γ₁⁻ usa n(ω) y Γ₁⁺ usa n(3ω), las frecuencias reales del fonón emitido:
  - con filtro, γ_pf cambia ≤ 1e-4 y x*(100) no cambia (10.114);
  - en el plano, γ_pf sube +1.1% en x = 10 y +2.9% en x = 8, y x*(100) pasa de 8.197 a 8.173 (−0.3%).

**Método:** modelo efectivo estático con el qubit explícito (P9) y el filtro explícito (N_f = 2), con baños térmicos (ver el docstring de `calc_termico.py`). Espectro con eigs disperso (shift-invert).
- Isla: κ₂/κ = 0.25, g_z/κ = 14, ω/κ = 200, |α|² = 4, con κ_f/ω = 0.05 o baño plano.
- Rejillas: x ∈ [1, 25] (40 valores) × κ₂/κ ∈ [0.02, 0.4] (g_z/κ = 14; χ|α|²/κ ≤ 0.27) y × γ/κ ∈ [1e-6, 1e-3] en la isla. N = 22, y N = 24 en la isla.
  - El barrido en g_z/κ ∈ [10, 40] se calculó, pero ya no se grafica, porque con filtro casi no mueve el umbral (x*(100) = 10.4 → 10.0).
  - Ojo: su fila g_z/κ = 10 tiene χ|α|²/κ = 0.333 > 0.3; las demás cumplen.
- **Figura:**
  - (a) η(x) en la isla. Más allá de x = 13 las curvas son el piso de truncamiento a T → 0 (no convergido): van tenues y punteadas, y el η(T→0) real es mayor, así que es una cota inferior.
  - (b) Mapa η(γ/κ, x) en la isla.
  - (c) Mapa η(κ₂/κ, x) con γ/κ = 2e-5.
  - En (b) y (c): contornos blancos para el baño filtrado y naranjas para el plano; continuos η = 100, discontinuos η = 220. El color es η con filtro.
  - (d) γ_bf/κ frente a n_q.
- **Datos:** `data/termico_curvas.csv`, `termico_umbrales.csv`, `termico_umbral_parametro.csv`, `termico_ajuste_bf.csv`, `termico_cbf.csv`, `termico_mapas.npz`, `termico_mapa_gamma.npz`. Caché: `data/termico/`.

**Umbrales en la isla** (N = 22; N = 24 cambia x* en ≤ 0.008):

| baño | γ/κ | x*(η=100) | x*(η=220) | f_q a 10 mK (η = 100 / 220) |
|---|---|---|---|---|
| filtrado | 2e-5 | 10.11 | 10.91 | 2.11 / 2.27 GHz |
| filtrado | 2e-4 | 7.78 | 8.59 | 1.62 / 1.79 GHz |
| plano | 2e-5 | 8.19 | 9.01 | 1.71 / 1.88 GHz |
| plano | 2e-4 | 7.18 | 8.00 | 1.50 / 1.67 GHz |

- **El filtro sube el requisito en x porque reduce γ_pf** (el numerador de η); γ_bf/n_q sube solo un 28% (0.032κ → 0.041κ en la isla). η mide sesgo, no calidad.
- **Dependencia en γ/κ** (panel (b)): el umbral depende de γ/κ mientras γ domine γ_pf. Con γ/κ ≲ 1e-5, en el plano el umbral se vuelve casi independiente de γ (x*(100) ≈ 8.4, régimen térmico puro). Con filtro sigue moviéndose.

**c_bf(κ₂/κ) por ajuste directo de γ_bf** (`ajuste_cbf.py`, `data/termico_cbf.csv`): γ_bf − γ_bf(T→0) = c_bf·n_q·κ con n_q ∈ [1e-5, 1e-2] (exponente local p = 1.00–1.02):

| κ₂/κ | 0.02 | 0.05 | 0.2 | 0.4 |
|---|---|---|---|---|
| c_bf con filtro | 2.62e-4 | 1.65e-3 | 2.75e-2 | 8.42e-2 |
| c_bf en el plano | 2.76e-4 | 1.67e-3 | 2.23e-2 | 6.20e-2 |

- La incertidumbre (semi-rango intercuartil) es ≤ 2%.
- **c_bf ∝ (G/κ)^s**, con G/κ = √(κ₂/κ)/2: **s = 3.96 ± 0.04 con filtro y 3.69 ± 0.05 en el plano.**
- **El "≈0.05 n_qκ" de las Tareas 36–37 no es una constante** y no debe citarse como tal: c_bf varía más de dos órdenes de magnitud con κ₂/κ.

**Supresión con |α|²** (x = 9 y 12, |α|² = 2–6): d ln γ_bf/d|α|² ≈ −0.55 y −0.64 en el plano, −0.43 y −0.42 con filtro.
- Con G/κ = 0.25, el plano coincide con la Tarea 44 (Gautier: ≈ −0.55 con g₂/κ = 0.3). El filtro debilita la supresión.
- La supresión fuerte (−1.0 a −1.4) solo aplica con g₂/κ ≲ 0.1.

**Convergencia y validaciones:**
- En la isla, 34 de 160 puntos cambian > 3% entre N = 22 y 24, todos con x ≳ 13 (el piso de T = 0). Ningún umbral cae ahí: la diferencia local de η junto a cada umbral es ≤ 1.5%.
- Región caliente n_q > 0.3 (x < 1.47): achurada y excluida.
- **Hermiticidad antes de hermitizar** (sin simetrizar por paridad) en todas las ρ: ‖ρ − ρ†‖ ≤ 1.4e-11 (tolerancia 1e-10); |Tr ρ − 1| ≤ 6.7e-16; mínimo autovalor ≥ −8.7e-17.
- **Validación con el modelo completo** (con filtro, temperatura y γ = 2e-5κ; N = 22; unidades de Ma κ = 0.03, ω = 6; `run_validacion_termica.sh`, `verif_termica.py`, `data/termico_validacion.csv`).
  Criterio: γ_pf y γ_bf dentro de ±5%, y γ_bf(T=0) < 1% del térmico.

  | κ₂/κ | x | γ_pf c/e | γ_bf c/e | η c/e | η completo / efectivo | γ_bf(T=0)/γ_bf térmico | hermiticidad cruda |
  |---|---|---|---|---|---|---|---|
  | 0.05 | 6.86 | 0.959 | 0.956 | 1.003 | 100.2 / 99.95 | 0.30% | 1.8e-9 |
  | 0.10 | 6.86 | 0.959 | 0.856 | 1.120 | 25.7 / 23.0 | 0.17% | 2.1e-9 |
  | 0.25 | 6.86 | 0.959 | 0.785 | 1.222 | 4.94 / 4.04 | 0.09% | 4.1e-10 |
  | 0.40 | 6.86 | 0.959 | 0.776 | 1.236 | 2.40 / 1.94 | 0.09% | 1.6e-12 |
  | 0.25 | 10.1 | 0.956 | 0.798 | 1.199 | 118.2 / 98.6 | **2.35%** | 1.2e-9 |

  - **Solo κ₂/κ = 0.05 cumple ±5% en γ_bf.** γ_pf c/e ≈ 0.959 en todos (es α_eff² ≈ 3.80 frente a |α|² = 4 del efectivo, más ~1%).
  - **La discrepancia de γ_bf depende de κ₂/κ y no de x** (fase 2: a κ₂/κ = 0.25, x = 6.86 y 10.1 dan 0.785 y 0.798). Cae monótonamente (0.956, 0.856, 0.785, 0.776) y se aplana; en κ₂/κ = 0.4 (χ|α|²/κ = 0.27, junto a la frontera ≈ 0.3) no se ve un desplome propio de la frontera. No se ajusta ni se aplica corrección empírica.
  - **El efectivo es conservador:** x*(η) del completo (estimado como η_completo(x) ≈ (η c/e a x = 6.86)·η_efectivo(x)) queda 0–0.2 por debajo: x*(100) efectivo/completo = 6.86/6.86, 8.36/8.25, 10.11/9.91, 10.84/10.63; x*(220) = 7.67/7.66, 9.17/9.05, 10.91/10.71, 11.64/11.43 (`data/termico_tabla_k2.csv`).
  - **El piso de T = 0 pasa el 1% a x = 6.86, pero no en la isla a x = 10.1** (el γ_bf térmico cae como e^{−x}). Con η_max ≈ 5000 frente a los umbrales (100, 220) se propone reportar el piso y no usarlo como criterio.
  - **Tolerancia del integrador** (fase 2, isla): atol 1e-14 / rtol 1e-12 en lugar de 1e-12 / 1e-10 cambia γ_bf en 2.6e-5 relativo y el piso en 0.10%; γ_pf no cambia.
  - **Hermiticidad cruda** (‖ρ − ρ†‖_F con Tr ρ = 1, antes de hermitizar) de las corridas completas con filtro: 1.6e-12 a 2.3e-9; **por encima de la tolerancia 1e-10 en 10 de las 11 corridas con filtro** (8 de 9 a tolerancia normal; solo κ₂/κ = 0.4 a x = 6.86 cumple; no se relajó). Log crudo con marcas: `data/filtro_completo/log_hermiticidad_positividad.txt` (`codigo/log_herm_posit.py`). No depende de la tolerancia del integrador.
  - **Positividad:** mínimo autovalor de (ρ + ρ†)/2 ≥ −2.5e-9. **El criterio de las figuras (−1e-9) se incumple en 2 de las 11 corridas con filtro** (κ₂/κ = 0.05 a T = 0: −2.5e-9; 0.1 a T = 0: −1.4e-9); no se cambia el criterio aquí, queda a decisión del autor.
  - **Baño plano (no validado):** completo sin filtro en (0.25, x = 6.86): γ_pf c/e = 1.52, γ_bf c/e = 0.858. El factor 1.52 es térmico (a T = 0, c/e = 0.940 y el completo coincide con la fórmula analítica a 1%); apagando n_q en el completo plano γ_pf baja de 1.369e-3 a 8.45e-4 (98% del aumento térmico) mientras que apagar n_m solo quita 1.1e-5. El efectivo no contiene ese canal del qubit (ver PENDIENTES P7). Las cantidades planas de `figura_termica` están marcadas "not validated". Barrido en N plano: N = 18 no converge; N = 22 → 26 cambia γ_bf(T=0) en −1.9% y γ_pf en ≤ 1e-5.
  - **Tiempos** (un hilo efectivo, filtrado, N = 22, N_f = 2): 90–125 min por corrida (dos en paralelo agotan los 14 GB). Plano: 4–14 min (N = 22), 20–25 min (N = 26).
  - El primer intento en unidades κ = 1 (ω = 200) no sirve: modo de pozo no convergido.
