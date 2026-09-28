# Figuras finales (PRA)

Entorno: `../verificacion_independiente/.venv` (Python 3, QuTiP 5.3.1). Todo se ejecuta **desde esta carpeta** (`figuras_finales/`).

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
  - (0.3, 12): ε_completo/ε_mapa = **2.29**. El confinamiento sí coincide (0.94 frente a 0.955 del mapa), pero **κ₁ = 2.26 veces el filtrado predicho** (ver P10).
  - Validaciones: |Tr ρ − 1| ≤ 2e-16 y mínimo autovalor ≥ −1e-12.
