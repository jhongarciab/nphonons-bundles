# Figuras finales (PRA)

Entorno: `../verificacion_independiente/.venv` (Python 3, QuTiP 5.3.1).
Estructura:
- `comun.py`: modelo y utilidades.
- `estilo.py`: estilo común (serif 8.5 pt, paleta Okabe–Ito, ancho de columna PRA 3.375 in).
- `calc_*.py`: cálculos pesados, con caché `.npz` en `data/`.
- `figN.py`: solo leen la caché, escriben los CSV y dibujan `figN.pdf` y `figN.png` (300 dpi).

**Rerun:** para cambiar solo el estilo basta con correr `python figN.py`, que no recalcula nada.
`python figN.py --rerun` calcula los puntos que falten en la caché, y `calc_*.py ... --rerun` fuerza el recálculo de un punto.

## Convenciones comunes (todas las figuras)
- **Marco de laboratorio, muestreo estroboscópico t = nT_p** (T_p = 2π/ω_p, fase 0 del drive). El estado estacionario es el autovector con λ = 1 del propagador de Floquet de un período.
  En este marco y en estos instantes el desplazamiento polarónico del oscilador vale +g_z/ω, con el qubit en |g⟩ y σ_z|g⟩ = −|g⟩.
  En el marco rotante a ω_p/2 ese mismo desplazamiento aparecería con signo alternante, (g_z/ω)e^{iω_pt/2}. Ver `C6_polaron.md`.
- **Código (C6, decisión final):** P_c y todas las curvas de confinamiento o resonancia usan el **código fijo {D(d)|±α_nom⟩}**, con d = g_z/ω analítico y α_nom² = Ω/G.
- **α_eff² = ⟨(a − d)²⟩** del estado estacionario (medido, no optimizado) se usa **solo** en la tasa de phase-flip (C1), evaluada en la resonancia.
  No sirve para P_c: el código adaptado sigue al estado, incluso lejos de la resonancia (en ω_p = 12 daría P_c = 0.967 con α_eff² = −2.48, sin gato).
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
- Marcadores por ω; relleno si κ₂/κ ≤ 0.25, hueco si es mayor (C3).
- Subpanel: cociente medido/predicho; la franja gris marca ±0.4%.
- **Datos:** `data/fig3a.csv`.

**(b)** Punto (g_x, ω, g_z/κ) = (0.05, 6, 4) con |α|²_nom = 2, 4 y 6 (N = 22, 22 y 26).
- γ_pf medido/predicho, con C1 y sin el +1.
- La curva punteada es la predicción analítica del cociente sin el +1: [Γ₁⁻x + Γ₁⁺(x+1)]/[x(Γ₁⁻+Γ₁⁺)].
- **Datos:** `data/fig3b.csv`.

**Convergencia:** (0.05, 6, 12), N = 22 → 28: γ_pf cambia 1.6e-5 relativo y P_c 1e-6.
**Validaciones (18 ρ):** |Tr ρ − 1| ≤ 2.2e-16, ‖ρ − ρ†‖ = 0, mínimo autovalor ≥ −7.2e-13. Peso de borde del modo de paridad ≤ 2.6e-6.
Caché: `data/fig3/*.npz`, con la ρ estacionaria y los 12 modos más lentos.
