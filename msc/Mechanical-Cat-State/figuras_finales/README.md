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
- **Código (C6, decisión aprobada):** {D(d)|±α_eff⟩}, con d = g_z/ω analítico y α_eff² = ⟨(a − d)²⟩ del estado estacionario (medido, no optimizado).
  El mismo α_eff² se usa en la tasa de phase-flip (C1).
- **Código fijo:** para anchos de resonancia se usa D(d)|±α_nom⟩, con α_nom² = Ω/G. Ver la Fig. 2(b) para la razón.
- **Validaciones:** en toda ρ, |Tr ρ − 1| < 1e-10, ‖ρ − ρ†‖ < 1e-10 y mínimo autovalor > −1e-9.
- Tolerancias del integrador: atol 1e-12, rtol 1e-10.

---

## Fig. 2 — Resonancia vestida (`fig2.py`, `calc_fig2.py`, `run_fig2.sh`)
**Modelo:** H completo de Ma en el laboratorio (sin RWA), ω = 6, g = 0.3, θ = π/4 (g_x = −0.2121, g_z = 0.2121), κ = 0.03 (κD[σ₋]). Unidades 2π·GHz.

**(a)** Ω = 0.06, ω_q = 12 fijo, ω_p ∈ [11.955, 12.015] con paso 0.0025 (25 puntos), N = 20.
- Curvas: P_c con el código polarónico (α_eff), con el código viejo (sin desplazamiento, α nominal = 2i) y con el código polarónico de α nominal.
- La línea punteada es 2(ω − 4g_x²/3ω) = 11.980 y la de trazo y punto es 2ω. La franja verde es la ventana con P_c > 0.99 (código α_eff): [11.9753, 11.9880].
- Recuadro: 1 − P_c en escala logarítmica.
- **Datos:** `data/fig2a.csv`. Caché: `data/fig2/gx-0.212132_wp*_wq12.000000_Om0.060000_N20.npz`.

**(b)** ω_q = ω_p (el qubit sigue al drive), |α|² = 4 (Ω = 4|G|, α = 2i), κ₂/κ ∈ {0.03, 0.1, 0.3, 1, 2}, obtenidos variando g_x = −√(κ₂/κ)·ωκ/(4g_z) con g_z fijo.
- 17 valores de ω_p por serie, en ω_p* + |G|·{−4, −2.5, −1.5, −1.2, −1.0, −0.8, −0.6, −0.3, 0, 0.3, 0.8, 1.1, 1.5, 2.0, 2.5, 4, 6}. La serie κ₂/κ = 1 tiene además el punto ω_p = 12 de (a).
- FWHM por spline cúbica a P_c = P_max/2 (misma definición que la Tarea 40), con el **código fijo** D(g_z/ω)|±α_nom⟩.
- Ajuste por el origen: **FWHM = cG, c = 2.93** (media 2.92 ± 0.25).
- **Datos:** `data/fig2b.csv` (FWHM y c por serie) y `data/fig2b_curvas.csv` (curvas crudas). Caché: `data/fig2/*wq==wp*.npz`.

**Método:** propagador de Floquet de un período en el laboratorio y estado estacionario (λ = 1). La ρ completa queda guardada en cada `.npz`, así que se pueden recalcular otras cantidades sin propagar.

**Convergencia:** ω_p = 11.98, N = 20 → 26: P_c(α_eff) 0.998621 → 0.998765 y P_c(viejo) 0.997407 → 0.997551, es decir **Δ = 1.4e-4**.
N = 20 subestima P_c en 1.4e-4 en el pico. V2 mostró que N = 22 ya coincide con N = 28 a <1e-5. No altera la forma de la curva ni el FWHM.

**Validaciones (111 ρ):** |Tr ρ − 1| ≤ 4.4e-16, ‖ρ − ρ†‖ = 0, mínimo autovalor ≥ 1.2e-13.
