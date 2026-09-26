# Ronda 16 — Tarea 47 (cierres finales)

Tablas: `tarea47_resultados.md` ((a), (c), (d)), `tarea47b_fig2_resultados.md` (Fig. 2 de Liu), `tarea47b_task46_resultados.md` (Tarea 46 repetida).
Código: `tarea47_analisis.py`, `tarea47b_fig2.py`, `tarea47b_run.py`, `run_t47*.sh`, `../validacion/run_t47a.sh`; `tarea46.py` acepta `dscale`; `tarea45d_worker.py` acepta variantes `*_retuned`.

## (a) Rama interior: irrelevante para la dinámica
Peso de |1.3α⟩|g⟩, |4⟩|g⟩ y |α⟩|e⟩ sobre la rama interior real (autovectores izquierdos, como en la Tarea 35), α²=2, resonancia vestida:
| Γ₂/κ | N | Re de la rama | pesos (coh / Fock / exc) |
|---|---|---|---|
| 0.5 | 26 / 32 | 0.1244 / 0.1261 | ≤1e-14 en los tres |
| 1.0 | 26 / 32 | 0.0896 / 0.0878 | ≤1.5e-13 |
| 3.0 | 26 / 32 | 0.1313 / 0.1249 | ≤1.1e-14 |
Todos <1e-4 (de hecho 1e-13 a 1e-16, nivel de ruido numérico): **la rama es irrelevante para estos estados iniciales**. Junto con la Tarea 35(c) (el retorno al código sigue la rama lenta estática, 0.175/0.23/0.26–0.29), la brecha *dinámicamente relevante* de Floquet es la de los modos estables (≈0.19–0.23), aunque la brecha espectral formal a Γ₂/κ≳0.25 sigue sin determinarse por la rama no convergida (ver Ronda 15).

## (b) Liu et al.: convención y Fig. 2
- Fuente: arXiv:2501.08675v2 (PDF leído). g_eff=4g_xg_z/ν=1.4124 (paper 1.41); Ec. (12): γD[σ̃₋] + (γ_φ/2)D[σ̃_z] (el γ_φ/4·L[σ̃_z] del paper); Fig. 2 (γ=2π·16 MHz, ε_p=2π·3.53 MHz, κ=0) declara **α=1.58** en el estacionario.
- **Resultado: α²=ε_p/g_eff=2.50 (α=1.58), no (ε_p/2)/g_eff.** Con la Ec. (11) tal cual (deriva ε_p(σ̃₊+σ̃₋)) el estacionario da ⟨m²⟩=2.499 y F frente a α=1.58 crece 0.34/0.57/0.85/0.95 en γt=7.5/15/30/45 (γ_φ=0); el gato impar (|1⟩|↓⟩) llega a 0.73/0.92/0.994/1.000. Con deriva ε_p/2 el estacionario es |α|²=1.25 y F frente a α=1.58 solo llega a 0.69. Nuestra F con γ_φ/γ=1 (0.65 a γt=45) baja más con γ_φ que la Fig. 2 (que parece casi uniforme en γ_φ/γ): el estacionario no depende de γ_φ (|↓⟩ es oscuro de D[σ̃_z]), solo el transitorio.
- **Consistencia con la Ec. (9):** con ε_p cos(ω_pt)σ̃_x la RWA da deriva ε_p/2 (Tarea 46: α²=1.25). Para que (9) reproduzca (10)–(11) y α=1.58, el coeficiente del coseno debe ser 2ε_p (coherente con ε_p=√2ε/2 y (σ_x+σ_z)=√2σ̃_x en su Ec. 8). Convención correcta usada: **2ε_p cos(ω_pt)σ̃_x, α²=2.5**.
- **Tarea 46 repetida (N=20, γt=200; convergencia N=20→28: ≤2e-3):**
| versión | ω_p | máx \|⟨a²⟩\| (analít. 2.50) | P_c | F | paridad | P_e |
|---|---|---|---|---|---|---|
| (i) γD[σ̃₋] | ν | 0.70 | 0.48 | 0.29 | +0.41 | 0.08 |
| (i) | re-sint. 33.767 | 1.27 | 0.66 | 0.35 | +0.14 | 0.09 |
| (ii) γD[σ₋] bare | ν | 0.81 | 0.50 | 0.26 | +0.19 | 0.13 |
| (ii) | re-sint. | 1.15 | 0.55 | 0.28 | +0.12 | 0.20 |
  (P_c del vacío = 0.16.) **Con la convención correcta tampoco se forma el gato en la Ec. (9) completa con γ=16, κ/ω=0.9** (F≤0.35 frente a 0.95 del efectivo); la fórmula de la tasa de paridad 2|α|²(Γ₁₋+Γ₁₊)=3.00 (con κ²/4; 3.55 sin) falla 96–99% (medida 0.025–0.130). La discrepancia entre el modelo completo y su modelo efectivo (Ec. 11) en ese régimen es de fondo: la eliminación adiabática no es válida a κ/ω≈0.9.

## (c) 'nocounter' re-sintonizado (g_x=0.05)
| g_z/κ | κ₁ respecto a full | previsto | P_c máx |
|---|---|---|---|
| 5 | 0.882 | 0.900 | 0.99936 (full 0.99930) |
| 12 | 0.898 | 0.900 | 0.99589 (full 0.99517) |
Confirmado: sin los términos de 3ω κ₁ baja ≈10% cuando se re-sintoniza con δ_osc=−g_x²/w (sin la re-sintonía subía 24% por la desintonía). Los contrarrotantes no explican la caída de P_c (0.9959 vs 0.9952 en g_z/κ=12).

## (d) Ancho de la resonancia: FWHM = c·G, G=2g_xg_z/w=√(κκ₂)/2 (7 valores de κ₂/κ)
| κ₂/κ | 0.03 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 1 |
|---|---|---|---|---|---|---|---|
| c=FWHM/G | 2.45 | 2.67 | 2.88 | 3.05 | 3.10 | 3.12 | 2.83 |
**c = 2.87 ± 0.25 (8.6%; rango 2.45–3.12)**; ajuste libre FWHM ∝ κ₂^0.55 = G^1.10 (casi lineal en G); c crece con κ₂/κ hasta ~0.3–0.5 y baja a 1. Confirma FWHM∝G (∝√κ₂) frente a la escala ingenua ∝κ₂. Con d=12 fijo (como la Tarea 40), no se separó el efecto de la desintonía qubit–drive.

## Caveats
- (a) mide relevancia por peso inicial (autovectores izquierdos) en los tres estados pedidos; no descarta que otras condiciones iniciales excitaran la rama.
- (b) el efectivo (Ec. 11) se simuló con N=24; la Tarea 46 completa con N=20 (convergencia a N=28 ≤2e-3 en |⟨a²⟩|).
- (d) c depende algo de κ₂/κ (±9%), y el ancho se define sobre P_c a κ₂t=120.
