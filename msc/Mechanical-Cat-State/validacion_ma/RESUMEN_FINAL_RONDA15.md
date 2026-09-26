# Ronda 15 — Tareas 45 (cierre metodológico) y 46 (esquema de Liu et al.)

Tablas: `tarea45a_resultados.md` (umbral de borde, rama interior), `tarea45_resultados_bcde.md` ((b)–(e)), `tarea46_resultados.md`.
Figuras: `tarea45e_resonancias.png`. Código: `tarea45a.py`, `tarea45_analisis.py`, `tarea45d_worker.py`, `tarea45e_worker.py`, `tarea43_worker.py` (ahora con N_f y `noevo`), `tarea46.py`, `tarea46_run.py`, `ma2.py` (opciones `counter`, `gz_direct`, `pair`), `../validacion/audit_worker.py` (perfil en Fock), `run_t45.sh`, `run_t45d2.sh`, `run_t45e2.sh`, `run_audit2.sh`.

## 45(a) — sensibilidad del umbral de borde y criterio único
- **Tarea 38** (39 puntos, α²=2, N=20): la brecha del 5º modo cambia con el umbral en 13 de 39 puntos (14 puntos T21/T23: cambia en 6); con umbral ≤3e-2 coincide con la Tarea 38 publicada; con 0.1 reaparece una rama real de Re≈0.09–0.13. **Tarea 42** (α²=4, N=22): 7 de 22 celdas cambian con 1e-3/1e-2 (los modos físicos con g_x=0.2 tienen peso 0.006–0.016); con ≥3e-2 no cambia nada. Los pesos son bimodales en α²=4 (≤0.016 físicos, 1.0 espurios).
- **Auditoría de la rama ambigua** (N=20/26/32/38): la rama real interior (99% de la masa en n≤10, n medio≈2, peso de borde ≤5e-3 y decreciente) **no converge**: Γ₂/κ=3.0 → 0.1310, 0.1313, 0.1249, 0.1161 (−5%, −7% por paso); Γ₂/κ=1.6 → 0.0935, 0.0918, 0.0876; Γ₂/κ=1.0 → 0.0901, 0.0896, 0.0878. El modo de borde más lento cae en paralelo (0.170→0.102), probablemente hibridándose con ella.
- **Criterio único adoptado:** espurio si (i) peso de borde >0.5 o (ii) su autovalor no es estable al 0.3% a lo largo de ≥3 truncamientos (N, N+6, N+12); lo que no cumple (ii) sin cumplir (i) se reporta como *no resuelto*.
- **Conclusión revisada (rebaja la de la Tarea 38):** para Γ₂/κ≲0.2 la brecha de Floquet (0.022→~0.19) es estable en N (sin cambio); para Γ₂/κ≳0.25 **no está determinada**: cota superior ≈0.09–0.12 (rama interior sin converger) o ≈0.22 si esa rama es un artefacto de hibridación con el borde (0.22 coincide con el modelo estático, 0.234). El cuádruplete de |Im|∝N sigue siendo espurio; γ_pf, γ_bf y la resonancia no dependen de esto.

## 45(b) — filtro armónico N_f=2, 3, 4 (κ_f=0.3 y 1, N=10, α²=2, sin evolución temporal)
- N_f=3 y 4 dan resultados idénticos (converge en N_f=3); frente a N_f=2: κ₁ cambia −1.5% (κ_f=0.3) y −0.1% (κ_f=1); brecha (peso≤0.5) +2.8% y +0.4%. **N_f=2 es suficiente (≤3%).** (A N=10 los valores absolutos no valen: κ₁ sale 2.5× el de N=16.)

## 45(c) — convergencia de la Tarea 43 (κ_f=0.3, N=16→20, N_f=2)
- **κ₁ converge:** 2.3497e-8 → 2.3490e-8 (Δrel 3e-4). **La brecha no:** 5º modo 2.04e-3→1.65e-3 y con peso ≤0.5 2.22e-3→2.09e-3 (−6%). La pérdida de brecha con el filtro (4.2e-3 plano → 9.8e-4 con κ_f=0.1) es cualitativa, con ~6% de incertidumbre en N.

## 45(d) — origen de la caída de P_c (Tarea 42(a), g_x=0.05, g_z/κ=12; P_c máx base = 0.9952)
| variante | P_c máx (g_z/κ=12) | κ₁/κ₂ (previsto 4.82e-4) | P_c máx (g_z/κ=5, base 0.9993) |
|---|---|---|---|
| sin contrarrotantes (3ω) | 0.9958 | 5.83e-4 | 0.9966 |
| **sin g_z σ_z a, con intercambio de pares explícito** | **0.99998** | 4.92e-4 | **0.99996** |
| ambos | 0.9999 | 4.33e-4 | — |
- **Lo que restaura P_c es apagar el acoplamiento directo g_z σ_z a**; los contrarrotantes no lo restauran (los quitan sin efecto en g_z/κ=12 y empeoran g_z/κ=5). El intercambio de pares se introdujo como +|G|(σ₊a²+h.c.); con −|G| (signo erróneo, primer intento) se forma un gato real en ±2 en vez de ±2i (P_c=0.037).

## 45(e) — ancho de la resonancia vestida (Tarea 40 con g_x variable, d=12, κ₂t=120, N=22)
| κ₂/κ | FWHM (unid. κ₂) | FWHM (unid. κ) | pico |
|---|---|---|---|
| 0.03 | 7.08 | 0.2125 | s*=0 (P_c=0.9986) |
| 0.1 | 4.55 | 0.4551 | 0 (0.9985) |
| 0.3 | 2.83 | 0.849 | 0 (0.9984) |
| 1 | 1.41 | 1.413 | 0 (0.9975) |
- **Ley de escala: FWHM ≈ 0.30·κ₂^0.54 (κ=0.03 fijo), es decir, el ancho crece como ≈√κ₂ y no como κ₂** (FWHM/κ₂ de 1.4 a 7.1); el pico coincide con w_p*(g_x) a <0.01 κ₂ en todos los casos. Curvas asimétricas (cola derecha más larga). Con d=12 fijo el qubit no queda en resonancia con el drive al barrer w_p, lo que ensancha la curva respecto a la escala κ₂ (no se repitió con d=w_p).

## Tarea 46 — esquema de Liu et al. (Ec. 9, base vestida; ν=35.4, g_x=g_z=3.5355, ε_p=3.53, Δ_m=ν/2, γ=16, N=16, 2π·MHz=1)
- **No se forma un gato bien definido con esos parámetros.** |α|² analítico = (ε_p/2)/|2g_xg_z/w| = 1.25; |⟨a²⟩| máx 0.43 (ω_p=ν, disipación (i)), 0.71 (re-sintonizado (i)), 0.39/0.62 (versión (ii) física); a γt=200: P_c=0.71/0.81 (i) y 0.64/0.67 (ii), F≤0.62, paridad 0.65/0.39 (i) y 0.33/0.26 (ii). Con |α|²=1.25 el vacío ya tiene P_c=0.53, así que P_c alto no implica gato.
- La versión física (ii) (γD[σ₋] bare, coeficientes −0.146 σ̃₊ + 0.854 σ̃₋ + 0.354 σ̃_z) dobla P_e (0.09–0.13 vs 0.03–0.04) y baja F y la paridad respecto a (i). El re-sintonizado ω_p=2(ν/2+δ_osc)=33.767 (δ_osc=−0.817) mejora |⟨a²⟩| y P_c 30–60%.
- **Fórmula de la paridad:** Γ₁₋=0.530, Γ₁₊=0.069 (con κ²/4; sin él 0.638/0.071) ⇒ 2|α|²(Γ₁₋+Γ₁₊)=1.50 (1.77 sin κ²/4). Tasa medida γt∈[20,200]: 0.011–0.068, **la fórmula falla 95–99%** en κ/ω=0.90 (no hay gato, la paridad satura en 0.26–0.65 en vez de decaer a 0 con esa tasa). Incluir κ²/4 cambia la predicción 16%.
- Convergencia N=16→24 (versión (ii), re-sint., γt=200): 2e-4 en P_c y F, 1e-5 en paridad, 9e-5 en P_e, 2.6e-3 en |⟨a²⟩|. Validaciones: |Tr ρ−1| ≤ 4e-16, ‖ρ−ρ†‖ ≤ 1.3e-15, autovalores ≥ −2.3e-12 (cumplen).

## Caveats / validaciones pendientes
- Brecha de Floquet para Γ₂/κ ≳ 0.25 sin resolver (ver 45(a)); las brechas de las Tareas 42, 43 tienen 3–6% de incertidumbre en N.
- (e) usó d=12 fijo; no se separó el efecto de la desintonía del qubit.
- Tarea 46: el gato objetivo se definió con |α|² analítico y fase numérica; ε_p cos(ω_p t) se integró sin RWA en el marco de laboratorio.
