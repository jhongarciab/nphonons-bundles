# Ronda 12 — Tareas 35-36: cuádruplete de Floquet y umbral térmico

Tablas completas: `tarea35_36_resultados.md`; figuras: `tarea35_retorno.png`, `tarea36_eta_a2{2,3}.png`.
Código: `tarea35_worker.py`, `tarea36_worker.py`, `tarea35_36_analisis.py`, `run_t35_36.sh`.

## Tarea 35 — el cuádruplete NO es físico (corrige la conclusión de la Ronda 11)
(a) **No converge en N.** Re y |Im| del cuádruplete cambian sistemáticamente con N=20→26→32:
| Γ₂/κ | Re | \|Im\| | \|Im\|/N |
|---|---|---|---|
| 0.13 | 0.1681→0.1629→0.1609 | 6.98→9.10→11.23 | 0.349/0.350/0.351 |
| 0.3 | 0.1288→0.1143→0.1070 | 13.04→16.92→20.81 | 0.652/0.651/0.650 |
| 1.0 | 0.1326→0.1044→0.0859 | 37.2→47.9→58.6 | 1.858/1.843/1.832 |
**|Im| ∝ N** (constante por Γ₂): es un modo del borde del truncamiento de Fock, no un modo del sistema. La brecha de Floquet de la Tarea 27/33 (que en Γ₂/κ≥0.13 ES ese cuádruplete) cae con N (0.168→0.161 en 0.13; 0.129→0.107 en 0.3) y **su máximo y su saturación en 0.10-0.17 no son fiables**. (En Γ₂/κ=1 la brecha a N=20/26 es una rama real de 0.09, pero a N=32 pasa a ser el cuádruplete, 0.086.)
(b) 2π/T_m=ω_m=1000 (unid. de κ). |Im|=7-37 (0.7-3.7% de 1000): el cuádruplete está cerca del múltiplo 0, no plegado; su Im crece con N, no es un alias.
(c) **Dinámica:** desde |1.3α⟩|g⟩, |4⟩|g⟩ y |α⟩|e⟩ la fuga del código decae con tasa tardía 0.175 (Γ₂/κ=0.13), 0.227-0.236 (0.3), 0.26-0.29 (1.0), es decir, sigue la rama lenta estática (paso 5: 0.177/0.219/0.234), no el cuádruplete (Re 0.168/0.129/0.133). La fracción del estado inicial sobre el cuádruplete (autovectores izquierdos) es 1e-6…1e-15. **Conclusión: el máximo no es físicamente relevante**; la brecha física es la del modelo estático, monótona y saturada en ~0.23-0.25.

## Tarea 36 — umbral térmico (Γ₂/κ=0.13, propagador 1e-15/1e-13)
Sesgo η=γ_pf/γ_bf (γ_bf ≈ 0.05·n_q κ, ya η≈145 en n_q=1e-4 frente a ~1200 sin qubit térmico):
| α² | η | n_q* | f_min 10 mK | 20 mK | 50 mK |
|---|---|---|---|---|---|
| 2 | 100 | 1.5e-4 | 1.83 GHz | 3.67 | 9.17 |
| 2 | 10 | 1.7e-3 | 1.33 | 2.66 | 6.66 |
| 2 | 1 | 2.1e-2 | 0.81 | 1.61 | 4.03 |
| 3 | 100 | 4.2e-4 | 1.62 | 3.24 | 8.09 |
| 3 | 10 | 4.6e-3 | 1.12 | 2.24 | 5.61 |
| 3 | 1 | 9.1e-2 | 0.52 | 1.04 | 2.59 |
(f_min = (k_B T/h)·ln(1+1/n_q*); n_q=1/(e^{hf/kT}−1).) α²=3 aguanta ~3× más n_q que α²=2 para el mismo η. P(código) y |⟨a²⟩| bajan de forma suave (α²=2: P=0.985 y |⟨a²⟩|=1.98 en n_q=8.6e-3).

## Validaciones y caveats
- **Falla la tolerancia de hermiticidad** (‖ρ−ρ†‖<1e-10) antes de hermitizar en 5 puntos: α²=2 con n_q=3.5e-3 (2e-10), 8.5e-3 (2e-10), 2.1e-2 (2e-10), 5.1e-2 (1e-10) y α²=3 con n_q=1e-4 (3e-10). Excesos ≤3×, limitados por la precisión del propagador con el autovalor casi degenerado del bit-flip; traza (≤4e-16) y autovalores de ρ (≥−8e-13) pasan. Los indicadores reportados (P(código), |⟨a²⟩|) no son sensibles a ese nivel.
- Convergencia N→N+6 en el punto más caliente (n_q=0.3): γ_pf 2e-4, γ_bf 1.7e-3, η 1.5e-3, P(código) 1.4e-4, |⟨a²⟩| 4e-3, **brecha 6.8%** (α²=2); α²=3: brecha 4.8%. Las tasas γ_pf, γ_bf y η convergen; la brecha no (mismo problema del truncamiento).
- **Corrección a la Ronda 11:** la conclusión "el máximo solo aparece con Floquet, luego es dependencia temporal" debe leerse ahora como "el máximo/saturación bajo 0.25 del completo procede de un modo espurio del borde de Fock del modelo de Floquet", no de un efecto físico. Falta verificar la brecha física con un criterio que excluya modos de borde (p. ej. peso en n>N−6) o N mayor. Las Tareas 21, 23, 27, 28, 33 usan esa brecha y deben revisarse.
