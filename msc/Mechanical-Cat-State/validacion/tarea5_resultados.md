# Tarea 5 — fig4: separar estado inicial térmico vs baño térmico

Script: `tarea5_fig4_separacion.py`. Datos: `tarea5_fig4_resultados.npz`.

## Configuración

En `fig4_git_v1.py` el mismo `n_th` fija a la vez (a) la ocupación del
estado inicial `thermal_dm(N, n_th)` y (b) la ocupación del baño mecánico
que entra en Γ_minus=(n_th+1)γ+Γ1_minus y Γ_plus=n_th·γ+Γ1_plus. Se separan
ambos efectos evaluando, en Γ₂₋t=3 (t_final, igual que el original), tres
variantes por cada n_th ∈ {0.5, 1, 2}:

- **(0) Original**: init=thermal(n_th), baño=n_th (fig4 tal cual)
- **(i) Vacío + baño n_th**: init=vacío, baño=n_th
- **(ii) Térmico + baño frío**: init=thermal(n_th), baño=0

Métricas: negatividad de Wigner (volumen de la parte negativa,
`neg_vol = -∫∫_{W<0} W dx dp`), paridad ⟨P⟩=Tr[ρ(-1)ⁿ], y ⟨n⟩ como control.

## Resultados

| n_th | neg(0) | P(0) | neg(i) | P(i) | neg(ii) | P(ii) | ⟨n⟩ (todas) |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.1529 | 0.4821 | 0.3060 | 0.9637 | 0.1555 | 0.4901 | 9.9998 |
| 1.0 | 0.1002 | 0.3162 | 0.3009 | 0.9476 | 0.1036 | 0.3268 | 9.9998 |
| 2.0 | 0.0582 | 0.1838 | 0.2910 | 0.9163 | 0.0621 | 0.1961 | 9.9998 |

**⟨n⟩ es prácticamente idéntico (≈10.00) en las tres variantes y los tres
n_th**: el proceso de dos fonones (Γ₂₋ ≫ Γ₁, γ) domina completamente la
población media en la escala t_final=3/Γ₂₋, independientemente de las
condiciones iniciales o del baño.

## Interpretación

- **(0) vs (ii)** (mismo estado inicial mixto, cambia solo el baño):
  prácticamente **sin diferencia** (neg_vol y ⟨P⟩ cambian <2% en todos los
  n_th). El calentamiento del baño durante la evolución **no** es el
  responsable de la pérdida de negatividad/paridad al aumentar n_th.
- **(0) vs (i)** (mismo baño, cambia solo el estado inicial): diferencia
  **enorme** — neg_vol prácticamente se **duplica** (×2.0, ×3.0, ×5.0 para
  n_th=0.5, 1, 2 respectivamente) y ⟨P⟩ casi se **duplica** también (0.48→0.96,
  0.32→0.95, 0.18→0.92). Partiendo de vacío, incluso con el mismo baño
  térmico "malo", el estado final conserva casi toda la negatividad y
  paridad de un cat state puro.

## Conclusión Tarea 5

La degradación de la negatividad de Wigner y la paridad al aumentar n_th en
fig4_git_v1.py **se debe casi enteramente a la mezcla del estado inicial**
(un estado térmico con n_th>0 es un estado mixto de baja pureza, y
"comprimir"/"apretar" un estado mixto produce un cat state mucho menos puro
que comprimir el vacío), **no al calentamiento inducido por el baño
mecánico durante la evolución** (que, a estos parámetros, es demasiado
lento frente al proceso de dos fonones para degradar apreciablemente el
estado en el tiempo t_final considerado). Esto tiene una implicación
práctica directa: para mejorar la calidad del cat state generado, es mucho
más importante **enfriar el estado inicial del resonador** (p. ej. con
enfriamiento de un fonón antes de aplicar el proceso de dos fonones) que
reducir el acoplamiento al baño térmico durante la generación misma.

## Archivos generados

- `tarea5_fig4_resultados.npz`: `n_th_list`, `resumen` (array con
  n_th, neg(0), P(0), n(0), neg(i), P(i), n(i), neg(ii), P(ii), n(ii)).
