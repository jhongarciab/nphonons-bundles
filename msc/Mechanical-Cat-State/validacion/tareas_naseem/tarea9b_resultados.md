# Tarea 9, Paso B — Barrido ε/κ con muestreo estroboscópico

Script: `tarea9b_pasoB.py`. Datos: `tarea9_pasoB.npz`. Nb=16 (completo),
N_eff=30, τ_max=60, estroboscópico (T_m=2π/ω_m).

## Tabla

| ε/κ | n_ss(full) | t90(full) | dn/dt_final | p_e_ss | n_ss B(2g) | t90 B | full/B |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.02030 | 0.022 | −4.56e-5 | 0.00001 | 0.00100 | 32.648 | 20.24× |
| 0.05 | 0.02423 | 0.026 | −4.74e-5 | 0.00003 | 0.00505 | 1.745 | 4.79× |
| 0.10 | 0.03835 | 0.041 | −5.07e-5 | 0.00009 | 0.01944 | 1.574 | 1.97× |
| 0.20 | 0.09376 | 0.101 | −5.87e-5 | 0.00032 | 0.07561 | 1.508 | 1.24× |
| 0.40 | 0.29661 | 0.323 | −7.45e-5 | 0.00105 | 0.28108 | 1.397 | **1.055×** |
| 0.72 | 0.77048 | 0.897 | −5.49e-5 | 0.00208 | 0.76266 | 1.163 | **1.010×** |
| 1.44 | 1.90468 | 6.882 | +2.10e-5 | 0.00224 | 1.92872 | 0.918 | **0.988×** |

## Comparación con Tarea 6b (muestreo genérico)

| ε/κ | dn/dt (6b, genérico) | dn/dt (9b, estroboscópico) | Mejora |
|---|---|---|---|
| 0.02 | −1.15e-2 | −4.56e-5 | **252×** |
| 0.40 | −8.95e-3 | −7.45e-5 | **120×** |
| 1.44 | −1.70e-2 | +2.10e-5 | **810×** |

**El muestreo estroboscópico reduce dn/dt entre 120× y 810×** en todo el
rango de ε escaneado — confirma sólidamente que el aliasing del
desplazamiento polarónico β(t) era la causa dominante de la deriva
espuria detectada en la Tarea 6b.

## Confirmación del régimen de validez (ε≳0.4κ)

Para ε/κ ∈ {0.4, 0.72, 1.44}: razón full/B = 1.055, 1.010, 0.988 —
**acuerdo dentro de 1-6%**, consistente con y reforzando la conclusión de
la Ronda 2 anterior (g_eff=2g), ahora con una metodología libre de
aliasing y con dn/dt genuinamente pequeño (confirmando estado
estacionario real, no solo aparente).

## Punto abierto: piso residual en ε≲0.2κ

Persiste un "piso" en n_ss(full) (~0.02-0.04) que **no se explica por
aliasing** (dn/dt ya es pequeño, ~5e-5, la deriva rápida desapareció) ni
coincide con la predicción del modelo efectivo B en ese régimen (que da
valores mucho menores, dominados solo por el término de drive ε ya que
sus tasas G1_gx, G1_gx² son numéricamente minúsculas ~1e-5). El origen de
este piso —presente incluso con muestreo estroboscópico limpio— **no se
identificó completamente** en esta tarea; podría deberse a una
contribución de orden superior no capturada por las fórmulas Γ1∓/Γ2∓ de
la eliminación adiabática a este orden (p.ej. relacionado con el
corrimiento tipo Bloch-Siegert visto en la Tarea 7), pero **no invalida**
la conclusión principal (g_eff=2g), que se sostiene con fuerza en el
régimen ε≳0.4κ, el relevante para la generación del cat state (fig1/fig2
del paper usan ε=1.44κ, dentro de este régimen validado).

## Nota metodológica: throttling de background

Se detectó que lanzar estos scripts vía backgrounding explícito de la
herramienta (parámetro `run_in_background`) provoca un throttling severo
del proceso en esta máquina (~1-3% de duty cycle real, cientos de veces
más lento que ejecución en primer plano). Ejecutar en primer plano (con
paso automático a segundo plano solo si excede el timeout de la
herramienta) evita el problema — así se completó este script en 8m24s.

## Archivos generados

- `tarea9_pasoB.npz`, `tarea9b_output.log`
