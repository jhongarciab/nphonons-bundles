# Tarea 11 — Dominio de validez (|α|²=ε/g_eff=2, escaneo en Γ₂/κ)

Script: `tarea11_dominio_validez.py`. Datos: `tarea11_resultados.npz`.
Muestreo estroboscópico, ventana [0, 10/Γ₂] cada caso, g_eff=2g (Tareas 7-8).

**Nota**: se omitió g_z×0.125 (Γ₂/κ≈0.032, τ_max≈309 — costo excesivo,
decisión explícita del usuario para acotar tiempo de esta ronda).

## Tabla

| g_z×f | g_eff | Γ₂ (=Γ₂/κ) | τ_max=10/Γ₂ | t90(full)/t90(B) | F_min | F_integrada (0,10/Γ₂) |
|---|---|---|---|---|---|---|
| 0.25 | 0.180 | **0.1296** | 77.16 | 1.127 | **0.9015** | 0.9595 |
| 0.50 | 0.360 | **0.5184** | 19.29 | 2.011 | **0.7319** | 0.9234 |
| 1.00 | 0.720 | **2.0736** | 4.82 | 3.174 | **0.4987** | 0.7583 |

## Tendencia clara: a menor Γ₂/κ, mejor acuerdo dinámico

F_min y F_integrada **mejoran monótonamente** al bajar Γ₂/κ:
Γ₂/κ=2.07→F_min=0.499; Γ₂/κ=0.518→F_min=0.732; Γ₂/κ=0.130→F_min=0.902.
Esto es consistente con la física esperada: la eliminación adiabática que
da lugar al modelo efectivo es una aproximación perturbativa en g/κ (o
g_eff/κ), válida en el límite de **acoplamiento débil relativo a κ**. Los
parámetros "de fábrica" del código (g_z, g_x tal como aparecen en
fig1/fig2/fig4/fig5) corresponden a Γ₂/κ=2.074 (fila g_z×1.0) — **el
régimen de PEOR acuerdo dinámico** de los tres probados (aunque el
estado estacionario, Tareas 6b/9b, sí se reproduce bien ahí).

También, t90(full)/t90(B) se aleja de 1 al aumentar Γ₂/κ (1.13→2.01→3.17):
el modelo efectivo consistentemente subestima el tiempo de subida real
del completo, y cada vez más al aumentar el acoplamiento.

## Validación

Traza/hermiticidad perfectas; positividad excelente en todos los casos
(mín. autovalor ~1e-17 a 1e-25, muy por dentro del umbral −1e-9).

## Respuesta a la pregunta: ¿a partir de qué Γ₂/κ se logra F_min>0.99?

**Ninguno de los tres casos probados alcanza F_min>0.99** (el mejor,
Γ₂/κ=0.130, da F_min=0.902). La tendencia (mejora de (1−F_min) por un
factor ~2.7 cada vez que Γ₂/κ baja por un factor 4) sugiere que el umbral
F_min>0.99 requeriría **Γ₂/κ notablemente menor que 0.13** — posiblemente
en el rango del caso omitido (Γ₂/κ≈0.032) o más abajo aún; extrapolando
groseramente la tendencia, el punto omitido probablemente da F_min en el
rango ~0.95-0.97, todavía por debajo de 0.99. **No se puede dar una cifra
exacta con los datos disponibles** — se necesitaría al menos un punto
adicional en Γ₂/κ≲0.03 (el caso costoso que se omitió) para acotarlo,
dejando esto como trabajo pendiente si se requiere el valor preciso.

## Conclusión Tarea 11

El modelo efectivo (g_eff=2g) reproduce el **estado estacionario** bien
en todo el rango de acoplamiento probado (Tareas 6b/9b), pero reproducir
la **dinámica completa** (F_min alto) exige ir a un régimen de
acoplamiento sustancialmente más débil (Γ₂/κ≪1) que el usado en las
figuras publicadas del paper (Γ₂/κ≈2.07). Esto es una limitación de
validez real del modelo efectivo como descripción de la dinámica
*transitoria*, no solo una cuestión de la normalización de g_eff (que ya
se estableció correcta en Tareas 7-8).

## Archivos generados

- `tarea11_resultados.npz`, `tarea11_output.log`
