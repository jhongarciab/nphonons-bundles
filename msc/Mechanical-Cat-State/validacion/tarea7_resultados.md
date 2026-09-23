# Tarea 7 — Medida directa de g_eff por Rabi de dos fotones

Script: `tarea7_rabi_dos_fotones.py`. Datos: `tarea7_resultados.npz`.

## Nota metodológica: corrección de etiquetado g/e

La primera corrida (proyectando `|g,2⟩=tensor(basis(Na,0),basis(Nb,2))` →
`|e,0⟩=tensor(basis(Na,1),basis(Nb,0))`, siguiendo el comentario de
`fig2/fig3_git_v*.py` que llama "ground" a `thermal_dm(Na,0)`) dio
**P_max ≈ 0** — ninguna transferencia de población. Se verificó
numéricamente que `sigmam()` mapea `basis(Na,0)` (sz=+1) → `basis(Na,1)`
(sz=-1), es decir **`basis(Na,0)` es físicamente el estado EXCITADO**
(mayor energía, +ω_q/2) y `basis(Na,1)` el estado BASE (-ω_q/2) — al
revés de como lo llaman los comentarios de los scripts originales (que
nunca verifican el signo de σ_z, solo usan el índice 0 como
"conveniencia"). Esto **no afecta las simulaciones disipativas de
fig1/fig2/fig4/fig5** (son insensibles a la etiqueta, solo importa la
dirección del decaimiento inducido por `sm`, que sí es física y correcta:
decae `basis(Na,0)`→`basis(Na,1)`), pero sí importa aquí porque se
necesita el par resonante correcto para ver la oscilación de Rabi de dos
fotones. Con el par correcto —inicial `|base,2⟩`, objetivo
`|excitado,0⟩`— **sí aparece** oscilación clara y de alto contraste
(P_max≈0.95–0.99, ver abajo).

## Resultados (g_z escalado ×{0.5, 1, 2}, g_x fijo)

| g_z ×f | g (teoría, ∝f) | P_max | Ω_gen (FFT) | Ω_R (corregido) | δ_eff | pred(g)=2√2g | pred(2g)=2√2·2g | Ω_R/pred(g) | Ω_R/pred(2g) |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.180 | 0.9744 | 1.0463 | 1.0328 | 0.1676 | 0.5091 | 1.0182 | **2.029** | **1.014** |
| 1.0 | 0.360 | 0.9852 | 1.9880 | 1.9733 | 0.2419 | 1.0182 | 2.0365 | **1.938** | **0.969** |
| 2.0 | 0.720 | 0.9567 | 3.9760 | 3.8889 | 0.8276 | 2.0365 | 4.0729 | **1.910** | **0.955** |

## Hallazgo central

**La frecuencia de Rabi de dos fotones medida coincide con 2√2·(2g), no
con 2√2·g** — razón Ω_R/pred(2g) = 0.955–1.014 (dentro de ~5%) en los tres
casos, mientras que Ω_R/pred(g) ≈ 1.91–2.03 (el doble, consistentemente).

Esto es una **medición limpia, no perturbativa, sin disipación ni
suposiciones markovianas** (evolución unitaria exacta del Hamiltoniano
completo, sin necesidad de eliminar adiabáticamente el qubit) — y da
evidencia **directa y contundente en contra de g_eff=g y a favor de
g_eff=2g**, exactamente lo opuesto a lo sugerido (con métodos más
indirectos y confundidos por efectos disipativos/de acoplamiento fuerte)
en las Tareas 2 y 3 de la ronda anterior, cuyas conclusiones el usuario ya
señaló como no concluyentes.

## Escala g_eff ∝ g_z

g_eff medido (=Ω_R/2√2) dividido por el factor de escala de g_z:
`[0.7303, 0.6976, 0.6875]` — variación relativa 6.08%. Razonablemente
consistente con proporcionalidad lineal (g_x fijo), con una desviación
sistemática pequeña que crece con g_z, correlacionada con el
desplazamiento de resonancia (ver abajo).

## Desplazamiento de resonancia (tipo Bloch-Siegert/Lamb)

P_max no llega exactamente a 1, y δ_eff crece con la fuerza de
acoplamiento: 0.168 → 0.242 → 0.828 (para g_z×0.5, ×1, ×2). Esto es
consistente con un corrimiento de tipo Bloch-Siegert (contribución de los
términos "contra-rotantes" no eliminados en el Hamiltoniano completo, que
crece con el acoplamiento) que desintoniza ligeramente el par resonante y
reduce el contraste de la oscilación, además de sesgar levemente el ajuste
lineal g_eff∝g_z hacia arriba a menor g_z (la desintonía relativa es mayor
cuando g es pequeño comparado con... realmente aquí crece en términos
absolutos con g, pero el efecto relativo en Ω_R es más notorio para
valores más grandes, visible en el corrimiento Ω_R/pred(2g) de 1.014 a
0.955 al aumentar g_z).

## Conclusión Tarea 7

A diferencia de las Tareas 2 y 3 (indirectas, con disipación y/o estados
mixtos, confundidas por acoplamiento fuerte de un fonón), esta medición
**directa y coherente** del elemento de matriz de dos fotones **sí
respalda la hipótesis del factor 2**: g_eff = 2g, consistente con la
Ec. (9) del paper `-2g(σ_+-σ_-)(a†²-a²)`, no con la normalización
implícita en Ec. (11) `+g(σ_+-σ_-)(a†²-a²)`. Este resultado apunta a que
el factor 2 faltante es real **en el acoplamiento coherente g_eff**,
aunque (ver Tarea 6) esto es una pregunta distinta de si el **drive
efectivo ε** usado en el modelo de squeezing está correctamente
normalizado respecto al drive real Ω del qubit — ambas cuestiones deben
tratarse por separado en la conclusión final.
