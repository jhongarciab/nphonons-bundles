# Tarea 6a — Saturación del qubit: línea base (fig2, Ω=4g)

Script: `tarea6a_fig2_saturacion.py`. Datos: `tarea6a_resultados.npz`.
Tolerancias: `atol=1e-10, rtol=1e-8` (según lo pedido esta ronda).

## Validación

| Chequeo (umbral) | Resultado |
|---|---|
| Traza | tr=1.000000 en todos los instantes muestreados — **OK** |
| Hermiticidad | 0 en todos los instantes muestreados — **OK** |
| Positividad (umbral −1e-9) | mín. autovalor oscilador ~ −1e-23 a −1e-24 (ruido de máquina); qubit siempre ≥ +8e-4 — **OK, umbral cumplido** con las tolerancias estrictas |

Con `atol=1e-10, rtol=1e-8` la violación de positividad observada en la
ronda anterior (Tareas 1-3, con tolerancias por defecto) **desaparece**:
confirma que aquella violación era ruido numérico del integrador, no un
problema físico.

## Estado estacionario

- `dn/dt` en t_final=60 (κ⁻¹) = **−9.36e-3** (⟨n⟩=1.8961); promedio de
  `|dn/dt|` en el último 10% del tiempo = 6.03e-3. Pequeño pero no
  exactamente cero — el sistema está *cerca* del estado estacionario pero
  aún relajando muy lentamente en t=60. Se toma el promedio de los últimos
  10 puntos como estimador de p_e_ss, n_ss.

## Comparación con la fórmula de saturación resonante

| Cantidad | Valor |
|---|---|
| p_e_ss (medido) | **0.002122** |
| p_e_ss (fórmula ε²/(2ε²+κ²/4), ε=Ω_full=4g) | **0.471573** |
| Diferencia relativa | **99.55%** |
| \|⟨σ₋⟩\|_ss (medido) | 0.035983 |
| n_ss (medido) | 1.898468 |

## Interpretación

La fórmula de saturación resonante de un qubit de dos niveles **no aplica
aquí**: predice p_e_ss≈0.47 (saturación casi completa), pero la dinámica
real da p_e_ss≈0.002 (qubit casi enteramente en su estado base). La razón
es que esa fórmula asume un qubit aislado, driveado y disipativo, sin
ningún otro acoplamiento. En este sistema el qubit está acoplado al
oscilador con **g_x=6κ y g_z=60κ**, ambos mucho mayores que κ y que el
propio drive (ε=4g=1.44κ). Esta hibridización fuerte qubit-oscilador
domina por completo sobre la física de saturación de un qubit libre:
el decaimiento colectivo del sistema acoplado (vía κ actuando sobre un
qubit fuertemente vestido por el oscilador) mantiene la población
excedida "desnuda" del qubit mucho más baja que la predicción ingenua.

## Conclusión Tarea 6a

Confirmado: **la fórmula de saturación simple no describe este régimen**
(discrepancia de 2 órdenes de magnitud). Cualquier argumento que use esa
fórmula para estimar el punto de saturación del qubit en este sistema
necesita, como mínimo, incluir el acoplamiento fuerte a la mecánica
(dressed-state picture), no solo κ y ε.
