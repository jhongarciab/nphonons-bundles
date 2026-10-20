# Revisión de F16 y F17 (16 oct 2026)

F17 muestra una recuperación sustancial con acoplador conmutable:
κ_in/κ=1,3,5,10,20 da F|h=0.617,0.781,0.832,0.878,0.904
(modelo reducido, χ=1, T=2000). La probabilidad de éxito queda cerca
de 0.77 para κ_in≥κ, de modo que la mejora procede principalmente de
preservar la coherencia mientras llega el segundo bin. Este es un
resultado del protocolo, no solo una eficiencia de carga alta.

La duración del segundo paquete es
\(t_L=[4+\ln(0.98)]/\kappa_{\rm in}\). El intervalo sin protección
entre el final de la primera escritura y el final de la segunda es
aproximadamente \(t_L+\tau_{12}+d_2\): 648/ω_b para 10κ y 449/ω_b
para 20κ. Esto explica por qué 10κ da 0.878 y **20κ**, no 10κ,
cruza F|h≈0.9 en el modelo real con estos pulsos.

## Error encontrado y corregido antes de combinar modelos

En `src/f16_full3dot.py`, el corte del paquete real era `TL-20`
independientemente de κ_in. Cuando κ_in=10κ deja aproximadamente 18 %
en la fuente, mientras para κ_in=κ deja aproximadamente 2 %. Se corrigió
a `TL+log(0.98)/κ_in`, la misma regla de F17, y se sincronizó a
`pcjhon`. El resultado previo de F16 con carga ideal no se afecta.

Se lanzó en `pcjhon` el punto conjunto Δxx=−9ω_b,
κ_in=0.01ω_b, T=2000/ω_b, con salida en
`data/f16_real_dxxm9_kin0p01.log`. No se debe multiplicar las ventajas
de F16 y F17 ni citar F|h=0.878 como resultado del modelo completo
antes de leer esta corrida.

Persisten la descarga ideal, el supuesto χ=1 y la auditoría del cambio
de marco rotante de F9. El modelo completo incluye estados de dos y
tres excitones, y por ello es el control necesario de fuga fuera del
sector usado en F17.
