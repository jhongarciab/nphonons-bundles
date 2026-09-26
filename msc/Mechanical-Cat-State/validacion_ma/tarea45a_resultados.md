# Tarea 45(a) — sensibilidad del umbral de borde

## Tarea 38 (α²=2, N=20; 5º modo por Re con resonancia vestida; 39 puntos: los 9 de la T27 y los 30 de la T28)

| Γ₂/κ | sin filtro | 1e-3 | 1e-2 | 3e-2 | 0.1 | 1e-3 + estabilidad N | ¿cambia con el umbral? |
|---|---|---|---|---|---|---|---|
| 0.0100 | 0.0223 | 0.0223 | 0.0223 | 0.0223 | 0.0223 | 0.0223 | no |
| 0.1300 | 0.1681 | 0.1682 | 0.1682 | 0.1682 | 0.1682 | 0.1682 | no |
| 0.3399 | 0.1260 | 0.2014 | 0.1560 | 0.1560 | 0.1560 | 0.2014 | SÍ |
| 0.3890 | 0.1239 | 0.2048 | 0.1425 | 0.1425 | 0.1425 | 0.2048 | SÍ |
| 0.4451 | 0.1228 | 0.2079 | 0.1305 | 0.1305 | 0.1305 | 0.2079 | SÍ |
| 0.4500 | 0.1227 | 0.2082 | 0.1296 | 0.1296 | 0.1296 | 0.2082 | SÍ |
| 0.5094 | 0.1199 | 0.2108 | 0.1199 | 0.1199 | 0.1199 | 0.2108 | SÍ |
| 0.5830 | 0.1109 | 0.2135 | 0.2135 | 0.1109 | 0.1109 | 0.2135 | SÍ |
| 0.6672 | 0.1033 | 0.2160 | 0.2160 | 0.1033 | 0.1033 | 0.2160 | SÍ |
| 0.7635 | 0.0973 | 0.2183 | 0.2183 | 0.0973 | 0.0973 | 0.2183 | SÍ |
| 0.8400 | 0.0940 | 0.2198 | 0.2198 | 0.2198 | 0.0940 | 0.2198 | SÍ |
| 0.8738 | 0.0929 | 0.2204 | 0.2204 | 0.2204 | 0.0929 | 0.2204 | SÍ |
| 1.0000 | 0.0901 | 0.2224 | 0.2224 | 0.2224 | 0.0901 | 0.2224 | SÍ |
| 1.6000 | 0.0935 | 0.2284 | 0.2284 | 0.2284 | 0.0935 | 0.2284 | SÍ |
| 3.0000 | 0.1310 | 0.2343 | 0.1310 | 0.1310 | 0.1310 | 0.2343 | SÍ |

Puntos donde cambia con el umbral: 13 de 39 (se muestran los que cambian y tres de referencia).

### T21/T23 (marco base, confinamiento = menor Re entre modos 'n')

| Γ₂/κ | sin filtro | 1e-3 | 1e-2 | 3e-2 | 0.1 | ¿cambia? |
|---|---|---|---|---|---|---|
| 0.0100 | 0.0152 | 0.0152 | 0.0152 | 0.0152 | 0.0152 | no |
| 0.0188 | 0.0309 | 0.0309 | 0.0309 | 0.0309 | 0.0309 | no |
| 0.0299 | 0.0488 | 0.0488 | 0.0488 | 0.0488 | 0.0488 | no |
| 0.0355 | 0.0568 | 0.0568 | 0.0568 | 0.0568 | 0.0568 | no |
| 0.0669 | 0.1016 | 0.1016 | 0.1016 | 0.1016 | 0.1016 | no |
| 0.1262 | 0.1609 | 0.1609 | 0.1609 | 0.1609 | 0.1609 | no |
| 0.1296 | 0.1627 | 0.1627 | 0.1627 | 0.1627 | 0.1627 | no |
| 0.2378 | 0.1554 | 0.1917 | 0.1917 | 0.1917 | 0.1917 | no |
| 0.4481 | 0.1285 | 0.2085 | 0.1285 | 0.1285 | 0.1285 | SÍ |
| 0.5184 | 0.1171 | 0.2114 | 0.2114 | 0.1171 | 0.1171 | SÍ |
| 0.8446 | 0.0914 | 0.2197 | 0.2197 | 0.2197 | 0.0914 | SÍ |
| 1.5918 | 0.0890 | 0.2279 | 0.2279 | 0.2279 | 0.0890 | SÍ |
| 2.0736 | 0.0991 | 0.2306 | 0.2306 | 0.0991 | 0.0991 | SÍ |
| 3.0000 | 0.1244 | 0.2339 | 0.1244 | 0.1244 | 0.1244 | SÍ |

Cambian 6 de 14.

## Tarea 42 (α²=4, N=22; 5º modo con peso de borde ≤ umbral)

| celda | sin filtro | 1e-3 | 1e-2 | 3e-2 | 0.1 | ¿cambia? | pesos de borde k=4..8 |
|---|---|---|---|---|---|---|---|
| a_gx0.05_gz12 | 3.849e-03 | 3.849e-03 | 3.849e-03 | 3.849e-03 | 3.849e-03 | no | 0.000, 0.000, 1.000, 1.000, 1.000 |
| a_gx0.05_gz2 | 5.064e-04 | 5.064e-04 | 5.064e-04 | 5.064e-04 | 5.064e-04 | no | 0.000, 0.000, 0.000, 0.000, 0.000 |
| a_gx0.05_gz3 | 1.033e-03 | 1.033e-03 | 1.033e-03 | 1.033e-03 | 1.033e-03 | no | 0.000, 0.000, 0.000, 0.000, 0.000 |
| a_gx0.05_gz5 | 2.153e-03 | 2.153e-03 | 2.153e-03 | 2.153e-03 | 2.153e-03 | no | 0.000, 0.000, 0.000, 1.000, 1.000 |
| a_gx0.05_gz8 | 3.449e-03 | 3.449e-03 | 3.449e-03 | 3.449e-03 | 3.449e-03 | no | 0.000, 0.000, 1.000, 1.000, 1.000 |
| b_gx0.2_gz12 | 4.599e-03 | 4.599e-03 | 4.599e-03 | 4.599e-03 | 4.599e-03 | no | 0.000, 0.000, 0.000, 0.000, 0.000 |
| b_gx0.2_gz2 | 4.253e-04 | nan | 5.173e-04 | 4.253e-04 | 4.253e-04 | SÍ | 0.016, 0.006, 0.006, 0.006, 0.006 |
| b_gx0.2_gz3 | 9.768e-04 | nan | 9.768e-04 | 9.768e-04 | 9.768e-04 | SÍ | 0.009, 0.002, 0.002, 0.002, 0.002 |
| b_gx0.2_gz5 | 2.100e-03 | 2.443e-03 | 2.100e-03 | 2.100e-03 | 2.100e-03 | SÍ | 0.004, 0.001, 0.001, 0.001, 0.001 |
| b_gx0.2_gz8 | 3.420e-03 | 3.935e-03 | 3.420e-03 | 3.420e-03 | 3.420e-03 | SÍ | 0.001, 0.000, 0.000, 0.000, 0.000 |
| c_w4_gx0.03 | 1.897e-03 | 1.897e-03 | 1.897e-03 | 1.897e-03 | 1.897e-03 | no | 0.000, 1.000, 1.000, 0.000, 0.000 |
| c_w4_gx0.05 | 3.220e-03 | 3.220e-03 | 3.220e-03 | 3.220e-03 | 3.220e-03 | no | 0.000, 1.000, 1.000, 1.000, 1.000 |
| c_w4_gx0.1 | 4.111e-03 | 4.111e-03 | 4.111e-03 | 4.111e-03 | 4.111e-03 | no | 0.000, 0.000, 0.000, 0.000, 0.000 |
| c_w4_gx0.2 | 1.892e-03 | 2.395e-03 | 1.892e-03 | 1.892e-03 | 1.892e-03 | SÍ | 0.006, 0.001, 0.001, 0.001, 0.001 |
| c_w6_gx0.03 | 1.038e-03 | 1.038e-03 | 1.038e-03 | 1.038e-03 | 1.038e-03 | no | 0.000, 0.000, 0.000, 0.000, 0.000 |
| c_w6_gx0.05 | 2.153e-03 | 2.153e-03 | 2.153e-03 | 2.153e-03 | 2.153e-03 | no | 0.000, 0.000, 0.000, 1.000, 1.000 |
| c_w6_gx0.1 | 3.594e-03 | 3.594e-03 | 3.594e-03 | 3.594e-03 | 3.594e-03 | no | 0.000, 0.000, 0.000, 0.000, 0.000 |
| c_w6_gx0.2 | 2.100e-03 | 2.443e-03 | 2.100e-03 | 2.100e-03 | 2.100e-03 | SÍ | 0.004, 0.001, 0.001, 0.001, 0.001 |
| c_w8_gx0.03 | 6.316e-04 | 6.316e-04 | 6.316e-04 | 6.316e-04 | 6.316e-04 | no | 0.000, 0.000, 0.000, 0.000, 0.000 |
| c_w8_gx0.05 | 1.469e-03 | 1.469e-03 | 1.469e-03 | 1.469e-03 | 1.469e-03 | no | 0.000, 0.000, 0.000, 0.000, 0.000 |
| c_w8_gx0.1 | 3.027e-03 | 3.027e-03 | 3.027e-03 | 3.027e-03 | 3.027e-03 | no | 0.000, 0.000, 0.000, 0.000, 0.000 |
| c_w8_gx0.2 | 2.151e-03 | 2.453e-03 | 2.151e-03 | 2.151e-03 | 2.151e-03 | SÍ | 0.003, 0.000, 0.000, 0.000, 0.000 |

Cambian 7 de 22 celdas.

## Criterio único adoptado

**Espurio si (i) peso de borde > 0.5, o (ii) su autovalor cambia >0.3% en Re (>1%+0.01 en Im) al pasar de N a N+6.** Razones: el peso de borde separa limpiamente los modos de borde (≈1.0) de los físicos (≤0.05) cuando α²=4 (Tarea 42); pero en α²=2 hay una rama real (Re≈0.09 en Γ₂/κ≥0.5) con peso intermedio (0.04 a N=20, 0.014 a N=26) que un umbral de peso clasifica de forma inconsistente entre N y N+6; su Re deriva con N (0.0901→0.0896→0.0878 para N=20/26/32) y su overlap con n crece (3.16→3.51→3.78), mientras la rama física de 0.2224 no cambia en 4 cifras. Por eso el criterio (ii) es necesario cuando hay dos truncamientos; con un solo N (Tarea 42, peso bimodal) el (i) basta.

| Γ₂/κ | 5º modo original | brecha con el criterio adoptado | (peso ≤ 0.1 solo) | ¿la rama 0.09 se excluye? |
|---|---|---|---|---|
| 0.0100 | 0.0223 | 0.0223 | 0.0223 | — |
| 0.1300 | 0.1681 | 0.1682 | 0.1682 | — |
| 0.2595 | 0.1330 | 0.1939 | 0.1870 | sí |
| 0.2970 | 0.1290 | 0.1978 | 0.1708 | sí |
| 0.3399 | 0.1260 | 0.2014 | 0.1560 | sí |
| 0.3890 | 0.1239 | 0.2048 | 0.1425 | sí |
| 0.4451 | 0.1228 | 0.2079 | 0.1305 | sí |
| 0.4500 | 0.1227 | 0.2082 | 0.1296 | sí |
| 0.5094 | 0.1199 | 0.2108 | 0.1199 | sí |
| 0.5830 | 0.1109 | 0.2135 | 0.1109 | sí |
| 0.6672 | 0.1033 | 0.2160 | 0.1033 | sí |
| 0.7635 | 0.0973 | 0.2183 | 0.0973 | sí |
| 1.0000 | 0.0901 | 0.2224 | 0.0901 | sí |
| 1.6000 | 0.0935 | 0.2284 | 0.0935 | sí |
| 3.0000 | 0.1310 | 0.1310 | 0.1310 | — |

Con el criterio adoptado la brecha física (39 puntos) va de 0.0223 a 0.1310; monótona: False; valor en Γ₂/κ=1: 0.2224 (modelo estático s5: 0.2338).

**Efecto en las conclusiones de la Tarea 38:** las correcciones ya publicadas (brecha física monótona ≈0.22–0.23) se mantienen porque la Tarea 38 usaba (peso>1e-3) o (inestable a 2%): el criterio adoptado da los mismos valores (tabla arriba, columna 3 vs 'sin filtro'). Lo que la sensibilidad muestra es que el umbral de peso *solo* no es suficiente en α²=2: con 0.1 reaparece la rama de 0.09 (no convergida con N). **Queda como incertidumbre**: esa rama no es identificable como física ni como artefacto con N≤32 (deriva −2% entre N=26 y 32); si fuera física, la brecha en Γ₂/κ≳0.5 sería ≈0.09 en vez de ≈0.22.

**Efecto en la Tarea 42:** los 7 cambios con umbrales bajos (1e-3, 1e-2) son celdas con g_x=0.2 (κ₂/κ≥0.07) donde los modos físicos llegan a peso 0.006–0.016; con el criterio adoptado (y con 3e-2, 0.1) la brecha es la publicada. La brecha física de la Tarea 42 tiene ~3% de incertidumbre en N (g_z/κ=12: 2.7% entre N=22 y 28).

## Actualización: la rama interior con N=32 y N=38 (auditoría de la rama ambigua)

Rama real de modos *interiores* (n medio ≈2; 99% de su masa en n≤10; peso de borde ≤5e-3 y decreciente con N), resonancia vestida, α²=2:

| Γ₂/κ | N=20 | N=26 | N=32 | N=38 | cambio por paso de N | modo de borde más lento (Re, N=20→38) |
|---|---|---|---|---|---|---|
| 1.6 | 0.0935 | 0.0918 | 0.0876 | —  | -1.8%, -4.6% | 0.147 → 0.117 → 0.096 |
| 3.0 | 0.1310 | 0.1313 | 0.1249 | 0.1161 |  | +0.3%, -4.9%, -7.0% | 0.170 → 0.142 → 0.120 → 0.102 |

(Γ₂/κ=1.0 con N=20/26/32: 0.0901 / 0.0896 / 0.0878 (Tarea 35).)

**Lectura.** La rama no es un modo de borde (perfil localizado en n≲10 y peso de borde que decrece con N), pero **no converge**: su Re cae 0.1310→0.1313→0.1249→0.1161 (N=20→38, Γ₂/κ=3), con caídas de −5% y −7% por paso y acelerándose, en paralelo al modo de borde más lento (0.170→0.102), con el que seguramente se hibrida. La estabilidad de 0.2% entre N=20 y 26 era casual: **con dos truncamientos el criterio (ii) es insuficiente**; hace falta estabilidad sobre tres o más (N, N+6, N+12).
**Criterio único revisado (adoptado para todo el trabajo):** un modo es *espurio* si (i) su peso de borde >0.5 (α²=4: separa limpiamente) o (ii) su autovalor no es estable al 0.3% en Re a lo largo de al menos tres truncamientos consecutivos (N, N+6, N+12); un modo que no cumple (ii) pero no cumple (i) se reporta como *no resuelto* (no se descarta ni se acepta como brecha).
**Conclusión revisada sobre las Tareas 21, 23, 27, 28:**
- Γ₂/κ ≲ 0.2: la brecha del 5º modo es estable en N (p. ej. 0.168 en Γ₂/κ=0.13) y crece monótonamente (0.022→~0.19): **sin cambio**.
- Γ₂/κ ≳ 0.25: la brecha de Floquet **no está determinada**. Hay una rama interior no convergida en 0.09–0.16 (N≤38) y un grupo estable en 0.19–0.23 (idéntico al modelo estático explícito, 0.234 en Γ₂/κ=1). La brecha física es, como mucho, la de la rama interior (cota superior ≈0.09–0.12 que sigue bajando con N) y, si esa rama fuera un artefacto de la hibridación con el borde, ≈0.22.
- Por tanto la afirmación de la Tarea 38 «la brecha física es monótona y satura en ~0.23» queda **rebajada a hipótesis** (compatible con el modelo estático), y el máximo/dip/subida de las Tareas 21, 23, 27, 28 no se puede atribuir por completo a un artefacto: el mínimo (~0.09 en Γ₂/κ≈1–1.6) y la subida en Γ₂/κ=3 (0.13) corresponden a esta rama no convergida. Lo que sí está establecido: el cuádruplete Im≈7–37 (peso de borde 1, |Im|∝N) es espurio, y γ_pf, γ_bf y la resonancia no dependen de esto.
