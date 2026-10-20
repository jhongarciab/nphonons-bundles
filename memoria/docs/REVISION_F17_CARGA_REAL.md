# Revisión de F17: carga real y desfase durante el segundo time-bin

Datos leídos de los logs `data/f17_ideal_tau*_chi*.log` en `pcjhon`
(16 oct 2026). No se repitieron las simulaciones en el Mac.

| Carga ideal, χ=1, τ₁₂ (1/ω_b) | F | s | F|h |
|---:|---:|---:|---:|
| 100 | 0.767 | 0.823 | 0.932 |
| 400 | 0.731 | 0.813 | 0.898 |
| 1000 | 0.696 | 0.812 | 0.858 |
| 2000 | 0.632 | 0.802 | 0.788 |
| 4000 | 0.552 | 0.798 | 0.692 |

Con carga real κ_in=κ, T=2000: F=0.473, s=0.766, F|h=0.617,
eficiencia de carga por rama 0.961/0.950.

La prueba **apoya fuertemente** el mecanismo propuesto: durante la
separación de bins, una rama tiene un excitón en D1 y la otra permanece
en G. El salto común proporcional al número total de excitones distingue
esas ramas y amortigua su coherencia. La caída de F|h ideal al alargar
τ₁₂ a 4000 es 0.240, que explica la mayor parte de la caída 0.315
entre el protocolo ideal corto y la carga real.

Queda una diferencia F|h=0.692 (ideal, τ₁₂=4000) frente a 0.617
(real). No atribuirla sin otra prueba solo al desfase: la carga real
también cambia la evolución durante la entrada, deja una cola del paquete
y permite fuga hacia el puerto. Un control decisivo es repetir la carga
real con γφ=0, manteniendo exactamente el mismo paquete, y examinar
por separado las poblaciones correctas y el módulo de la coherencia
entre |R0,A1⟩ y |R1,A2⟩. Esto distinguirá pérdida de carga de pérdida de
fase incluso cuando ambas produzcan el mismo F|h.

El resultado central de F13 (0.93 condicionado) corresponde a bins
introducidos idealmente con solo 100/ω_b entre ellos. En un paquete
temporal de duración ~4000/ω_b, esa cifra deja de describir la fuente
operativa. La afirmación actual debe usar 0.62 condicionado para esta
carga real y estos pulsos, hasta que exista una escritura más rápida o
una codificación protegida durante la llegada del segundo bin.

Esta revisión hereda la auditoría pendiente de cambios de marco del
láser entre ranuras en F9/F17.
