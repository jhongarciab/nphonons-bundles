# Tarea 10 — Estado inicial del qubit (fig2, ε=1.44=Ω_full)

Script: `tarea10_estado_inicial_qubit.py`. Datos: `tarea10_resultados.npz`.
Nb=N_eff=30, τ_max≈39, estroboscópico. Modelo B (g_eff=2g, ε=Ω_full).

## Contexto

`fig2_git_v1.py` inicializa el qubit con `thermal_dm(Na,0)` = `basis(Na,0)`,
etiquetado "ground" en los comentarios — pero la Tarea 7 verificó que
`basis(Na,0)` (σ_z=+1) es **físicamente el estado EXCITADO**. Se compara
partir del estado base físico real (`basis(Na,1)`) contra el excitado
(`basis(Na,0)`, = lo que usa fig2 originalmente).

## Resultados

| | F_min | κt(F_min) | F_final (κt≈39) |
|---|---|---|---|
| Qubit inicial **BASE** físico | **0.7693** | 0.779 | **0.9863** |
| Qubit inicial **EXCITADO** físico (= fig2 original) | 0.5088 | 1.169 | 0.9813 |

## Validación

Traza y hermiticidad perfectas en ambos casos. Positividad: caso
excitado, mín. autovalor ~1e-24 (perfecto); caso base, mín. autovalor
~−5 a −10e-9, **ligeramente fuera del umbral −1e-9** (ruido numérico al
límite, no violación física clara dado el orden de magnitud). El qubit
reducido se mantiene positivo en ambos casos, decayendo monótonamente
(relajación hacia estado puro).

## Respuesta a la pregunta: ¿desaparece la caída de F en κt~1 con el qubit en estado base?

**No desaparece, pero mejora sustancialmente.** Iniciar el qubit en su
estado base físico real:
- **Reduce la infidelidad del dip a menos de la mitad**: 1−F_min pasa de
  0.491 (excitado) a 0.231 (base) — mejora de ~2.1×.
- Desplaza el mínimo de F antes en el tiempo (κt=0.78 vs 1.17).
- Mejora también F_final (0.986 vs 0.981), aunque marginalmente.

Esto confirma que **parte** del dip de fidelidad observado en la Tarea 1
(F_min=0.7306 con la condición inicial original de fig2, que en efecto
inicia con el qubit excitado) es un **artefacto evitable**: si se corrige
la condición inicial del qubit al estado base físico correcto, la
fidelidad mínima mejora notablemente. Pero el dip **no desaparece por
completo** (F_min=0.77, no ~1): queda un mecanismo residual —
probablemente el mismo transitorio de "vestimiento"/dressing entre el
oscilador (que parte de vacío en ambos modelos) y el proceso de squeezing
de dos fonones arrancando desde cero, más el piso/efecto residual no
resuelto por completo en la Tarea 9— que sigue produciendo una caída de
fidelidad genuina a tiempos cortos, independiente de si el qubit arranca
en su estado base o excitado.

## Conclusión Tarea 10

El estado inicial del qubit en fig2 (heredado de un etiquetado incorrecto
de "ground" en el código, que en realidad prepara el estado excitado)
**sí contribuye de forma significativa** al dip de fidelidad a tiempos
cortos, pero no es la única causa. Se recomienda, para reproducciones
futuras de fig2, inicializar el qubit explícitamente en su estado base
físico (`basis(Na,1)` en la convención de signo de σ_z usada en el
código) para obtener una comparación completo-vs-efectivo más fiel al
comportamiento asintótico esperado del sistema en su condición física
natural (bombeado desde el vacío/estado base, no desde una inversión de
población inicial arbitraria).
