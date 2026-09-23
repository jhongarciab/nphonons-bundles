# Tarea 13 — Verificar el quench (eps=0, qubit base vs excitado)

Script: `tarea13_verificar_quench.py`. Datos: `tarea13_resultados.npz`.
Nb=16, τ_max≈29.8 (estroboscópico), umbral positividad −1e-8.

## Resultados

| | ⟨n⟩ final | dn/dt final | p_e_ss |
|---|---|---|---|
| Qubit inicial **BASE** físico | **6.197e-3** | +3.39e-4 | 4.15e-6 |
| Qubit inicial **EXCITADO** físico (= todas las corridas previas salvo Tarea 10) | **2.219e-2** | −1.06e-4 | 5.40e-6 |

## Validación

Traza y hermiticidad perfectas. Positividad OK con el umbral −1e-8 en
ambos casos (mín. autovalor oscilador ~−3e-9 a −3e-10, qubit siempre
positivo). Nota: con el umbral anterior (−1e-9, Tarea 9) el caso
excitado hubiera fallado marginalmente; con −1e-8 pasa limpio.

## Contraste con las predicciones

1. **Hipótesis BASE (⟨n⟩<1e-4): NO se confirma tal cual.** El piso con
   qubit en base (6.2e-3) es **~3.6× menor** que con qubit excitado
   (2.22e-2) — mejora real y sustancial— pero no llega a 1e-4. Queda un
   residuo no explicado por el quench del qubit.

2. **Hipótesis EXCITADO (piso ≈ (2g_z/ω_m)²=0.0144 + contribución del
   par):** ajusta muy bien si se interpreta el piso BASE (6.2e-3) como
   "la contribución del par" (proceso físico real, independiente de la
   condición inicial del qubit) y se le suma la predicción de quench:

       0.0144 (quench) + 0.0062 (par, = piso BASE) = 0.0206
       medido (excitado) = 0.0222   →   razón = 1.08 (8% de diferencia)

   **Buen acuerdo** (dentro de ~8%) con esta descomposición aditiva:
   piso(excitado) ≈ (2g_z/ω_m)² + piso(base).

## Conclusión Tarea 13

El "piso" total observado en configuraciones con qubit inicial excitado
(todas las corridas de las Rondas 2-3 salvo la Tarea 10) se descompone
razonablemente bien en dos contribuciones aditivas:
- Un término de **quench del estado inicial** ≈(2g_z/ω_m)²=0.0144,
  específico de arrancar con el qubit en el estado "equivocado"
  (excitado en vez de base), que se anula al iniciar en el estado base.
- Un término **residual genuino** (~6.2e-3 con τ_max=30), presente
  incluso con el qubit en su estado base físico correcto y con muestreo
  estroboscópico limpio, que no se explica por el quench ni por el
  aliasing — su origen sigue sin resolverse por completo (posiblemente
  requiere τ_max mucho mayor para ver si relaja más, dado que dn/dt aquí
  es pequeño pero no cero, +3.4e-4).

**Recomendación para las Tareas 14-15**: usar SIEMPRE el qubit en estado
base físico (como se pide), aceptando que persiste un piso residual
pequeño (~6e-3 en τ_max=30) no relacionado con el artefacto de quench ya
identificado y corregido.
