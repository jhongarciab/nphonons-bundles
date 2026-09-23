# Ronda 9 — Tareas 24-25 (resonancia vestida): resumen

Entorno: Python 3.11 + QuTiP 4.7.6 (`../requirements.txt`). Datos: `tarea24_cache/`, `tarea25_cache/`;
tablas completas en `tarea25_resultados.md`, análisis 5×5 en `tarea24_grid5_analisis.py/.npz`.

## Tarea 24 — ubicación de la resonancia vestida
- Verificación paso 0: OK (4e-10). Se corrigió `tarea24_verificacion.py`: el orden de pares conjugados con igual |μ| depende de la plataforma; ahora empareja por vecino más cercano sobre 8 candidatos.
- Las 4 rejillas (Γ₂/κ = 0.03, 0.13, 0.52, 2.07) están completas.
- **La resonancia es un pico muy agudo en δ_m = +0.048 (= |δ₁|, signo OPUESTO a la predicción ingenua (δ₁,0)=(−0.048,0))**, con Im(λ_bf) ≈ 1e-13 y γ_bf ≈ 4e-8 (0.03), 1e-7 (0.13), 5e-7 (0.52), 1.9e-6 (2.07). Ancho ~0.01: a ±0.012 γ_bf sube 1-2 órdenes.
- **Corrección al hallazgo anterior**: la rejilla 9×9 de 0.52 (paso 0.036) saltó el pico; el punto (0.06, 0.144) daba γ_bf=1.2e-4, ~250× peor que en el pico (4.8e-7).
- Δ_q influye poco (factor ≤1.6 en γ_bf); mejor Δ_q=+0.144 en los 4 casos.
- **Punto de trabajo común: (δ_m, Δ_q) = (0.048, 0.144).**

## Tarea 25 (a) — γ vs α² en la resonancia vestida (α²=1..5)
Pendientes d ln(γ_bf)/dα² (completo | efectivo resonante):
| Γ₂/κ | completo | efectivo |
|---|---|---|
| 0.03 | −1.985 | −2.456 |
| 0.13 | −1.433 | −2.484 |
| 0.52 | −1.219 | −2.511 |
| 2.07 | −1.465 | −2.733 |

- γ_pf coincide completo/efectivo (pendiente +0.38 en ambos).
- γ_bf completo se aplana a α² alto (5 pts: 3.2e-9 → 2.8e-8 → 1.4e-7 → 4.6e-7 al subir Γ₂/κ) mientras el efectivo sigue cayendo ~e^{-2.5α²}; el efectivo subestima γ_bf hasta ~3 órdenes en α²=5. El acuerdo completo-efectivo es bueno solo en α²=1 y Γ₂/κ pequeño (8.5e-6 vs 8.2e-6).
- Comparado con Tarea 22 (sin resonancia, pendientes −1.28…−1.62): γ_bf baja ~3-4 órdenes de magnitud, pero las pendientes no son sustancialmente más pronunciadas (salvo Γ₂/κ=0.03).

## Tarea 25 (b) — brecha de confinamiento
- Efectivo: γ_conf crece con Γ₂/κ (0.04-0.15 en 0.03 → 2.9-10.5 en 2.07); modo con overlap n≈1.0-2.4, P≈0, Im=0.
- Completo: γ_conf casi constante ≈0.04-0.16 en todos los Γ₂/κ (saturado), Im(λ)≈0. **El efectivo sobrestima la brecha ~1-2 órdenes de magnitud a Γ₂/κ ≳ 0.5.**
- Overlaps del modo completo son irregulares en α² (n: 4.6 → 0.2 en Γ₂/κ=2.07), sugiriendo mezcla/reordenamiento de modos.
- **Caveat**: en 3 celdas el modo elegido como "confinamiento" en el completo es espurio (Im(λ) grande y overlaps ≈0): Γ₂/κ=0.13 α²=2 (Im −7.0), 0.52 α²=1 (Im +20), y Γ₂/κ=0.03/0.13 α²=1 con Im −1.7e-2/−1.3e-1. Su γ_conf no es fiable; requiere revisar `clasificar()` (umbral del modo de confinamiento) antes de citar esas filas.
