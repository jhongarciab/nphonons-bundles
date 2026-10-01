# Pre-registro: completo con filtro a κ_f/ω = 0.2 (escrito antes de lanzar las corridas)

Fecha de redacción: 2026-10-01, antes del commit que lo contiene. No se ha lanzado ninguna corrida de esta serie.

## Corridas

Modelo completo con filtro, N = 20, N_f = 2, en (κ₂/κ, x) = (0.25, 6.86): g_x = 0.0535714, ω = 6, g_z = 0.42, κ = 0.03, **κ_f = 1.2 (κ_f/ω = 0.2)**, |α|² = 4, γ = 6e-7. Una pareja: térmica (`--x=6.86`) y T = 0 (sin `--x`). Comando: `calc_filtro_completo.py 0.0535714 6.0 0.42 0.03 1.2 4 20 2 --gam=6e-07 [--x=6.86]`. Se registran tasas, `al2eff`, hermiticidad, mínimo autovalor, tiempo real y memoria máxima (`/usr/bin/time`).

## Predicciones del efectivo (κ = 1; N = 22, N_f = 2; `calc_termico`, κ_f/ω = 0.2)

| cantidad | valor |
|---|---|
| γ_pf a T = 0 (x = 60) | 1.66411e-04 |
| γ_pf térmico (x = 6.86) | 1.78395e-04 |
| Δ_e = aumento térmico sin ansatz | 1.1984e-05 |
| γ_pf térmico con ansatz (n_q) | 1.85736e-04 |
| **E_ansatz** = aumento por el ansatz con n_q | **+7.341e-06** (+4.12%) |
| γ_bf térmico / a T = 0 (efectivo) | 3.4148e-05 / 8.1797e-09 |
| E con ocupación física (referencia) | +5.607e-05 |

## Estadístico y regla de decisión (fijados antes)

- **Estadístico:** E_obs = (r_x − r_0)·γ_e(x), con r_0 = γ_pf(completo, T = 0)/γ_pf(efectivo, x = 60) y r_x = γ_pf(completo, térmico)/γ_pf(efectivo, x = 6.86), todo en κ = 1 (las tasas del completo divididas entre κ = 0.03), y γ_e(x) = 1.78395e-04. Se reporta también la diferencia cruda D = Δ_c − Δ_e = E_obs + (r_0 − 1)Δ_e.
- **Regla (sobre E_obs):** E_obs > 3e-6 implica canal de tamaño ansatz; |E_obs| < 1e-6 implica que no hay canal; entre 1e-6 y 3e-6 (y E_obs < −1e-6), indeterminado.
- **Esperado con canal de tamaño ansatz:** E_obs ≈ +7.3e-6 (D ≈ +6.8e-6 con r_0 = 0.956). **Sin canal:** E_obs ≈ 0 (D ≈ −5.3e-7 con r_0 = 0.956).
- Supuesto: r_0 ≈ 0.956, el valor medido con κ_f/ω = 0.05. Si r_0 cambia con κ_f, D cambia en (r_0 − 1)Δ_e, pero E_obs no.
- La regla se aplica a E_obs porque quita la referencia α_eff²/T = 0, que fue la fuente de la compensación en los diagnósticos previos.

## Comprobaciones previas

- **N_f = 2 basta:** efectivo con N_f = 2, 3 y 4 a κ_f/ω = 0.2: γ_pf térmico 1.7839497e-04, 1.7839545e-04, 1.7839545e-04 (cambio 2.7e-6 relativo); γ_pf a T = 0 idéntico (1.6641095e-04); con ansatz 1.8573609e-04 y 1.8573631e-04 (E_ansatz 7.341e-06 en los tres casos). γ_bf térmico sí cambia 0.8% de N_f = 2 a 3 (3.4148e-05 a 3.3872e-05; N_f = 4 igual que 3); no entra en la regla, que usa γ_pf.
- **4J²/κ_f = κ se mantiene:** J = √(κ κ_f)/2 en `calc_filtro_completo.py`; para κ_f = 1.2, J = 0.09487 y 4J²/κ_f = 0.03000 = κ.
- **γ ≪ κ:** γ/κ = 2e-5; además γ/κ_f = 5e-7.
- **Resolución:** el residual de N = 20 en γ_pf (−3.2e-5 relativo entre N = 20 y 22 con κ_f/ω = 0.05, unos 5e-9κ) es mucho menor que los umbrales de la regla (1e-6 y 3e-6). Las dos corridas usan N = 20, de modo que r_0 y r_x comparten truncamiento.
- ke(ω) = κ_f²/(4ω² + κ_f²) = 9.9e-3 (16 veces el de κ_f/ω = 0.05).

## Qué no se hace

No se lanza `nogz_pair`, N = 20 en κ₂/κ = 0.05, N = 24 ni Naseem, y no se tocan figuras.
