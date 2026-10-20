# F18. ¿Cuánto acoplo fonónico λ necesita la memoria? (propuesta A5, 16 oct 2026)

Código `src/f18_lambda_scan.py` (búsqueda + evaluación), `src/f18b_pulse_dephasing.py`; datos `data/f18_lambda_S*.json|log`, `data/f18b_pulse_dephasing.json`. Cómputo en pcjhon.
Modelo reducido, $\bm\lambda=S\,(0.20,\,0.14,\,0)\,\omega_b$, brechas $(0.35,\,0.60)$, $f_b=100$ GHz, $\kappa=10^{-3}$, $\gamma=2\times10^{-4}$, $\gamma_\varphi=4\times10^{-4}$, $\chi=1$, tasas de relajación de F11.
Para cada $S$ se **reoptimizan** los 6 parámetros de los pulsos ($\Omega$, duración, $\delta_2$ de cada ranura; $\Omega\le0.9\,\omega_b$) con E/S ideales, objetivo $F$ a $T=2000$; el punto de partida es el diseño analítico de `memoria_teoria.tex` (Ec. de $t_\pi$ y de $\delta_2^\star$ con polarón).
Luego se evalúa con carga y lectura reales (10$\kappa$, $T=2000$). Pasivo con los mismos puertos: $F=0.087$, $s=0.090$.

## Resultados
| $S$ | $\lambda_1,\lambda_2$ | $g_1,g_2$ | $\Omega$ óptima | duración (ranura 1, 2) | $F_h$ (E/S ideales, $\chi{=}1$) | $F_h$ ($\chi{=}0$) | E/S reales: $F$ / $s$ / $F_h$ |
|---:|---|---|---:|---|---:|---:|---|
| 1 | 0.200, 0.140 | 0.057, 0.061 | 0.19 / 0.21 | 188, 201 | 0.933 | 0.658 | 0.596 / 0.703 / **0.848** |
| 0.7 | 0.140, 0.098 | 0.040, 0.043 | 0.28 | 185, 206 | 0.924 | 0.652 | 0.576 / 0.687 / 0.838 |
| 0.5 | 0.100, 0.070 | 0.029, 0.031 | 0.39 | 185, 204 | 0.915 | 0.644 | 0.564 / 0.674 / 0.836 |
| 0.33 | 0.066, 0.046 | 0.019, 0.020 | 0.60 | 203, 223 | 0.884 | 0.618 | 0.461 / 0.634 / 0.727 |
| 0.2 | 0.040, 0.028 | 0.011, 0.012 | 0.90 (tope) | 206, 258 | 0.776 | 0.537 | 0.334 / 0.580 / 0.576 |
| 0.12 | 0.024, 0.017 | 0.007, 0.007 | 0.90 (tope) | 361, 399 | 0.752 | 0.507 | 0.295 / 0.509 / 0.579 |
**Hallazgos**
1. **El diseño analítico funciona sin optimizar:** con $S=1$ el punto de partida da $F=0.72$ (el óptimo, $0.745$); con $S=0.2$ da $0.51$ (óptimo $0.53$).
2. **Ley de escala de los pulsos:** la duración casi no cambia (190--260/$\omega_b$) porque $\Omega$ crece como $1/\lambda$, que mantiene $\Omega_{\rm eff}=\Omega g/\Delta_I\simeq0.0085$--$0.011$. El precio es un desplazamiento de luz mayor ($\delta_2\simeq-\Omega^2/\Delta_I$: $-0.02\to-0.60$) y más población virtual del brillante.
3. **Ventaja sobre el pasivo en $F$ y $s$ hasta $S=0.12$:** con E/S reales, $F=0.295$ frente a $0.087$ ($\times3.4$) y $s=0.51$ frente a $0.09$.
4. **Fidelidad condicionada:** $F_h\ge0.8$ (E/S reales) pide $S\gtrsim0.5$; $F_h\ge0.7$ pide $S\gtrsim0.33$; $F_h>0.5$ se mantiene hasta $S=0.12$ (entrelazado).
5. **Con ruido local ($\chi=0$)** $F_h$ cae a $0.51$--$0.66$ para todo $S$: el requisito de correlación del ruido no se relaja con λ.

## Criterio analítico para el acoplo mínimo
Con $p_B=(\Omega/\Delta_I)^2$ la población virtual del brillante, $\Omega_{\rm eff}=\Omega g/\Delta_I=\pi/(2t_\pi)$ da
$$g_{\min}=\frac{\Omega_{\rm eff}}{\sqrt{p_{B,\max}}}=\frac{\pi}{2t_\pi\sqrt{p_{B,\max}}}.$$
Con $t_\pi\simeq190/\omega_b$: $g_{\min}=0.026$ ($p_{B,\max}=0.1$) o $0.019$ ($p_{B,\max}=0.2$). El barrido lo confirma: $S=0.5$ ($g\simeq0.03$, $p_B=0.09$) conserva $F_h=0.84$, $S=0.33$ ($g\simeq0.02$, $p_B=0.20$) cae a $0.73$.
En unidades físicas ($f_b=100$ GHz): $\Omega\simeq0.4\,\omega_b$ es un Rabi de $\sim40$ GHz ($\hbar\Omega=0.17$ meV); $S\le0.2$ pide $\Omega\ge0.9\,\omega_b$ (90 GHz), probablemente fuera de lo razonable.

## F18b. Desfase inducido por la excitación durante los pulsos
$\gamma_\varphi$ durante los pulsos $=\gamma_{\varphi0}\,[1+K(\Omega/0.2)^2]$; fuera de ellos $\gamma_{\varphi0}$. Pulsos de F18 (no reoptimizados con esta penalización); E/S ideales, $T=2000$. $K$ es **un parámetro de sensibilidad**, no una constante medida.
| $S$ | $\Omega$ | $K=0$ | $K=0.25$ | $K=1$ |
|---:|---:|---|---|---|
| 1 | 0.20 | 0.745 / 0.798 / 0.933 | 0.736 / 0.795 / 0.926 | 0.712 / 0.786 / 0.906 |
| 0.7 | 0.28 | 0.727 / 0.787 / 0.924 | 0.710 / 0.780 / 0.910 | 0.663 / 0.762 / 0.870 |
| 0.5 | 0.39 | 0.710 / 0.775 / 0.915 | 0.679 / 0.763 / 0.890 | 0.598 / 0.728 / 0.822 |
| 0.33 | 0.60 | 0.676 / 0.765 / 0.884 | 0.611 / 0.736 / 0.830 | 0.465 / 0.663 / 0.701 |
| 0.2 | 0.90 | 0.529 / 0.682 / 0.776 | 0.431 / 0.632 / 0.682 | 0.267 / 0.526 / 0.508 |
| 0.12 | 0.90 | 0.447 / 0.595 / 0.752 | 0.330 / 0.529 / 0.623 | 0.180 / 0.407 / 0.443 |
(columnas: $F$ / $s$ / $F_h$). Con $K\le0.25$ la ventaja se mantiene hasta $S\simeq0.33$--$0.5$ ($F_h\ge0.83$ en $S=0.33$ con $K=0.25$); con $K=1$ el rango útil baja a $S\gtrsim0.5$.

## Lectura
- **$\lambda$ no es un bloqueo duro:** una memoria con ventaja de éxito y de $F$ sobre el pasivo sigue siendo posible con $\lambda$ cinco veces menor que la del modelo, con el coste de subir $\Omega$ proporcionalmente.
- **El límite lo fija $\Omega$**, es decir, la potencia: población virtual del brillante, desplazamiento de luz y desfase por excitación. Objetivo de plataforma: $g\gtrsim0.03\,\omega_b$ ($\lambda\sim0.1$) con $\Omega\lesssim0.4\,\omega_b$ para $F_h\approx0.8$ (E/S reales).
- Reservas: no se reoptimizaron brechas, pulsos con rampa (la admixtura del brillante sigue como rizado), ni se incluyó el desfase por excitación en la búsqueda; E/S reales evaluadas con los pulsos optimizados para E/S ideales; $\Omega\le0.9$ impuesto como tope.

## Corrección numérica (17 oct 2026)
La tabla de arriba se calculó con tolerancia del integrador 1e-6, que deja matrices densidad no positivas (ver F17d en `RESULTADOS_F7_F9_MEMORIA_RAMAN.md`). Se repitió con 1e-9: reevaluación de los pulsos antiguos (`data/f18tol9_reeval_S*.json`) y reoptimización en caliente (`data/f18_lambda_S*.json`, logs `f18tol9_opt_S*.log`; los resultados con 1e-6 están en `data/old_tol1e6_f18/`).
E/S reales, T=2000, F/s/F_h: S=1: 0.606/0.703/0.862; 0.7: 0.585/0.686/0.852; 0.5: 0.573/0.674/0.851; 0.33: 0.468/0.634/0.737; 0.2: 0.336/0.580/0.580; 0.12: 0.249/0.477/0.522.
Para S≥0.2 los pulsos óptimos no cambian y F_h sube 0.01–0.015. Para S=0.12 el óptimo se desplaza (duraciones 361/399 → 370/396) y F_h con E/S reales baja de 0.58 a 0.52 (los pulsos se optimizan con E/S ideales, y la lectura real es sensible a ellos). El requisito F_h≳0.8 ⇒ S≳0.5 se mantiene. F18b (desfase por excitación, K) no se repitió.
