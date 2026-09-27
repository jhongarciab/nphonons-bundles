# V2 — Reporte de verificación independiente de R2 (resonancia vestida, modelo completo de Ma)

Fecha: 2026-09-26. Código: `v2_lab.py` (simulación), `v2_analisis.py` (análisis), `run_v2.sh`. Datos: `res_v2/*.npz`, `res_v2/analisis.txt`, `res_v2/log_*.txt`.
No se leyó ni importó código del trabajo original.

## Método (difiere del original)
- **Marco de laboratorio:** H(t) = ω a†a + (ω_q/2)σ_z + (a+a†)(g_xσ_x + g_zσ_z) + Ω(σ₊e^{−iω_pt} + h.c.), sin RWA. Incluye contrarrotantes y el término directo g_zσ_z(a+a†).
  Disipador κD[σ₋] con κ = 2Γ = 0.03, sin pérdida en el oscilador. Parámetros de Ma: ω=6, ω_q=12, g=0.3, θ=π/4, Ω=0.06.
  Estado inicial: |0⟩⊗|g⟩.
- **Floquet sobre T_p = 2π/ω_p**, el período del drive en el laboratorio (el original usó T = 4π/ω_p en el marco rotante).
  El propagador del superoperador sobre un período se obtiene con `qutip.propagator` (atol 1e-12, rtol 1e-10).
- **Evolución:** el propagador se aplica período a período, vector a vector, sin potencias por cuadrados.
- **Fase de muestreo:** estroboscópica, en t = nT_p (fase 0 del drive). En esos instantes el marco rotante (oscilador a ω_p/2, qubit a ω_p)
  coincide con el laboratorio salvo a → (−1)ⁿa, lo que deja invariantes el espacio del código {|±α⟩}, ⟨a²⟩, P_e y la paridad.
- **Micromovimiento:** tras el último instante estroboscópico se integra un período más con 40 muestras. Cada muestra se pasa al marco rotante
  (R = exp[it(ω_p/2)(a†a + σ_z)]) y de ahí se sacan el promedio sobre el período y el rango [mín, máx].
- **Código:** span{|2i⟩, |−2i⟩} (α² = Ω/G = −4, G = 2g_xg_z/ω = −0.015), con estados coherentes desnudos, proyectando tras trazar el qubit.

## Resultados principales

| ω_p | Γt | cantidad | mío (estrob. t=nT_p) | mío (promedio en el período) | trabajo | dif. rel. (estrob.) |
|---|---|---|---|---|---|---|
| 12.00 | 300 | P_c | 0.81180 | 0.811 | 0.8118 | 0 |
| 12.00 | 300 | P_e | 0.1144 | ≈0.116 | ≈0.12 | −5% (micromovimiento) |
| 12.00 | 300 | \|⟨a²⟩\| | 2.482 | ≈2.48 | ≈2.45 | +1.3% |
| 11.98 | 300 | P_c | 0.99754 | 0.9975 | 0.9975 | <1e-4 |
| 11.98 | 300 | P_e | 0.00265 | 0.0060 | ≈0.003 | −12% (el trabajo no fija la fase) |
| 11.98 | 300 | \|⟨a²⟩\| | 3.9734 | 3.982 | ≈3.97 | +1e-3 |
| 11.98 | 300 | paridad | +0.0016 | — | ≈0.002 | — |
| barrido | 60 | ω_p del máximo de P_c | 11.97993 (spline), 11.9800 (discreto) | — | 11.9806 (medido), 11.9800 (predicho) | −6e-4 respecto al medido; 0 respecto al predicho |

- Los promedios en el período de ω_p=12.00 son a Γt=60 (tabla del barrido). A Γt=300 solo se guardó el micromovimiento de P_c.
- En ω_p=11.98, P_e oscila entre 0.0026 y 0.0094 dentro del período; el mínimo cae en la fase estroboscópica.
  El trabajo (Tarea 39) reporta un rango 0.003–0.010, compatible con esto.
- P_c no depende de la fase: su rango en el período es ≤ 4e-3 en todos los puntos y ≤ 3e-5 en la resonancia.

## Barrido fino (Γt = 60, N = 22)

| ω_p | P_c | P_e estrob. | P_e promedio | \|⟨a²⟩\| estrob. |
|---|---|---|---|---|
| 11.960 | 0.1305 | 0.2690 | 0.2684 | 1.994 |
| 11.965 | 0.1753 | 0.2364 | 0.2378 | 2.881 |
| 11.970 | 0.7033 | 0.0821 | 0.0875 | 5.865 |
| 11.975 | 0.9610 | 0.0155 | 0.0199 | 4.708 |
| 11.9775 | 0.99046 | 0.0059 | 0.0097 | 4.294 |
| 11.9785 | 0.99526 | 0.0040 | 0.0076 | 4.157 |
| 11.9795 | 0.99736 | 0.0029 | 0.0063 | 4.032 |
| **11.9800** | **0.99755** | 0.0027 | 0.0060 | 3.973 |
| 11.9806 | 0.99712 | 0.0026 | 0.0059 | 3.906 |
| 11.9815 | 0.99528 | 0.0030 | 0.0061 | 3.810 |
| 11.9825 | 0.99174 | 0.0042 | 0.0072 | 3.710 |
| 11.985 | 0.97728 | 0.0099 | 0.0125 | 3.482 |
| 11.990 | 0.93025 | 0.0335 | 0.0356 | 3.090 |
| 11.995 | 0.86787 | 0.0719 | 0.0733 | 2.747 |
| 12.000 | 0.80015 | 0.1213 | 0.1228 | 2.452 |
| 12.005 | 0.71895 | 0.1853 | 0.1859 | 2.169 |

- La resonancia es asimétrica: cae mucho más rápido del lado de ω_p menor.
- La ventana con P_c > 0.99 va de ω_p ≈ 11.9775 a 11.9825.
- El trabajo (Tarea 39/40) a 11.98 da P_c 0.9975, P_e 0.0026 y |⟨a²⟩| 3.973, que coinciden con mis valores estroboscópicos.
  A 11.9806 el trabajo da el máximo; yo tengo 0.99712 ahí, frente a 0.99755 en 11.9800.

## Evolución temporal (estroboscópica)

| ω_p | Γt=30 | Γt=60 | Γt=150 | Γt=300 |
|---|---|---|---|---|
| 11.98: P_c / paridad | 0.99618 / +0.375 | 0.99755 / +0.194 | 0.99755 / +0.027 | 0.99754 / +0.0016 |
| 12.00: P_c / paridad | 0.7960 / +0.709 | 0.8004 / +0.553 | 0.8077 / +0.274 | 0.8118 / +0.115 |

- En 11.98, P_c satura en ≈Γt 60. La paridad decae hacia 0, así que el estado final es una mezcla de |±2i⟩, no el gato par, tal como dice el trabajo.
- En 12.00, P_c sigue subiendo muy lentamente (0.796 → 0.812) sin llegar a 0.99.

## Validaciones y convergencia
- En toda la serie de todas las corridas: |Tr ρ − 1| ≤ 9.3e-11, ‖ρ − ρ†‖ ≤ 4.5e-13 y mínimo autovalor ≥ −1e-24. Las tres tolerancias se cumplen.
- Convergencia en N (ω_p = 11.98, Γt = 60), N=22 → 28:

  | cantidad | N=22 | N=28 | diferencia |
  |---|---|---|---|
  | P_c | 0.99755 | 0.99755 | <1e-5 |
  | P_e | 0.00265 | 0.00265 | <1e-5 |
  | \|⟨a²⟩\| | 3.9734 | 3.9734 | <1e-4 |
  | paridad | 0.1939 | 0.1937 | 1e-3 |

  La tabla de Γt=30 no se usa para convergencia: la rejilla de muestreo difiere (30.2 frente a 30.0).

## Veredicto
**Confirma R2.** En ω_p = 12.00, P_c satura en 0.81 y no hay gato. En ω_p = 11.98, P_c = 0.9975, P_e ≈ 0.003 y |⟨a²⟩| = 3.97.
El máximo del barrido está en 11.9799 ± 0.0005, que coincide con la predicción 2(ω − 4g_x²/3ω) = 11.9800. El 11.9806 medido por el trabajo difiere en 6e-4, por debajo de su paso de barrido.

## Dudas y aproximaciones
- El código se define con estados coherentes desnudos |±2i⟩. No se corrige el desplazamiento estático ∓g_z/ω ≈ 0.035 del oscilador por g_z, ni el vestido del qubit.
  Esto limita P_c por debajo de 1, pero el trabajo usa, según sus informes, la misma definición: P_c coincide al 1e-5.
- Mi paso de barrido (≥5e-4) no permite distinguir 11.9800 de 11.9806 mejor que ~3e-4. Que el spline caiga en 11.97993 es indicativo, no concluyente a ese nivel.
- El P_e del trabajo (0.003) solo coincide con mi valor estroboscópico. El promedio en el período es 0.006, así que el trabajo probablemente muestreó en fase equivalente.
- En esta verificación no se midió el FWHM.
- Por un reinicio de sesión, el barrido se terminó en paralelo con 1 hilo por proceso. Esto no afecta a los resultados.
