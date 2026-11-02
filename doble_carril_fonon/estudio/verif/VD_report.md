# VD: verificación independiente del criterio ρ* y auditoría de la tabla R7

Código: `verif/vd/vd_check.py` (QuTiP propio, escrito antes de abrir ningún script del autor).

## Parte 1. Derivación

**Notación.** p = e^{-T/T₁}. Sea φ la fase de frecuencia acumulada con ruido gaussiano; |⟨e^{-iφ}⟩| = e^{-f₁}, y con la convención sobre |⟨a⟩|², f₁ = T/(2Tφ) (blanco). (El dossier usa f_I = T/Tφ = 2f₁ en la línea 66: la fórmula coincide, pero f_I no está definido en el texto.)

**(a) Carril simple.** Para |ψ⟩ = a|0⟩+b|1⟩ tras amortiguamiento y desfase: ρ₀₀ = |a|²+(1−p)|b|², ρ₁₁ = p|b|², ρ₀₁ = √p·e^{-f₁}·ab*. La fidelidad por estado cardinal es: |0⟩ → 1; |1⟩ → p; ±x, ±y → ½(1+√p e^{-f₁}) (cada uno). Promedio: F_s = [1+p+4·½(1+√p e^{-f₁})]/6 = **(3+p+2√p e^{-f₁})/6**. Coincide con lo esperado.

**(b) Doble carril.** Tras pérdida, ρ = p·ρ_{1ex} + (1−p)|00⟩⟨00|, con independencia del estado lógico (pérdidas iguales). La coherencia ⟨10|ρ|01⟩ = p·ab*·⟨e^{i(φ₁−φ₂)}⟩. Var(φ₁−φ₂) = 2σ²(1−ρ) y f₁ = σ²/2, así que ⟨·⟩ = e^{-σ²(1−ρ)} = e^{-2(1−ρ)f₁}; por tanto **C = e^{-2(1−ρ)f₁}**. Probabilidad de éxito: **p = e^{-T/T₁}**. Condicionado al éxito, la fidelidad es 1, 1 para |10⟩, |01⟩ y ½(1+C) para los cuatro estados ecuatoriales. Entonces F_h = [2+4·½(1+C)]/6 = **(2+C)/3**.

**(c) Umbral.** 6F_d = 2p(2+C)+3(1−p) = 3+p+2pC. Frente a 6F_s = 3+p+2√p e^{-f₁}:

F_d > F_s ⇔ pC > √p e^{-f₁} ⇔ √p·C > e^{-f₁} ⇔ −T/(2T₁) − 2(1−ρ)f₁ > −f₁ ⇔ f₁(2ρ−1) > T/(2T₁).

Con ruido blanco f₁ = T/(2Tφ): (2ρ−1)/(2Tφ) > 1/(2T₁) ⇔ 2ρ−1 > Tφ/T₁ ⇔ **ρ > ½(1+Tφ/T₁)**, independiente de T. (Se asume 2ρ−1>0 para que haya solución; si ρ ≤ ½ el doble carril nunca gana.)
Sin desfase: C = 1 ⇒ F_d = p+(1−p)/2 = (1+p)/2, y 6F_s − 6F_d = 2√p(1−√p)·… concretamente 6(F_s−F_d) = 2√p − 2p = 2√p(1−√p) > 0 para 0<p<1. Luego F_d < F_s para todo p<1, con igualdad solo en p=1 (T=0) o p=0.
Valores: ρ* = 0.6002 (Tφ=107) y 0.5094 (Tφ=10), con T₁=534. **Confirmado el dossier (0.60 y 0.51).**

**(e) Generalizaciones.**
- *Cuasi-estático* f₁ = κT² = ½(T/Tφ)²: la condición f₁(2ρ−1) > T/(2T₁) da ρ > ½[1 + 1/(2κT T₁)] = **½(1 + Tφ²/(T·T₁))**. Depende de T: el umbral baja al crecer T (a T=100, Tφ=107, T₁=534: ρ* = 0.607) y sube sin límite para T→0.
- *Amplitudes desiguales* (el carril simple usa α=1; modo k con sensibilidad α_k): Var(Δφ) = σ²(α₁²+α₂²−2ρα₁α₂), luego C = exp[−f₁(α₁²+α₂²−2ρα₁α₂)]. Condición: f₁(1 − α₁² − α₂² + 2ρα₁α₂) > T/(2T₁). Blanco: **ρ > [Tφ/T₁ + α₁²+α₂²−1]/(2α₁α₂)**. Para α₁=α₂=1 se recupera ½(1+Tφ/T₁); en cuasi-estático, sustituir Tφ/T₁ por Tφ²/(T·T₁). Nota: si α₁²+α₂² es mucho mayor que 1+... el umbral puede superar 1 (imposible); y para el doble carril con α₁≠α₂ queda un desfase de fase no compensable por la Z común aunque ρ=1 (si α₁≠α₂, C<1 incluso con ρ=1).

**(d) Tabla numérica** (Lindblad en sector de 1 fonón + |00⟩, operadores √(γ(1±ρ)/2)(n₁±n₂) con γ=1/Tφ; carril simple: √(1/T₁)σ₋ y √γ n). T₁=534 µs. Num = QuTiP, An = fórmulas anteriores; F_s, F_h y F_d = p·F_h+(1−p)/2.

| Tφ | T | ρ | ρ* | F_s num/an | F_h num/an | p num/an | F_d num | F_d−F_s | lado correcto |
|---:|---:|---:|---:|---|---|---|---:|---:|:--:|
| 107 | 25 | 0.50 | 0.600 | 0.94876 / 0.94876 | 0.96326 / 0.96326 | 0.9543 / 0.9543 | 0.94207 | −0.0067 | sí |
| 107 | 25 | 0.70 | 0.600 | 0.94876 / 0.94876 | 0.97745 / 0.97745 | 0.9543 / 0.9543 | 0.95561 | +0.0069 | sí |
| 107 | 100 | 0.50 | 0.600 | 0.82843 / 0.82843 | 0.87560 / 0.87560 | 0.8292 / 0.8292 | 0.81146 | −0.0170 | sí |
| 107 | 100 | 0.70 | 0.600 | 0.82843 / 0.82843 | 0.91854 / 0.91854 | 0.8292 / 0.8292 | 0.84707 | +0.0186 | sí |
| 107 | 200 | 0.50 | 0.600 | 0.72316 / 0.72316 | 0.79763 / 0.79763 | 0.6876 / 0.6876 | 0.70465 | −0.0185 | sí |
| 107 | 200 | 0.70 | 0.600 | 0.72316 / 0.72316 | 0.85699 / 0.85699 | 0.6876 / 0.6876 | 0.74547 | +0.0223 | sí |
| 10 | 25 | 0.409 | 0.509 | 0.75234 / 0.75234 | 0.74281 / 0.74280 | 0.9543 / 0.9543 | 0.73170 | −0.0206 | sí |
| 10 | 25 | 0.609 | 0.509 | 0.75234 / 0.75234 | 0.79220 / 0.79220 | 0.9543 / 0.9543 | 0.77883 | +0.0265 | sí |
| 10 | 100 | 0.409 | 0.509 | 0.64025 / 0.64025 | 0.66757 / 0.66757 | 0.8292 / 0.8292 | 0.63896 | −0.0013 | sí |
| 10 | 100 | 0.609 | 0.509 | 0.64025 / 0.64025 | 0.67337 / 0.67337 | 0.8292 / 0.8292 | 0.64376 | +0.0035 | sí |
| 10 | 200 | 0.409 | 0.509 | 0.61461 / 0.61461 | 0.66667 / 0.66667 | 0.6876 / 0.6876 | 0.61460 | −0.00001 | sí |
| 10 | 200 | 0.609 | 0.509 | 0.61461 / 0.61461 | 0.66680 / 0.66680 | 0.6876 / 0.6876 | 0.61469 | +0.00008 | sí |

Acuerdo numérico/analítico a ≤1e-5 en todos los puntos; el signo de F_d−F_s sigue a ρ−ρ* en los 12 casos. (Con Tφ=10 y T≥100 la diferencia es minúscula porque ambas fidelidades están cerca del suelo ½-⅔.)

## Parte 2. Auditoría de R7 y §0 contra `src/design2_eval_si.json`
(g=3, fsr=3.3, T1q=300; n=1 es blanco, n=2 cuasi-estático; los valores se comparan redondeados a 3 cifras.)

**Tabla R7 (líneas 76-81): las 6 filas × 8 columnas coinciden con el JSON.** Redondeos límite verificados: p_pass(100)=0.80147→0.801; F_h ideal(100)=0.86052→0.861; B2(Φ=0.9, 200)=0.64853→0.649. Idéntica a `results_tables.md` Tabla R-A (comprobadas filas Tφ=107 y 10).

**§0 (líneas 21-29): todas las cifras coinciden.**
- Línea 26: F_ef 0.781 (JSON 0.7814) frente a B0 0.828 (0.8284); 0.826 (0.8259) frente a 0.834 (0.8343); 0.871 (0.8712) frente a 0.828. Φ=0.42→ρ=0.66 cuasi-estático: 1−0.58²=0.664 (R1). Correcto.
- Línea 27: F_c 0.851 / 0.907, B0 0.828 / 0.834, p=0.80; ganancias +0.023 y +0.073 («+0.02 a +0.07»: correcto).
- Línea 29: ρ*=0.51 y F_ef 0.88 (0.8799) frente a 0.75 (B0 0.7523), T=25, Tφ=10, Φ=0.90: correcto.
- Línea 25: ρ*=0.60 correcto.
- **No encontré ninguna cifra incorrecta ni de R7 ni de §0.** Las «discrepancias» a priori (0.781 vs 0.828, etc.) son comparaciones F_ef-vs-B0 legítimas, no errores.

**Afirmaciones que se sostienen solo parcialmente (precisión, no error numérico):**
1. Línea 28 y R7(i) «pierde ≤ 0.012 de fidelidad y ≈0.03 de probabilidad». Las cifras valen para F_c (no F_ef) con Tφ=107 y T≤200 µs: máx(F_h ideal − F_c) = 0.0105 (T=25), prob. 0.0225-0.0322 (T=200 … 25; 0.0278 a T=100, que es el «0.028» de R7). Fuera de ese rango no vale: con Tφ=107 hasta 0.0185 (T=500); con Tφ=10 hasta 0.0193 (T=25-200); la pérdida en F_ef respecto de F_ef ideal llega a 0.024 (Tφ=107) y 0.027 (Tφ=10) (p. ej. 0.781 vs 0.799 a T=100). Sugiero escribir «≤0.012 en F_c, para Tφ=107 y T ≤ 200 µs» y dar también el coste en F_ef (≈0.02).
2. R7(ii) «con g=1.1 MHz, F_ef = 0.75–0.76»: se cumple para T=100, Φ=0.42 blanco, FSR=3.3 y T1q=100-300 (0.752-0.759; con T1q=28 µs baja a 0.718; con FSR=1 sube a 0.769-0.776). Falta decir «T=100, Φ=0.42, FSR=3.3». Además, para g=1.1 el chequeo es peor (F_h ideal − F_c hasta 0.019 y Δp hasta 0.107 en T ≤ 200, T1q ≥ 100), lo que contradice la lectura (i) si se aplica a g ≳ 1 (línea 28: «viable con g ≳ 1 MHz»): la frase de «≤0.012» es solo para g = 3.
3. Línea 29: «con Φ=0.90, F_ef = 0.88 frente a 0.75» es F_ef a T=25 µs con Tφ=10; a T≥100 la ventaja se reduce (0.724 vs 0.640 a T=100; 0.636 vs 0.615 a T=200) y para Φ=0.42 y Tφ=10 el doble carril pierde frente a B0 a T=25 (0.708 vs 0.752).
4. R7(iv)/R8: R8 (HBAR) coincide con `design2_eval_hbar.json` (F_c 0.605, B0 0.864, B2 = p = 0.614, F_ef ideal 0.807). Pero R8 compara F_c = 0.605 con 0.614 «sin chequear» (que es B2 y, a la vez, p_ideal): es F_c frente a B2 no heraldada, comparación que el propio dossier declara no justa; la correcta es F_ef 0.603 frente a B0 0.864 (aunque la conclusión no cambia).
5. Cuasi-estático: «0.826 frente a 0.834 (empate)» es −0.008, un empate dentro de la incertidumbre del modelo; con el heraldado ideal el doble carril sí gana (0.844 vs 0.834), coherente con ρ*(T=100)=0.607<0.66 (umbral T-dependiente, ver (e)). Conviene decir que el ρ*=0.60 de §0 es solo para blanco y que en cuasi-estático el umbral depende de T.

**Coherencia de F_ef y de la convención de Tφ.**
- F_ef = p·F_c + (1−p)/2 usada igual en §0, línea 70, R7, `results_tables.md`; comparada siempre con B0 (no heraldada), no con F_c. Coherente. En §0.3 se compara F_c con B0, pero acompañado de la probabilidad de paso (0.80), de acuerdo con el aviso de la línea 70.
- Tφ sobre |⟨a⟩|² (R2): B0 = (3+p+2√p·e^{-T/(2Tφ)})/6 reproduce el JSON a 4 cifras para todos los T, Tφ=107 y 10 (p. ej. 0.8284, 0.6402); ρ* usa Tφ/T₁ coherente con esa convención (verificado en (c) y (d)). Coherente.
- Defecto de notación: la línea 66 usa f_I sin definir; definir f_I = T/Tφ (por modo, sobre |⟨a⟩|²) o escribir e^{-T/(2Tφ)}.
- Línea 171 («F_c 0.774 → 0.851») y las líneas 62-63 (JC completo) no se pueden comprobar con el JSON; el propio dossier avisa de que el JC usa la convención antigua.

## Resumen (10 líneas)
1. Derivé F_s=(3+p+2√p e^{-f₁})/6, C=e^{-2(1−ρ)f₁}, F_h=(2+C)/3, p=e^{-T/T₁}: todo coincide con lo esperado.
2. Umbral: F_d>F_s ⇔ f₁(2ρ−1)>T/(2T₁) ⇔ ρ>½(1+Tφ/T₁) (blanco); ρ*=0.600 (Tφ=107) y 0.509 (Tφ=10).
3. Sin desfase 6(F_s−F_d)=2√p(1−√p)>0: el doble carril pierde para todo p<1.
4. QuTiP propio (12 puntos, ambos lados del umbral): acuerdo ≤1e-5 con las fórmulas y signo de F_d−F_s correcto en los 12.
5. Cuasi-estático: ρ>½(1+Tφ²/(T T₁)), depende de T (ρ*=0.607 a T=100, Tφ=107).
6. Amplitudes desiguales: ρ>[Tφ/T₁+α₁²+α₂²−1]/(2α₁α₂); con α₁≠α₂ C<1 incluso con ρ=1.
7. Tabla R7: las 48 cifras coinciden con el JSON (redondeos incluidos); §0 sin errores numéricos.
8. «Pierde ≤0.012 y ≈0.03» solo vale para F_c, g=3, Tφ=107, T≤200 (hasta 0.019 con Tφ=10 o T=500; en F_ef ≈0.02-0.027).
9. «g=1.1: F_ef 0.75-0.76» solo para T=100, Φ=0.42, FSR=3.3; con g=1.1 el chequeo es peor (ΔF_c ≤0.019, Δp ≤0.107), tensión con «viable con g≳1».
10. F_ef y convención de Tφ coherentes en todo el dossier; reparos menores: f_I sin definir (línea 66) y R8 compara F_c con B2 (no heraldada).
