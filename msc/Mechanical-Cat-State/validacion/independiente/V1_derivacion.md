# V1 — Derivación independiente de H_ef (James–Jerke, segundo orden)

H₀ = ω a†a + (ω_q/2)σ_z, V = (a + a†)(g_x σ_x + g_z σ_z). En la imagen de interacción respecto a H₀,
V(t) = Σₙ [hₙ e^{−iωₙt} + h.c.] con frecuencias positivas:

| término | hₙ | ωₙ |
|---|---|---|
| rotante de g_x | g_x σ₋ a† | ω_q − ω ≈ ω |
| contrarrotante de g_x | g_x σ₋ a | ω_q + ω ≈ 3ω |
| g_z | g_z σ_z a | ω |

James–Jerke: H_ef = Σ_{m,n} (1/ω̄_{mn}) [h_m†, h_n] e^{i(ω_m−ω_n)t}, conservando pares con ω_m ≈ ω_n (ω_q ≈ 2ω).

- Diagonal rotante: (g_x²/ω)[σ₊a, σ₋a†] = (g_x²/ω)[(n+1)|e⟩⟨e| − n|g⟩⟨g|].
- Diagonal contrarrotante: (g_x²/3ω)[σ₊a†, σ₋a] = (g_x²/3ω)[n|e⟩⟨e| − (n+1)|g⟩⟨g|].
- Diagonal g_z: (g_z²/ω)[a†, a] = −g_z²/ω (constante; σ_z² = 1).
- Cruzado rotante g_x × g_z (ambos a ω; fase e^{i(ω_q−2ω)t}, resonante):
  (g_xg_z/ω)[σ₊a, σ_z a] + h.c. = (g_xg_z/ω)(σ₊σ_z − σ_zσ₊)a² + h.c. = −(2g_xg_z/ω)(σ₊a² + h.c.).
- Contrarrotante × g_z: 3ω vs ω, no resonante → se descarta.

Suma (más el drive en el marco rotante a ω_p, RWA):

H_ef = (g_x²/ω)[(4n/3 + 1)|e⟩⟨e| − (4n/3 + 1/3)|g⟩⟨g|] − G(σ₊a² + h.c.) − g_z²/ω + Ω(σ₊ + σ₋),  G = 2g_xg_z/ω.

**Coincide con R1** (salvo la constante −g_z²/ω, irrelevante). Corrimientos: oscilador en |g⟩ −4g_x²/(3ω), en |e⟩ +4g_x²/(3ω);
qubit (n = 0) +4g_x²/(3ω). De ellos, 3/4 viene del término rotante y 1/4 del contrarrotante: sin contrarrotantes
se obtiene −g_x²/ω para el oscilador en |g⟩ y +g_x²/ω para el qubit (n = 0); es decir, los valores de Ma (0 y 3g_x²/ω) no se reproducen con ninguna de las dos restricciones obvias — no tengo el
paper para ver su derivación.

Pendiente de la rama g_z × g_x contrarrotante: no hay término de pares desde 3ω a este orden.
