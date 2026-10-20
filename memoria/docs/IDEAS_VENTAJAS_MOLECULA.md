# ¿Qué puede hacer la molécula excitónica que un emisor único no puede? (cambio de rumbo, 15 oct 2026)

Premisa del cambio: buscar ternas no es una ventaja frente a Bin. Se busca una función o sintonización de la molécula que
mejore las métricas de Bin (pureza, Q requerido, tasa, estadística) o que haga algo que un emisor único no puede.

**Nota de método.** «J=0» no es un emisor único: son dos emisores independientes que comparten el modo. El emisor único
se simula con un punto (λ₂=0 y Ω₂=0, o nqd=1).

## Evidencia ya obtenida (régimen de Bin: λ=0.14, Ω=0.1, γ=2×10⁻⁴, γφ=4×10⁻⁴, bombeo en la banda de n=2; F1)
Composición de ciclos entre todos los fotones; Δ por máximo de ⟨b†²b²⟩ estacionario; 500 trayectorias.

| κ | caso | J | P(2 fonones) | P(3) | P(≥4) | flujo (10⁻⁵ω_b) |
|---:|---|---:|---:|---:|---:|---:|
| 0.003 | emisor único | — | 0.61 | 0.20 | 0.10 | 38 |
| 0.003 | dímero simétrico | 0.10 | 0.60 | 0.20 | 0.10 | 41 |
| 0.003 | dímero simétrico | 0.25 | 0.60 | 0.20 | 0.10 | 40 |
| 0.003 | acoplo antisimétrico (λ₁=−λ₂) | 0.10 | 0.56 | 0.23 | 0.11 | 41 |
| 0.003 | acoplo antisimétrico | 0.25 | 0.43 | 0.30 | 0.17 | 40 |
| 0.010 | emisor único | — | 0.45 | 0.25 | 0.26 | 38 |
| 0.010 | dímero simétrico | 0.10–0.25 | 0.45 | 0.25 | 0.26 | 46 |
| 0.010 | acoplo antisimétrico | 0.25 | 0.24 | 0.35 | 0.37 | 42 |

Lecturas: (i) el dímero simétrico **no mejora** la composición; el flujo sube 7–22 % (a Ω=0.1 la emisión está
limitada por κ, no por la fuerza de bombeo: la superradiancia ×2 no aparece). (ii) La composición del emisor único es
independiente de Ω (0.45/0.25/0.26 a Ω=0.05–0.2) y empeora con κ (0.61 → 0.45): a Q=100 queda margen. (iii) El acoplo
antisimétrico **empeora** la pureza de n=2 (la resonancia de orden n+1 se acerca en ω−2J), refutando la idea de
«protección por paridad» para el orden 2 en este régimen. (iv) Con acoplo de intercambio (λ₂=0) y S=ω el máximo de f₂ cae
en el cruce y la salida es de tres fonones (P(3)≈0.45–0.64): no sirve para n=2 sin rediseño (F2).

## Ideas (ordenadas por unicidad frente al emisor único)

| # | Idea | Qué da la molécula que el emisor único no | Métrica a probar | Estado |
|---|---|---|---|---|
| 1 | **Cavidad-QED de fonones con acoplo de intercambio sintonizable** | Un emisor único tiene acoplo longitudinal (diagonal, `σ_z`). El par híbrido (|+⟩,|−⟩) tiene acoplo transversal `g=δλJ/S` con S sintonizable por Stark: es un átomo de Jaynes–Cummings para fonones | desdoblamiento de vacío `2g√(n+1)` resuelto (g>κ,γφ), bloqueo de fonón (g²<1), preparación de Fock por intercambio, paquetes tipo Muñoz con pureza alta a Q bajo | Sin probar; g≈0.05 ω_b alcanzable con λ₂≈0 y J≈0.4–0.5 ω_b |
| 2 | **Paquetes correlacionados en dos modos / entrelazamiento** | Cada punto en un antinodo de un modo distinto; el Förster enlaza los modos | correlación cruzada g_ab, testigo de entrelazamiento | Sin probar; el emisor único también puede acoplar dos modos (Wang/Zou/Liao 2023) |
| 3 | **Escalado superradiante con N puntos + desfase por excitación** | √N en la tasa efectiva con un bombeo por punto 1/√N; el desfase por excitación crece ∝Ω² (Cosacchi, Ramsay) | pureza y tasa vs bombeo por punto con γφ=K Ω² | Solo útil si Ω_eff<κ; a κ-limitado no hay ganancia (F1) |
| 4 | **Etiqueta espectral del camino** | Las dos líneas ópticas (separadas ≈ε) identifican la rama de la cascada; el fotón de un emisor único no | P(n | línea) vs P(n) global a Q bajo, n=2 | Parcial: para n=3, 0.65 → 0.72 (color) → 0.83 (ventana); no probado para n=2 |
| 5 | **Resonancia inmune al ruido de modo común** | S depende solo de la diferencia ε: el ruido de carga común no desplaza el cruce | anchura de la resonancia con deriva espectral común y diferencial | Sin probar; solo protege la composición, no la excitación |

## Siguiente paso propuesto
Probar primero la idea 1 (única de verdad): barrer ε, J y δλ (λ₂≈0) en el régimen de acoplo fuerte y mirar
(a) el espectro del modo de fonón (¿doblete resuelto?), (b) g²(0) y la distribución de fonones (¿bloqueo?),
(c) la preparación determinista de |1⟩, |2⟩ por pulsos de intercambio.
