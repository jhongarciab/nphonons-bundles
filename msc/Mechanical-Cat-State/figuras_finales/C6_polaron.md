# C6 — Base polarónica del espacio del código

Código: `comun.py` (modelo y utilidades), `calc_c6_polaron.py` (caché `data/c6_<caso>_N<N>.npz`; `--rerun` fuerza el recálculo).
Logs: `data/log_c6_*.txt`.

## Derivación
Con el qubit en |g⟩ (σ_z|g⟩ = −|g⟩), la parte del oscilador de H es

ω a†a − g_z(a + a†) = ω (a − g_z/ω)†(a − g_z/ω) − g_z²/ω,

de modo que el oscilador está desplazado en **d = +g_z/ω, real, en el marco de laboratorio**. Con el qubit en |e⟩ el desplazamiento sería −g_z/ω.
En el marco rotante a ω_p/2 el desplazamiento rota como (g_z/ω)e^{iω_pt/2}. En los instantes estroboscópicos t = nT_p vale ±g_z/ω, y como el código es simétrico,
**el código es {D(g_z/ω)|±α⟩}**. El signo es independiente del de g_x y de la fase de α: solo depende del signo de g_z, con σ_z = |e⟩⟨e| − |g⟩⟨g|.

## Verificación numérica
Estado estacionario de Floquet: autovector con λ = 1 del propagador de un período en el laboratorio, estroboscópico en t = nT_p.
Se maximiza P_c = Tr[Π_code(d) ρ] sobre un desplazamiento complejo d (Nelder–Mead).

| caso | N | α | d predicho (g_z/ω) | d óptimo | d_opt/d_pred | P_c(d=0) | P_c(d=g_z/ω) | P_c(d=−g_z/ω) | P_c(d_opt) |
|---|---|---|---|---|---|---|---|---|---|
| Ma, ω_p = 11.98 | 22 | 2i | 0.03536 | 0.03451 + 2e-5 i | 0.976 | 0.997544 | **0.998725** | 0.993890 | 0.998726 |
| Ma, ω_p = 11.98 | 28 | 2i | 0.03536 | 0.03451 + 2e-5 i | 0.976 | 0.997551 | **0.998732** | 0.993897 | 0.998733 |
| g_x=0.05, ω=6, g_z/κ=12 (T42) | 22 | 2 | 0.06000 | 0.05992 | 0.999 | 0.995044 | **0.998682** | 0.984188 | 0.998682 |

**Cuánto sube P_c con d = g_z/ω:**
- Ma: de 0.99754 a 0.99873 (1 − P_c baja de 2.46e-3 a 1.27e-3, un factor 1.9).
- Punto T42 con g_z/κ = 12: de 0.99504 a 0.99868 (1 − P_c baja de 4.96e-3 a 1.32e-3, un factor 3.8).

El signo queda confirmado: con −g_z/ω P_c baja respecto a d = 0.

**Desplazamiento óptimo:** coincide con ⟨a⟩ estroboscópico, el centro de la mezcla simétrica: 0.03449 en Ma y 0.05992 en T42.
- En Ma es un 2.4% menor que g_z/ω. La corrección de campo medio −g_z⟨σ_z⟩/ω, con ⟨σ_z⟩ = −0.9947, explica un tercio (da 0.03517). El resto no se identificó; el candidato es la respuesta estroboscópica a g_x⟨σ_x(t)⟩, pero no está verificado.
- La diferencia de P_c entre d = g_z/ω y d_opt es ≤ 1e-6, así que **en todas las figuras se usa d = g_z/ω analítico**.

**Tamaño del gato:** si además se libera |α|, el óptimo da |α|² = 3.98 en Ma y 3.87 en T42.
- En T42 P_c sube a 0.99979, frente a 0.99868 con |α|² = 4.
- En Ma el cambio es despreciable (0.998759 frente a 0.998732).
- Esto es coherente con C1 (|α|² = |⟨a²⟩|). Propongo definir el código como {D(g_z/ω)|±α_eff⟩}, con α_eff² fijado por el ⟨a²⟩ polarónico.
  **Queda a tu decisión.** Por defecto uso α nominal con d = g_z/ω, como pide C6.

## Validaciones y convergencia
- ρ estacionarias: |Tr ρ − 1| ≤ 2.2e-16, ‖ρ − ρ†‖ = 0, mínimo autovalor ≥ 4.2e-10 (todas las tolerancias se cumplen).
- Convergencia N = 22 → 28 (Ma): P_c(g_z/ω) cambia 7e-6 y d_opt es idéntico a 5 cifras.
