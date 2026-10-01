"""Diagnóstico (fase 6, punto 4): efectivo CON FILTRO con el desplazamiento dependiente del estado, β = 2g_z/ω = 0.14, en la misma aproximación
secular que en el plano: canales D[σ∓a], D[σ∓a†] con tasa β²κ·ke(ω)·(n_q+1) (σ₋) y β²κ·ke(ω)·n_q (σ₊), ke(ω) = κ_f²/(4ω²+κ_f²) (κ_f/ω = 0.05).
Los términos σ₊a†b etc. del acoplamiento qubit-filtro son no seculares (giran a ±ω): su efecto a segundo orden es justo ese factor lorentziano.
NO es una corrección ni se aplica al mapa. Reporta γ_pf, γ_bf, η con y sin kick en κ₂/κ = 0.25 y 0.4 (x = 6.86) y a T → 0 (x = 60)."""
import time
import numpy as np
import calc_termico as CT
comp = {0.25: (1.652555e-4, 3.348950e-5), 0.4: (1.654430e-4, 6.908483e-5)}
for k in (0.25, 0.4):
    for nombre, x in (('x=6.86', 6.86), ('T→0 (x=60)', 60.0)):
        t0 = time.time(); e1 = CT.punto(x, k, 14.0, 1, 2e-5, 22, kick=True, rerun=True); dt = time.time() - t0
        e0 = CT.punto(x, k, 14.0, 1, 2e-5, 22)
        extra = f' | completo γ_pf={comp[k][0]:.4e} γ_bf={comp[k][1]:.4e}' if x == 6.86 else ''
        print(f'κ₂/κ={k} {nombre:11s}: sin kick γ_pf={float(e0["gpf"]):.5e} γ_bf={float(e0["gbf"]):.5e} η={float(e0["eta"]):.4g} | con kick γ_pf={float(e1["gpf"]):.5e} '
              f'γ_bf={float(e1["gbf"]):.5e} η={float(e1["eta"]):.4g} | Δγ_bf={float(e1["gbf"]) / float(e0["gbf"]) - 1:+.2e} Δγ_pf={float(e1["gpf"]) / float(e0["gpf"]) - 1:+.2e}{extra} | tiempo {dt:.1f} s')
