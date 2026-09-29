# -*- coding: utf-8 -*-
"""Tarea 41: delta_1 = gx^2 [Im S1- + Im S1+] frente a kappa/w_m, comparado con -4 gx^2/(3 w_m).
Usa las constantes y la formula de ../validacion/modelo_comun.py (venv de QuTiP 4: ../.venv/bin/python)."""
import sys, numpy as np
sys.path.insert(0, "../validacion")
import modelo_comun as mc, modelo_ladder as ml
gx, wm = mc.gx, mc.om_m
ImS = lambda D, x: -D / (x**2 + D**2)          # misma forma que modelo_comun / modelo_ladder (x = kappa/2)
d1 = lambda kap: gx**2 * (ImS(wm, kap / 2) + ImS(3 * wm, kap / 2))
lim = -4 * gx**2 / (3 * wm)
# comprobacion contra la implementacion de modelo_ladder (kappa=1)
chk = ml.valores(0.13)['lamb']
L = ["# Tarea 41 — δ₁ frente a κ/ω_m (modelo_comun: g_x=%.1f, ω_m=%.0f, κ variable)\n" % (gx, wm),
     f"Comprobación: δ₁(κ/ω_m={1/wm:.0e}) = {d1(1.0):.8f}; `modelo_ladder.valores()['lamb']` = {chk:.8f} (κ=1). Límite −4g_x²/(3ω_m) = {lim:.8f}.\n",
     "| κ/ω_m | δ₁ | −4g_x²/(3ω_m) | razón δ₁/(−4g_x²/(3ω_m)) | 1 − razón | corrección analítica (7/9)(κ/2ω_m)² |", "|---|---|---|---|---|---|"]
for r in np.logspace(-4, -1, 10):
    v = d1(r * wm); ra = v / lim
    L.append(f"| {r:.3e} | {v:.8f} | {lim:.8f} | {ra:.9f} | {1-ra:.3e} | {(7/9)*(r/2)**2:.3e} |")
g_ma = -0.3 * np.sin(np.pi / 4); w_ma = 6.0; kap_ma = 0.03
dM = g_ma**2 * (ImS(w_ma, kap_ma / 2) + ImS(3 * w_ma, kap_ma / 2))
L += ["", f"Parámetros de Ma (g_x={g_ma:.4f}, w={w_ma}, κ={kap_ma}, κ/w={kap_ma/w_ma:.3f}): δ₁={dM:.8f} vs −4g_x²/(3w)={-4*g_ma**2/(3*w_ma):.8f}; w_p*=2(w+δ₁)={2*(w_ma+dM):.6f}.",
      "", "**Convenciones.** Naseem: oscilador ω_m, qubit a 2ω_m, referencia ω_r=ω_d/2, δ_m=ω_m−ω_r y δ₁ es el corrimiento (Lamb) de la frecuencia del oscilador; la resonancia vestida es δ_m=−δ₁ (>0 pues δ₁<0). "
      "Ma: oscilador w, qubit d=2w, marco a w_p/2 y w_p; resonancia w_p/2 = w + δ₁ ⇒ w_p*=2(w−4g_x²/(3w)) (mismo δ₁, con el signo tal cual). "
      "En ambos g_x entra al cuadrado (signo irrelevante) y los términos contrarrotantes dan el factor (1+1/3): D₁₋=ω, D₁₊=3ω. "
      "La diferencia de unidades: Naseem mide en κ (ω_m/κ=1000) y Ma en 2π·GHz (κ/w=0.005)."]
open("tarea41_resultados.md", "w").write("\n".join(L)); print("\n".join(L))
