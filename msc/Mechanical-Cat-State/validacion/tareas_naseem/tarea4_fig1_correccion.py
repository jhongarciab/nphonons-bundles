# -*- coding: utf-8 -*-
"""
Tarea 4 (fig1): en fig1_git_v1.py, Re_S1_plus usa Delta1_minus en vez de
Delta1_plus (bug de copiar-pegar):

    Re_S1_plus = x / (x**2 + Delta1_minus**2)   # <-- deberia ser Delta1_plus

Se corre el modelo ORIGINAL (con el bug, tal como esta en fig1_git_v1.py, sin
modificarlo) y una version CORREGIDA (Delta1_plus), y se compara la distancia
de traza entre los estados finales.

NO se modifica fig1_git_v1.py.
"""

import numpy as np
from qutip import destroy, thermal_dm, mesolve, lindblad_dissipator, Options, tracedist

# ---------------------------
# Parametros (identicos a fig1_git_v1.py)
# ---------------------------
N = 60
gz = 2 * np.pi * 6e6
r = 0.1
gx = r * gz
omega_m = 2 * np.pi * 100e6
omega_q = 2 * omega_m
kappa = 2 * np.pi * 100e3
gamma = 2 * np.pi * 15
g = (gz * gx) / omega_m
eps = 4 * g
chi = -2j * eps * g / kappa
n_th = 0

x = kappa / 2
D1_minus = omega_q - omega_m
D1_plus = omega_q + omega_m
D2_minus = omega_q - 2 * omega_m
D2_plus = omega_q + 2 * omega_m

Re_S1_minus = x / (x**2 + D1_minus**2)
Gamma1_minus = 2 * gx**2 * Re_S1_minus
Gamma_minus = (n_th + 1) * gamma + Gamma1_minus

# --- ORIGINAL (con el bug tal como esta en fig1_git_v1.py) ---
Re_S1_plus_bug = x / (x**2 + D1_minus**2)     # BUG: usa D1_minus
Gamma1_plus_bug = 2 * gx**2 * Re_S1_plus_bug
Gamma_plus_bug = n_th * gamma + Gamma1_plus_bug

# --- CORREGIDO ---
Re_S1_plus_fix = x / (x**2 + D1_plus**2)      # correcto: usa D1_plus
Gamma1_plus_fix = 2 * gx**2 * Re_S1_plus_fix
Gamma_plus_fix = n_th * gamma + Gamma1_plus_fix

Re_S2_minus = x / (x**2 + D2_minus**2)
Gamma2_minus = 2 * g**2 * Re_S2_minus
Re_S2_plus = x / (x**2 + D2_plus**2)
Gamma2_plus = 2 * g**2 * Re_S2_plus

Im_S2_minus = -D2_minus / (x**2 + D2_minus**2)
Im_S2_plus = -D2_plus / (x**2 + D2_plus**2)
delta_k = g**2 * (Im_S2_minus + Im_S2_plus)

print(f"D1_minus = {D1_minus:.4e}   D1_plus = {D1_plus:.4e}   (Hz)")
print(f"Re_S1_plus (bug, con D1_minus)     = {Re_S1_plus_bug:.6e}")
print(f"Re_S1_plus (correcto, con D1_plus) = {Re_S1_plus_fix:.6e}")
print(f"Razon bug/correcto = {Re_S1_plus_bug/Re_S1_plus_fix:.4f}  "
      f"(esperado (D1_plus/D1_minus)^2 = {(D1_plus/D1_minus)**2:.4f})")
print(f"Gamma_plus (bug)      = {Gamma_plus_bug:.6e} Hz")
print(f"Gamma_plus (corregido)= {Gamma_plus_fix:.6e} Hz")
print(f"Gamma_minus           = {Gamma_minus:.6e} Hz  (sin cambios, ~5-6 ordenes mayor)")

a = destroy(N)
ad = a.dag()
H = chi.conjugate() * ad**2 + chi * a**2 + delta_k * (ad * a)**2
rho0 = thermal_dm(N, n_th)
tlist = np.linspace(0, 3, 31) / Gamma2_minus
options = Options(nsteps=100000)

diss_bug = [Gamma_minus * lindblad_dissipator(a),
            Gamma_plus_bug * lindblad_dissipator(ad),
            Gamma2_minus * lindblad_dissipator(a**2),
            Gamma2_plus * lindblad_dissipator(ad**2)]

diss_fix = [Gamma_minus * lindblad_dissipator(a),
            Gamma_plus_fix * lindblad_dissipator(ad),
            Gamma2_minus * lindblad_dissipator(a**2),
            Gamma2_plus * lindblad_dissipator(ad**2)]

print("\nCorriendo version ORIGINAL (bug)...")
res_bug = mesolve(H, rho0, tlist, diss_bug, [], options=options)
print("Corriendo version CORREGIDA...")
res_fix = mesolve(H, rho0, tlist, diss_fix, [], options=options)

# ---------------------------
# Comparacion: distancia de traza entre estados a cada tiempo
# ---------------------------
td = np.array([tracedist(res_bug.states[i], res_fix.states[i]) for i in range(len(tlist))])

print("\n=== Distancia de traza bug vs corregido ===")
for i in [0, 5, 10, 15, 20, 25, 30]:
    print(f"  paso {i:2d} (t*Gamma2_minus={tlist[i]*Gamma2_minus:.2f}): "
          f"tracedist = {td[i]:.3e}")

print(f"\nDistancia de traza MAXIMA sobre toda la evolucion = {td.max():.3e}")
print(f"Distancia de traza en el estado FINAL              = {td[-1]:.3e}")

np.savez("tarea4_fig1_resultados.npz", tlist=tlist, tracedist=td,
         Gamma_plus_bug=Gamma_plus_bug, Gamma_plus_fix=Gamma_plus_fix,
         Gamma_minus=Gamma_minus)
print("\nGuardado: tarea4_fig1_resultados.npz")
