# -*- coding: utf-8 -*-
"""
Tarea 4 (fig4): mismo bug que en fig1_git_v1.py, presente tambien en
fig4_git_v1.py (linea ~150):

    Re_S1_plus = x / (x**2 + Delta1_minus**2)   # <-- deberia ser Delta1_plus

Se corre, para cada n_th en [0, 0.5, 1, 2] (igual que fig4_git_v1.py), la
version ORIGINAL (bug) y la CORREGIDA, y se compara la distancia de traza
entre los estados finales (t_final = 3 / Gamma2_minus, igual que el original).

NO se modifica fig4_git_v1.py.
"""

import numpy as np
from qutip import destroy, thermal_dm, mesolve, lindblad_dissipator, Options, tracedist

N = 100
gz = 2 * np.pi * 6e6
r = 0.1
gx = r * gz
omega_m = 2 * np.pi * 100e6
omega_q = 2 * omega_m
kappa = 2 * np.pi * 100e3
gamma = 2 * np.pi * 15
g = (gz * gx) / omega_m
eps = 10 * g
chi = -2j * eps * g / kappa
n_th_list = [0.0, 0.5, 1.0, 2.0]

x = kappa / 2.0
D1_minus = omega_q - omega_m
D1_plus = omega_q + omega_m
D2_minus = omega_q - 2 * omega_m
D2_plus = omega_q + 2 * omega_m

Re_S2_minus = x / (x**2 + D2_minus**2)
Gamma2_minus = 2 * g**2 * Re_S2_minus
Re_S2_plus = x / (x**2 + D2_plus**2)
Gamma2_plus = 2 * g**2 * Re_S2_plus

Im_S2_minus = -D2_minus / (x**2 + D2_minus**2)
Im_S2_plus = -D2_plus / (x**2 + D2_plus**2)
delta_k = g**2 * (Im_S2_minus + Im_S2_plus)

tlist = np.linspace(0, 3, 61) / Gamma2_minus

a = destroy(N)
ad = a.dag()
H = chi.conjugate() * ad**2 + chi * a**2 + delta_k * (ad * a)**2
options = Options(nsteps=100000)

Re_S1_minus = x / (x**2 + D1_minus**2)
Gamma1_minus = 2 * gx**2 * Re_S1_minus

Re_S1_plus_bug = x / (x**2 + D1_minus**2)   # bug
Gamma1_plus_bug = 2 * gx**2 * Re_S1_plus_bug

Re_S1_plus_fix = x / (x**2 + D1_plus**2)    # correcto
Gamma1_plus_fix = 2 * gx**2 * Re_S1_plus_fix

print(f"Gamma1_plus (bug)      = {Gamma1_plus_bug:.6e} Hz")
print(f"Gamma1_plus (corregido)= {Gamma1_plus_fix:.6e} Hz  "
      f"(razon = {Gamma1_plus_bug/Gamma1_plus_fix:.4f}, esperado 9.0)")

resumen = []
for n_th in n_th_list:
    Gamma_minus = (n_th + 1) * gamma + Gamma1_minus
    Gamma_plus_bug = n_th * gamma + Gamma1_plus_bug
    Gamma_plus_fix = n_th * gamma + Gamma1_plus_fix

    diss_bug = [Gamma_minus * lindblad_dissipator(a),
                Gamma_plus_bug * lindblad_dissipator(ad),
                Gamma2_minus * lindblad_dissipator(a**2),
                Gamma2_plus * lindblad_dissipator(ad**2)]
    diss_fix = [Gamma_minus * lindblad_dissipator(a),
                Gamma_plus_fix * lindblad_dissipator(ad),
                Gamma2_minus * lindblad_dissipator(a**2),
                Gamma2_plus * lindblad_dissipator(ad**2)]

    rho0 = thermal_dm(N, n_th)
    print(f"\nn_th={n_th}: corriendo bug y corregido...")
    res_bug = mesolve(H, rho0, tlist, diss_bug, [], options=options)
    res_fix = mesolve(H, rho0, tlist, diss_fix, [], options=options)

    td_final = tracedist(res_bug.states[-1], res_fix.states[-1])
    td_all = np.array([tracedist(res_bug.states[i], res_fix.states[i]) for i in range(len(tlist))])
    print(f"  Gamma_plus bug={Gamma_plus_bug:.4e}  fix={Gamma_plus_fix:.4e}  "
          f"tracedist_final={td_final:.3e}  tracedist_max={td_all.max():.3e}")
    resumen.append((n_th, Gamma_plus_bug, Gamma_plus_fix, td_final, td_all.max()))

print("\n=== TABLA RESUMEN (Tarea 4, fig4) ===")
print(f"{'n_th':>6} {'Gamma_plus_bug':>16} {'Gamma_plus_fix':>16} {'tracedist_final':>16} {'tracedist_max':>14}")
for n_th, gpb, gpf, tdf, tdm in resumen:
    print(f"{n_th:>6.1f} {gpb:>16.4e} {gpf:>16.4e} {tdf:>16.3e} {tdm:>14.3e}")

np.savez("tarea4_fig4_resultados.npz",
         n_th_list=np.array(n_th_list),
         resumen=np.array(resumen))
print("\nGuardado: tarea4_fig4_resultados.npz")
