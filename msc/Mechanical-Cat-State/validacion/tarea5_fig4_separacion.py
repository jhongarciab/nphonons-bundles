# -*- coding: utf-8 -*-
"""
Tarea 5 (fig4): separar el efecto del n_th del bano del efecto del n_th del
estado inicial (en fig4_git_v1.py ambos estan atados al mismo n_th).

Para cada n_th en {0.5, 1, 2} se comparan tres variantes a t_final = 3/Gamma2_minus:

  (0) ORIGINAL (como en fig4_git_v1.py): estado inicial termico(n_th),
      bano termico con ocupacion n_th (Gamma_minus, Gamma_plus dependen de n_th)
  (i) estado inicial VACIO, bano con n_th (Gamma_minus/Gamma_plus con n_th real)
  (ii) estado inicial TERMICO(n_th), bano FRIO n_th=0 (Gamma_minus/Gamma_plus con n_th=0)

Se compara, para cada variante: negatividad de Wigner (volumen de la parte
negativa) y paridad <P> = Tr[rho * (-1)^n].

NO se modifica fig4_git_v1.py.
"""

import numpy as np
from qutip import destroy, thermal_dm, mesolve, lindblad_dissipator, Options, wigner, Qobj, expect

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

x = kappa / 2.0
D1_minus = omega_q - omega_m
D1_plus = omega_q + omega_m
D2_minus = omega_q - 2 * omega_m
D2_plus = omega_q + 2 * omega_m

Re_S1_minus = x / (x**2 + D1_minus**2)
Gamma1_minus = 2 * gx**2 * Re_S1_minus
Re_S1_plus = x / (x**2 + D1_plus**2)   # version corregida (Tarea 4); cambio despreciable
Gamma1_plus = 2 * gx**2 * Re_S1_plus

Re_S2_minus = x / (x**2 + D2_minus**2)
Gamma2_minus = 2 * g**2 * Re_S2_minus
Re_S2_plus = x / (x**2 + D2_plus**2)
Gamma2_plus = 2 * g**2 * Re_S2_plus

Im_S2_minus = -D2_minus / (x**2 + D2_minus**2)
Im_S2_plus = -D2_plus / (x**2 + D2_plus**2)
delta_k = g**2 * (Im_S2_minus + Im_S2_plus)

t_final = 3 / Gamma2_minus
tlist = np.array([0.0, t_final])   # solo nos interesa el estado final
options = Options(nsteps=100000)

a = destroy(N)
ad = a.dag()
H = chi.conjugate() * ad**2 + chi * a**2 + delta_k * (ad * a)**2

xvec = np.linspace(-6, 6, 600)
dx = xvec[1] - xvec[0]

# Operador de paridad: (-1)^n
P = Qobj(np.diag((-1.0) ** np.arange(N)))


def gammas_para(bath_nth):
    Gamma_minus = (bath_nth + 1) * gamma + Gamma1_minus
    Gamma_plus = bath_nth * gamma + Gamma1_plus
    return Gamma_minus, Gamma_plus


def correr(init_nth, bath_nth):
    Gamma_minus, Gamma_plus = gammas_para(bath_nth)
    diss = [Gamma_minus * lindblad_dissipator(a),
            Gamma_plus * lindblad_dissipator(ad),
            Gamma2_minus * lindblad_dissipator(a**2),
            Gamma2_plus * lindblad_dissipator(ad**2)]
    rho0 = thermal_dm(N, init_nth)
    res = mesolve(H, rho0, tlist, diss, [], options=options)
    rho_final = res.states[-1]

    W = wigner(rho_final, xvec, xvec)
    neg_vol = -dx * dx * np.sum(W[W < 0])   # volumen de la parte negativa

    parity = expect(P, rho_final)
    n_mean = expect(ad * a, rho_final)

    return neg_vol, parity, n_mean


n_th_list = [0.5, 1.0, 2.0]
resumen = []

for n_th in n_th_list:
    print(f"\n=== n_th = {n_th} ===")

    neg0, par0, n0 = correr(init_nth=n_th, bath_nth=n_th)       # (0) original
    print(f"(0) ORIGINAL   (init={n_th}, bano={n_th}): "
          f"neg_vol={neg0:.5f}  <P>={par0:.5f}  <n>={n0:.4f}")

    negi, pari, ni = correr(init_nth=0.0, bath_nth=n_th)        # (i) vacio + bano n_th
    print(f"(i) VACIO+BANO (init=0,    bano={n_th}): "
          f"neg_vol={negi:.5f}  <P>={pari:.5f}  <n>={ni:.4f}")

    negii, parii, nii = correr(init_nth=n_th, bath_nth=0.0)     # (ii) termico + bano frio
    print(f"(ii) TERM+FRIO (init={n_th},  bano=0)   : "
          f"neg_vol={negii:.5f}  <P>={parii:.5f}  <n>={nii:.4f}")

    resumen.append((n_th, neg0, par0, n0, negi, pari, ni, negii, parii, nii))

print("\n=== TABLA RESUMEN (Tarea 5) ===")
hdr = f"{'n_th':>5} | {'neg(0)':>8} {'P(0)':>8} {'n(0)':>7} | " \
      f"{'neg(i)':>8} {'P(i)':>8} {'n(i)':>7} | {'neg(ii)':>8} {'P(ii)':>8} {'n(ii)':>7}"
print(hdr)
for row in resumen:
    n_th, neg0, par0, n0, negi, pari, ni, negii, parii, nii = row
    print(f"{n_th:>5.1f} | {neg0:>8.5f} {par0:>8.5f} {n0:>7.4f} | "
          f"{negi:>8.5f} {pari:>8.5f} {ni:>7.4f} | {negii:>8.5f} {parii:>8.5f} {nii:>7.4f}")

np.savez("tarea5_fig4_resultados.npz", n_th_list=np.array(n_th_list),
         resumen=np.array(resumen))
print("\nGuardado: tarea5_fig4_resultados.npz")
