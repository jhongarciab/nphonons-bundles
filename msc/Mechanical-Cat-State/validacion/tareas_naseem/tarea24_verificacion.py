# -*- coding: utf-8 -*-
"""
Tarea 24, paso 0: verificacion del marco conmensurable de frecuencia
unica w_r.

Construye el Hamiltoniano completo en el marco:
  - mecanico rotante a w_r: termino estatico delta_m * b^dagger b,
    delta_m = omega_m - w_r
  - qubit rotante a 2*w_r: termino estatico (Delta_q/2) sigma_z,
    Delta_q = omega_q - 2*w_r
  - acoplamientos:
      g_x (sigma_+ e^{+2i w_r t} + sigma_- e^{-2i w_r t})
          (b e^{-i w_r t} + b^dagger e^{+i w_r t})
      g_z sigma_z (b e^{-i w_r t} + b^dagger e^{+i w_r t})
  - drive: eps(sigma_+ + sigma_-) + eps(sigma_+ e^{+4i w_r t} + sigma_- e^{-4i w_r t})

con w_r = omega_m (valor base). Con delta_m=0, Delta_q=0 (y los mismos
g_z, g_x, eps, Nb que la Tarea 18-ii) debe reproducir EXACTAMENTE los
autovalores del propagador de la Tarea 18(ii) (diferencia relativa <1e-6
en las 6 tasas lentas). Si no pasa, no seguir con el resto de la Tarea 24.

NO se modifica ningun script original.
"""

import numpy as np
from qutip import (tensor, qeye, destroy, sigmam, sigmap, sigmaz, Options, propagator)

r = 0.1
gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
gz_d, om_m, Gam_m = gz_org / kappa_org, omega_m_org / kappa_org, Gamma_m_org / kappa_org
kap = 1.0
n_th = 0
gx_dim = r * gz_d
g_dim = gz_d * gx_dim / om_m
Na, Nb = 2, 20
omega_q_dim = 2 * om_m   # baseline, Delta_q=0 aqui
eps = 4 * g_dim          # = 1.44, punto del paper (igual que Tarea 18-ii)

wr = om_m                # w_r = omega_m (valor base)
delta_m = 0.0
Delta_q = 0.0
T_r = 2 * np.pi / wr

opts = Options(atol=1e-12, rtol=1e-10, nsteps=2_000_000)

b = tensor(qeye(Na), destroy(Nb))
bd = b.dag()
sm = tensor(sigmam(), qeye(Nb))
sp = tensor(sigmap(), qeye(Nb))
sz = tensor(sigmaz(), qeye(Nb))

H_static = delta_m * bd * b + (Delta_q / 2) * sz + eps * (sp + sm)
H = [
    H_static,
    [gx_dim * sp * b, lambda t, _: np.exp(1j * wr * t)],
    [gx_dim * sp * bd, lambda t, _: np.exp(3j * wr * t)],
    [gx_dim * sm * b, lambda t, _: np.exp(-3j * wr * t)],
    [gx_dim * sm * bd, lambda t, _: np.exp(-1j * wr * t)],
    [gz_d * sz * b, lambda t, _: np.exp(-1j * wr * t)],
    [gz_d * sz * bd, lambda t, _: np.exp(1j * wr * t)],
    [eps * sp, lambda t, _: np.exp(4j * wr * t)],
    [eps * sm, lambda t, _: np.exp(-4j * wr * t)],
]
c_ops = [np.sqrt(kap) * sm, np.sqrt((n_th + 1) * Gam_m) * b, np.sqrt(n_th * Gam_m) * bd]

print(f"g_dim={g_dim}, eps={eps}, wr={wr}, T_r={T_r:.6e}")
print("Calculando propagador (marco conmensurable, delta_m=0, Delta_q=0)...")
U = propagator(H, T_r, c_ops, options=opts)
print(f"... listo. shape={U.shape}")

evals, evecs = U.eigenstates()
orden = np.argsort(-np.abs(evals))
evals_nuevo = evals[orden][:8]

d = np.load("tarea18_resultados.npz")
evals_ref = d["evals_ii"]

print("\n=== Comparacion con Tarea 18(ii) ===")
print(f"{'k':>3} {'mu_ref (Tarea18)':>28} {'mu_nuevo (marco w_r)':>28} {'dif_relativa':>14}")
max_dif = 0.0
for k in range(6):
    mu_ref = evals_ref[k]
    # empareja por vecino mas cercano: pares conjugados con igual |mu| salen
    # en orden arbitrario segun LAPACK/plataforma
    mu_new = evals_nuevo[np.argmin(np.abs(evals_nuevo - mu_ref))]
    dif = abs(mu_new - mu_ref) / max(abs(mu_ref), 1e-30)
    max_dif = max(max_dif, dif)
    print(f"{k:>3} {mu_ref:>28.10f} {mu_new:>28.10f} {dif:>14.3e}")

print(f"\nDiferencia relativa maxima = {max_dif:.3e}")
if max_dif < 1e-6:
    print("VERIFICACION: OK (diferencia < 1e-6). Se puede continuar con la Tarea 24.")
else:
    print("VERIFICACION: FALLA. NO CONTINUAR -- hay un error en la reconstruccion del "
          "Hamiltoniano en el marco w_r.")

np.savez("tarea24_verificacion.npz", evals_ref=evals_ref, evals_nuevo=evals_nuevo, max_dif=max_dif)
