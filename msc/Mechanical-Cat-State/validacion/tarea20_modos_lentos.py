# -*- coding: utf-8 -*-
"""
Tarea 20: identificar los modos lentos del propagador de Floquet en el
punto del paper (eps=1.44, Nb=20).

Se toman los 8 autovalores de mayor modulo y, para cada autovector X_k
(en el espacio completo qubit+oscilador), se calcula el overlap
Tr[O^dagger X_k] con los operadores:
  - P (paridad del oscilador, extendida con identidad en el qubit)
  - n = a^dagger a (extendido)
  - a (extendido)
  - a^2 (extendido)
  - sigma_z (extendido con identidad en el oscilador)

Se clasifica cada modo segun cual overlap domina:
  phase-flip       -> P
  coherencia logica / bit-flip -> a
  fuga / confinamiento -> n, a^2
  del qubit        -> sigma_z

Se confirma si el modo 4 (lambda~0.099, identificado en la Tarea 18-ii)
es la "brecha de confinamiento".

NO se modifica ningun script original.
"""

import numpy as np
from qutip import (tensor, qeye, destroy, sigmam, sigmaz, sigmax, Options, propagator,
                    vector_to_operator, Qobj)

r = 0.1
gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
gz_d, om_m, Gam_m = gz_org / kappa_org, omega_m_org / kappa_org, Gamma_m_org / kappa_org
kap = 1.0
n_th = 0
gx_dim = r * gz_d
g_dim = gz_d * gx_dim / om_m
omega_q_dim = 2 * om_m
omega_d_dim = 2 * om_m
T_m = 2 * np.pi / om_m
Na = 2
Nb = 20
eps = 4 * g_dim

opts = Options(atol=1e-12, rtol=1e-10, nsteps=2_000_000)

b = tensor(qeye(Na), destroy(Nb))
bd = b.dag()
sm = tensor(sigmam(), qeye(Nb))
sz_full = tensor(sigmaz(), qeye(Nb))
sx = tensor(sigmax(), qeye(Nb))
H0 = 0.5 * omega_q_dim * sz_full
H = [H0,
     [gx_dim * sx * b, lambda t, _: np.exp(-1j * om_m * t)],
     [gx_dim * sx * bd, lambda t, _: np.exp(+1j * om_m * t)],
     [gz_d * sz_full * b, lambda t, _: np.exp(-1j * om_m * t)],
     [gz_d * sz_full * bd, lambda t, _: np.exp(+1j * om_m * t)],
     [eps * sx, lambda t, _: np.exp(+1j * omega_d_dim * t)],
     [eps * sx, lambda t, _: np.exp(-1j * omega_d_dim * t)]]
c_ops = [np.sqrt(kap) * sm, np.sqrt((n_th + 1) * Gam_m) * b, np.sqrt(n_th * Gam_m) * bd]

print(f"g_eff=2g={2*g_dim:.5f}, eps={eps:.5f}, Nb={Nb}")
print("Calculando propagador de Floquet...")
U = propagator(H, T_m, c_ops, options=opts)
print(f"... listo. shape={U.shape}")

evals, evecs = U.eigenstates()
orden = np.argsort(-np.abs(evals))
evals = evals[orden]
evecs = [evecs[i] for i in orden]

# Operadores de referencia (extendidos al espacio completo)
P_osc = Qobj(np.diag((-1.0) ** np.arange(Nb)))
P_full = tensor(qeye(Na), P_osc)
n_full = bd * b
a_full = b
a2_full = b * b
sz_ref = sz_full  # ya es sigma_z extendido con identidad en el oscilador

operadores = {
    "P (paridad)": P_full,
    "n (numero)": n_full,
    "a (bit-flip/coherencia)": a_full,
    "a^2 (confinamiento)": a2_full,
    "sigma_z (qubit)": sz_ref,
}

n_modos = 8
print(f"\n=== Primeros {n_modos} modos: overlaps y clasificacion ===")
resultados = []
for k in range(n_modos):
    mu = evals[k]
    if abs(mu) > 1 - 1e-13:
        lam = 0.0 + 0.0j
    else:
        lam = -np.log(mu) / T_m
    Xk = vector_to_operator(evecs[k])

    overlaps = {}
    for nombre, op in operadores.items():
        ov = abs((op.dag() * Xk).tr())
        overlaps[nombre] = ov

    # normalizar por la norma de Xk para comparar overlaps de forma justa
    normXk = Xk.norm()
    overlaps_norm = {k2: v / normXk for k2, v in overlaps.items()}

    clasificacion = max(overlaps_norm, key=overlaps_norm.get) if k > 0 else "trivial (mu=1)"

    print(f"\nModo {k}: mu={mu:.6f}  |mu|={abs(mu):.6f}  lambda={lam.real:.6e}"
          f"{'+' if lam.imag>=0 else ''}{lam.imag:.6e}j")
    for nombre, ov in overlaps_norm.items():
        print(f"    overlap {nombre:<26} = {ov:.5e}")
    print(f"    --> Clasificacion: {clasificacion}")

    resultados.append(dict(k=k, mu=mu, lam=lam, overlaps=overlaps_norm, clase=clasificacion))

print("\n=== TABLA RESUMEN (Tarea 20) ===")
print(f"{'k':>3} {'lambda_real':>13} {'clasificacion':>28}")
for r_ in resultados:
    print(f"{r_['k']:>3} {r_['lam'].real:>13.6e} {r_['clase']:>28}")

# Verificar si el modo 4 es la brecha de confinamiento
if len(resultados) > 4:
    modo4 = resultados[4]
    es_confinamiento = "n (numero)" in modo4['clase'] or "a^2" in modo4['clase']
    print(f"\nModo 4: lambda={modo4['lam'].real:.6e}, clasificado como '{modo4['clase']}'")
    print("Confirmacion 'modo 4 es la brecha de confinamiento':",
          "SI" if es_confinamiento else "NO")

np.savez("tarea20_resultados.npz",
         evals=np.array([complex(r_['mu']) for r_ in resultados]),
         lambdas=np.array([complex(r_['lam']) for r_ in resultados]),
         clases=np.array([r_['clase'] for r_ in resultados], dtype=object))
print("\nGuardado: tarea20_resultados.npz")
