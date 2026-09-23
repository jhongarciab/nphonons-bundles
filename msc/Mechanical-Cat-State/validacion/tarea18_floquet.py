# -*- coding: utf-8 -*-
"""
Tarea 18: analisis de Floquet del propagador de un periodo.

Como omega_q = omega_d = 2*omega_m, el Liouvilliano es periodico con
T_m = 2*pi/omega_m. Se calcula U = qutip.propagator(H, T_m, c_ops,
options) (superoperador, atol=1e-12, rtol=1e-10) y se obtiene:

  (a) el punto fijo (autovalor 1 de U), normalizado a traza 1; se valida
      traza, hermiticidad y positividad (umbral -1e-8).
  (b) los 6 autovalores de mayor modulo, convertidos a tasas
      lambda_k = -ln(mu_k)/T_m.

Caso (i): eps=0, g_z y g_x del paper (Nb=16). <n> del punto fijo,
comparado con (g_z/omega_m)^2=0.0036 y con el piso de la Tarea 16 (6.19e-3).

Caso (ii): eps=1.44 (punto del paper, Nb=20). Paridad del punto fijo,
|<a^2>|, tasas lentas, e identificacion del modo de decaimiento de
paridad (proyectando cada autovector sobre el operador de paridad).

NO se modifica ningun script original.
"""

import numpy as np
from qutip import (tensor, qeye, destroy, sigmam, sigmaz, sigmax, lindblad_dissipator,
                    Options, propagator, vector_to_operator, Qobj)

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

opts = Options(atol=1e-12, rtol=1e-10, nsteps=2_000_000)


def construir_H_c_ops(Nb, eps):
    b = tensor(qeye(Na), destroy(Nb))
    bd = b.dag()
    sm = tensor(sigmam(), qeye(Nb))
    sz = tensor(sigmaz(), qeye(Nb))
    sx = tensor(sigmax(), qeye(Nb))
    H0 = 0.5 * omega_q_dim * sz
    H = [H0,
         [gx_dim * sx * b, lambda t, _: np.exp(-1j * om_m * t)],
         [gx_dim * sx * bd, lambda t, _: np.exp(+1j * om_m * t)]]
    H += [[gz_d * sz * b, lambda t, _: np.exp(-1j * om_m * t)],
          [gz_d * sz * bd, lambda t, _: np.exp(+1j * om_m * t)]]
    if eps != 0:
        H += [[eps * sx, lambda t, _: np.exp(+1j * omega_d_dim * t)],
              [eps * sx, lambda t, _: np.exp(-1j * omega_d_dim * t)]]
    c_ops = [np.sqrt(kap) * sm, np.sqrt((n_th + 1) * Gam_m) * b, np.sqrt(n_th * Gam_m) * bd]
    return H, c_ops, b, bd, sz


def validar_estado(rho):
    tr = rho.tr()
    herm_err = (rho - rho.dag()).norm()
    min_eig = np.min(rho.eigenenergies().real)
    ok = abs(tr - 1) < 1e-6 and herm_err < 1e-8 and min_eig > -1e-8
    return tr, herm_err, min_eig, ok


def analizar_floquet(Nb, eps, etiqueta, n_modos=6):
    print(f"\n=== {etiqueta}: Nb={Nb}, eps={eps} ===")
    H, c_ops, b, bd, sz = construir_H_c_ops(Nb, eps)

    print("Calculando propagador de un periodo...")
    U = propagator(H, T_m, c_ops, options=opts)
    print(f"... listo. U shape={U.shape}")

    evals, evecs = U.eigenstates()
    # Ordenar por modulo descendente
    orden = np.argsort(-np.abs(evals))
    evals = evals[orden]
    evecs = [evecs[i] for i in orden]

    print(f"Autovalor de mayor modulo: {evals[0]:.8f}  (esperado ~1)")

    # Punto fijo: reshape a operador, hermitizar, normalizar traza
    rho_fixed_raw = vector_to_operator(evecs[0])
    rho_fixed = (rho_fixed_raw + rho_fixed_raw.dag()) / 2
    rho_fixed = rho_fixed / rho_fixed.tr()

    tr, herm_err, min_eig, ok = validar_estado(rho_fixed)
    print(f"Punto fijo: tr={tr:.6f}  herm_err={herm_err:.3e}  mineig={min_eig:.3e}  "
          f"{'OK' if ok else 'FALLA umbral -1e-8'}")

    n_op = bd * b
    n_fixed = (n_op * rho_fixed).tr().real
    print(f"<n> del punto fijo = {n_fixed:.6f}")

    # Tasas de los primeros n_modos autovalores (incluye el trivial mu=1)
    print(f"\nPrimeros {n_modos} autovalores y tasas lambda=-ln(mu)/T_m:")
    tasas = []
    for k in range(n_modos):
        mu = evals[k]
        if abs(mu) < 1e-14:
            lam = np.inf
        else:
            lam = -np.log(mu) / T_m   # complejo en general
        print(f"  mu_{k} = {mu:.6f}   |mu|={abs(mu):.6f}   lambda={lam:.6e}  "
              f"(Re={lam.real:.6e}, Im={lam.imag:.6e})")
        tasas.append((mu, lam))

    return dict(rho_fixed=rho_fixed, n_fixed=n_fixed, evals=evals, evecs=evecs,
                tasas=tasas, b=b, bd=bd, sz=sz, U=U, Nb=Nb)


# ------------------------------------------------------------
# Caso (i): eps=0, parametros del paper, Nb=16
# ------------------------------------------------------------
res_i = analizar_floquet(16, 0.0, "CASO (i): eps=0, parametros del paper")

pred_quench = (gz_d / om_m) ** 2
piso_tarea16 = 6.19010e-3
print(f"\nComparacion caso (i):")
print(f"  <n>_fijo (Floquet)        = {res_i['n_fixed']:.6f}")
print(f"  (g_z/omega_m)^2           = {pred_quench:.6f}")
print(f"  Piso medido Tarea 16      = {piso_tarea16:.6f}")
print(f"  Razon n_fijo/(gz/wm)^2    = {res_i['n_fixed']/pred_quench:.4f}")
print(f"  Razon n_fijo/piso_T16     = {res_i['n_fixed']/piso_tarea16:.4f}")
if abs(res_i['n_fixed'] - pred_quench) < 0.3 * pred_quench:
    diag = "El piso es RELAJACION al estado base vestido (n_fijo ~ (gz/wm)^2)"
elif res_i['n_fixed'] > 3 * pred_quench:
    diag = "El piso es CALENTAMIENTO con un estacionario caliente (n_fijo >> (gz/wm)^2)"
else:
    diag = "Ambiguo / intermedio"
print(f"  Diagnostico: {diag}")

# ------------------------------------------------------------
# Caso (ii): eps=1.44, Nb=20
# ------------------------------------------------------------
eps_paper = 4 * g_dim
res_ii = analizar_floquet(20, eps_paper, "CASO (ii): eps=1.44 (punto del paper)")

Nb_ii = res_ii['Nb']
P_op = Qobj(np.diag((-1.0) ** np.arange(Nb_ii)))
P_op_full = tensor(qeye(Na), P_op)  # operador de paridad en el espacio completo (qubit+osc)
# proyectamos la paridad del OSCILADOR reducido; para el punto fijo completo,
# usamos el operador de paridad que actua solo en la parte del oscilador,
# extendido con identidad en el qubit, y luego trazamos sobre el qubit.

from qutip import ptrace
rho_fixed_full = res_ii['rho_fixed']
rho_osc_fixed = ptrace(rho_fixed_full, 1)
P_osc = Qobj(np.diag((-1.0) ** np.arange(Nb_ii)))
paridad_fixed = (P_osc * rho_osc_fixed).tr().real
a_op = destroy(Nb_ii)
a2_fixed = (a_op * a_op * rho_osc_fixed).tr()

print(f"\nCaso (ii) -- punto fijo:")
print(f"  Paridad (oscilador reducido) = {paridad_fixed:.6f}")
print(f"  <a^2> = {a2_fixed:.6f}   |<a^2>|={abs(a2_fixed):.6f}")

# Identificar el modo de decaimiento de paridad: proyectar cada autovector
# (reshape a operador en el espacio COMPLETO qubit+osc) sobre P_op_full,
# y tomar el de mayor |overlap| entre los modos no triviales (k>=1).
print(f"\nProyeccion de los modos sobre el operador de paridad (Tr[P^dagger X_k]):")
overlaps = []
for k in range(1, len(res_ii['tasas'])):
    Xk = vector_to_operator(res_ii['evecs'][k])
    overlap = (P_op_full.dag() * Xk).tr()
    overlaps.append(abs(overlap))
    mu_k, lam_k = res_ii['tasas'][k]
    print(f"  modo {k}: mu={mu_k:.6f}  |overlap con P|={abs(overlap):.4e}  "
          f"lambda={lam_k.real:.6e}")

idx_paridad = int(np.argmax(overlaps)) + 1  # +1 porque overlaps empieza en k=1
mu_paridad, lam_paridad = res_ii['tasas'][idx_paridad]
gamma_phase_flip = lam_paridad.real
print(f"\nModo identificado como decaimiento de paridad: k={idx_paridad}, "
      f"mu={mu_paridad:.6f}")
print(f"gamma_phase-flip = {gamma_phase_flip:.6e}  (unidades de kappa)")

np.savez("tarea18_resultados.npz",
         n_fixed_i=res_i['n_fixed'], pred_quench=pred_quench,
         paridad_fixed_ii=paridad_fixed, a2_fixed_ii=a2_fixed,
         gamma_phase_flip=gamma_phase_flip,
         evals_i=np.array([complex(e) for e in res_i['evals'][:6]]),
         evals_ii=np.array([complex(e) for e in res_ii['evals'][:6]]))
print("\nGuardado: tarea18_resultados.npz")
