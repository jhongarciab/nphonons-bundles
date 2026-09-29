# -*- coding: utf-8 -*-
"""
Tarea 6a (linea base): modelo completo de fig2_git_v1.py (Omega = 4g),
con tolerancias estrictas (atol=1e-10, rtol=1e-8), guardando ademas:

    p_e(t) = <(1+sigma_z)/2>
    |<sigma_->(t)>|
    <n>(t)

Se compara p_e en estado estacionario contra la formula de saturacion de un
qubit driveado resonantemente:

    p_e_ss = eps^2 / (2 eps^2 + kappa^2/4),   eps = Omega_full = 4g

Validacion tras el mesolve: traza, hermiticidad, positividad (umbral -1e-9)
del estado reducido del oscilador Y del estado reducido del qubit.

NO se modifica fig2_git_v1.py.
"""

import numpy as np
from qutip import (tensor, qeye, destroy, thermal_dm, mesolve,
                    sigmam, sigmaz, sigmax, Options, ptrace)

# ------------------------------------------------------------
# Parametros (identicos a fig2_git_v1.py / tarea1 / tarea2)
# ------------------------------------------------------------
r = 0.1
gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
gz_d, om_m, Gam_m = gz_org / kappa_org, omega_m_org / kappa_org, Gamma_m_org / kappa_org
kap = 1.0
n_th = 0

gx_dim = r * gz_d
g_dim = gz_d * gx_dim / om_m
Omega_full = 4 * g_dim   # eps del enunciado = drive real = 4g

print(f"g = {g_dim:.6f} kappa,  Omega_full = eps = {Omega_full:.6f} kappa")

Na, Nb = 2, 50
b = tensor(qeye(Na), destroy(Nb))
bd = b.dag()
sm = tensor(sigmam(), qeye(Nb))
sz = tensor(sigmaz(), qeye(Nb))
sx = tensor(sigmax(), qeye(Nb))
omega_q_dim = 2 * om_m
omega_d_dim = 2 * om_m

H0 = 0.5 * omega_q_dim * sz
H = [
    H0,
    [gx_dim * sx * b, lambda t, _: np.exp(-1j * om_m * t)],
    [gx_dim * sx * bd, lambda t, _: np.exp(+1j * om_m * t)],
    [gz_d * sz * b, lambda t, _: np.exp(-1j * om_m * t)],
    [gz_d * sz * bd, lambda t, _: np.exp(+1j * om_m * t)],
    [Omega_full * sx, lambda t, _: np.exp(+1j * omega_d_dim * t)],
    [Omega_full * sx, lambda t, _: np.exp(-1j * omega_d_dim * t)],
]

from qutip import lindblad_dissipator
diss_full = [kap * lindblad_dissipator(sm),
             (n_th + 1) * Gam_m * lindblad_dissipator(b),
             n_th * Gam_m * lindblad_dissipator(bd)]

rho0_full = tensor(thermal_dm(Na, 0), thermal_dm(Nb, 0))

tau_max = 60.0     # extendido respecto a tarea1/2 (39) para asegurar estado estacionario
n_steps = 150
tlist = np.linspace(0, tau_max, n_steps)
options = Options(nsteps=2_000_000, store_states=True, atol=1e-10, rtol=1e-8)

print(f"\nCorriendo modelo completo (tau_max={tau_max}, atol=1e-10, rtol=1e-8)...")
res = mesolve(H, rho0_full, tlist, diss_full, [sz, sm, bd * b], options=options)
print("... listo.")

sz_t = res.expect[0]
sm_t = res.expect[1]     # <sigma_->  (complejo)
n_t = res.expect[2]

pe_t = (1 + sz_t.real) / 2
abs_sm_t = np.abs(sm_t)

# ------------------------------------------------------------
# Validacion: traza, hermiticidad, positividad (umbral -1e-9)
# ------------------------------------------------------------
def validar_estado(rho):
    tr = rho.tr()
    herm_err = (rho - rho.dag()).norm()
    min_eig = np.min(rho.eigenenergies().real)
    ok = abs(tr - 1) < 1e-6 and herm_err < 1e-8 and min_eig > -1e-9
    return tr, herm_err, min_eig, ok

idx_check = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
print("\n=== VALIDACION en instantes seleccionados ===")
falla_alguna = False
for idx in idx_check:
    rho_full = res.states[idx]
    rho_osc = ptrace(rho_full, 1)
    rho_qb = ptrace(rho_full, 0)
    tr_o, he_o, me_o, ok_o = validar_estado(rho_osc)
    tr_q, he_q, me_q, ok_q = validar_estado(rho_qb)
    ok = ok_o and ok_q
    falla_alguna = falla_alguna or (not ok)
    print(f"  t={tlist[idx]:6.2f}  osc: tr={tr_o:.3e} herm={he_o:.3e} mineig={me_o:.3e} "
          f"| qubit: tr={tr_q:.3e} herm={he_q:.3e} mineig={me_q:.3e}  "
          f"{'OK' if ok else 'FALLA'}")
print("Resultado global:", "FALLA en al menos un instante" if falla_alguna else "OK (umbral -1e-9)")

# ------------------------------------------------------------
# Verificacion de estado estacionario: dn/dt ~ 0
# ------------------------------------------------------------
dndt = np.gradient(n_t, tlist)
print(f"\ndn/dt en t_final = {dndt[-1]:.3e}   (<n> final = {n_t[-1]:.4f})")
print(f"dn/dt promedio ultimo 10% del tiempo = {np.mean(np.abs(dndt[-n_steps//10:])):.3e}")

# ------------------------------------------------------------
# Comparacion con formula de saturacion
# ------------------------------------------------------------
pe_ss_medido = np.mean(pe_t[-10:])
pe_ss_formula = Omega_full**2 / (2 * Omega_full**2 + kap**2 / 4)

print(f"\np_e estacionario (medido, promedio ultimos 10 puntos) = {pe_ss_medido:.6f}")
print(f"p_e estacionario (formula eps^2/(2eps^2+kappa^2/4))   = {pe_ss_formula:.6f}")
print(f"Diferencia relativa = {abs(pe_ss_medido - pe_ss_formula)/pe_ss_formula:.4%}")

print(f"\n|<sigma_->| estacionario (medido) = {np.mean(abs_sm_t[-10:]):.6f}")
print(f"n estacionario (medido)           = {np.mean(n_t[-10:]):.6f}")

np.savez("tarea6a_resultados.npz",
         tlist=tlist, sz_t=sz_t, sm_t=sm_t, n_t=n_t, pe_t=pe_t, abs_sm_t=abs_sm_t,
         pe_ss_medido=pe_ss_medido, pe_ss_formula=pe_ss_formula,
         Omega_full=Omega_full, g_dim=g_dim)
print("\nGuardado: tarea6a_resultados.npz")
