# -*- coding: utf-8 -*-
"""
Tarea 12: estado oscuro en el estacionario de fig2 (eps=1.44=Omega_full).

En el estacionario, reporta: p_e, |<sigma_->|, <a^2> (con fase), paridad.
Compara <a^2> con la prediccion del estado oscuro de magnitud eps/g_eff
(el estado "oscuro" del proceso de dos fonones, para el cual la parte
disipativa de dos fotones se anula, es un gato de amplitud alpha^2 tal que
chi*alpha^2 (parte de squeezing) equilibra la relacion de balance detallado;
la magnitud esperada es |alpha|^2 = eps/g_eff, ver Tareas 7-8, 11).

Se calcula la fidelidad con el gato PAR |C+> de amplitud alpha^2=<a^2>
(deduciendo la fase directamente de la <a^2> medida, no asumida).

Usa muestreo estroboscopico y validacion (traza, hermiticidad,
positividad, umbral -1e-9). NO modifica ningun script original.
"""

import numpy as np
from qutip import (tensor, qeye, destroy, thermal_dm, mesolve, lindblad_dissipator,
                    sigmam, sigmaz, sigmax, Options, ptrace, expect, coherent, Qobj)
from qutip.metrics import fidelity as state_fidelity

r = 0.1
gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
gz_d, om_m, Gam_m = gz_org / kappa_org, omega_m_org / kappa_org, Gamma_m_org / kappa_org
kap = 1.0
n_th = 0
gx_dim = r * gz_d
g_dim = gz_d * gx_dim / om_m
g_eff_2g = 2 * g_dim
eps = 4 * g_dim   # = Omega_full = 1.44

Na, Nb_full = 2, 30
omega_q_dim = 2 * om_m
omega_d_dim = 2 * om_m
T_m = 2 * np.pi / om_m

print(f"g_eff=2g = {g_eff_2g:.6f}, eps = {eps:.6f}, eps/g_eff = {eps/g_eff_2g:.6f}")

options = Options(nsteps=2_000_000, atol=1e-10, rtol=1e-8, store_states=True)


def tlist_estrobo(tau_max, n_puntos=100):
    K_total = int(round(tau_max / T_m))
    stride = max(1, K_total // n_puntos)
    m = np.arange(0, K_total + 1, stride)
    return m * T_m


tau_max = 60.0   # suficiente para estado estacionario (Tarea 6/9: t90~7 max)
tlist = tlist_estrobo(tau_max)

b = tensor(qeye(Na), destroy(Nb_full))
bd = b.dag()
sm = tensor(sigmam(), qeye(Nb_full))
sz = tensor(sigmaz(), qeye(Nb_full))
sx = tensor(sigmax(), qeye(Nb_full))
H0 = 0.5 * omega_q_dim * sz
H = [H0,
     [gx_dim * sx * b, lambda t, _: np.exp(-1j * om_m * t)],
     [gx_dim * sx * bd, lambda t, _: np.exp(+1j * om_m * t)],
     [gz_d * sz * b, lambda t, _: np.exp(-1j * om_m * t)],
     [gz_d * sz * bd, lambda t, _: np.exp(+1j * om_m * t)],
     [eps * sx, lambda t, _: np.exp(+1j * omega_d_dim * t)],
     [eps * sx, lambda t, _: np.exp(-1j * omega_d_dim * t)]]
diss_full = [kap * lindblad_dissipator(sm),
             (n_th + 1) * Gam_m * lindblad_dissipator(b),
             n_th * Gam_m * lindblad_dissipator(bd)]
rho0_full = tensor(thermal_dm(Na, 0), thermal_dm(Nb_full, 0))

print("Corriendo modelo completo hasta estado estacionario...")
res = mesolve(H, rho0_full, tlist, diss_full, [sz, sm, bd * b, b * b], options=options)
print("... listo.")

sz_t, sm_t, n_t, a2_t = res.expect

pe_t = (1 + sz_t.real) / 2

rho_ss = res.states[-1]
rho_osc_ss = ptrace(rho_ss, 1)


def validar_estado(rho):
    tr = rho.tr()
    herm_err = (rho - rho.dag()).norm()
    min_eig = np.min(rho.eigenenergies().real)
    ok = abs(tr - 1) < 1e-6 and herm_err < 1e-8 and min_eig > -1e-9
    return tr, herm_err, min_eig, ok


tr, herm_err, min_eig, ok = validar_estado(rho_osc_ss)
print(f"\nValidacion estado estacionario del oscilador: tr={tr:.3e} herm={herm_err:.3e} "
      f"mineig={min_eig:.3e}  {'OK' if ok else 'FALLA umbral -1e-9'}")

# Paridad
P_op = Qobj(np.diag((-1.0) ** np.arange(Nb_full)))
paridad_ss = expect(P_op, rho_osc_ss)

pe_ss = np.mean(pe_t[-8:])
sm_ss = np.mean(sm_t[-8:])
n_ss = np.mean(n_t[-8:])
a2_ss = np.mean(a2_t[-8:])   # <a^2> COMPLEJO, con fase

print(f"\n=== ESTACIONARIO (eps=1.44) ===")
print(f"p_e_ss           = {pe_ss:.6e}")
print(f"|<sigma_->|_ss    = {abs(sm_ss):.6e}")
print(f"<n>_ss            = {n_ss:.6f}")
print(f"<a^2>_ss          = {a2_ss:.6f}   (|<a^2>|={abs(a2_ss):.6f}, fase={np.angle(a2_ss):.6f} rad)")
print(f"Paridad_ss        = {paridad_ss:.6f}")

# Prediccion del estado oscuro: |alpha|^2 = eps/g_eff
alpha2_pred_mag = eps / g_eff_2g
print(f"\nPrediccion |alpha^2| = eps/g_eff = {alpha2_pred_mag:.6f}")
print(f"|<a^2>|_ss medido    = {abs(a2_ss):.6f}")
print(f"Razon medido/prediccion (magnitud) = {abs(a2_ss)/alpha2_pred_mag:.4f}")

# Construir el gato PAR |C+> con alpha^2 = <a^2>_ss medido (fase incluida):
# alpha = sqrt(<a^2>_ss) (una de las dos raices; se elige la de argumento/2)
alpha_medido = np.sqrt(a2_ss)   # numpy toma la raiz principal
cat_plus = (coherent(Nb_full, alpha_medido) + coherent(Nb_full, -alpha_medido)).unit()

F_cat = state_fidelity(rho_osc_ss, cat_plus * cat_plus.dag())
print(f"\nalpha (deducido de <a^2>_ss) = {alpha_medido:.6f}")
print(f"Fidelidad con gato par |C+> de amplitud alpha (alpha^2=<a^2>_ss) = {F_cat:.6f}")

# Tambien probar signo opuesto de alpha (mismo alpha^2, pero por si la
# convencion de raiz no es la fisicamente relevante -- para un gato par da
# el MISMO estado, se reporta igual para verificar)
cat_plus_alt = (coherent(Nb_full, -alpha_medido) + coherent(Nb_full, alpha_medido)).unit()
F_cat_alt = state_fidelity(rho_osc_ss, cat_plus_alt * cat_plus_alt.dag())
print(f"(control, deberia ser igual) F_cat_alt = {F_cat_alt:.6f}")

np.savez("tarea12_resultados.npz",
         pe_ss=pe_ss, sm_ss=sm_ss, n_ss=n_ss, a2_ss=a2_ss, paridad_ss=paridad_ss,
         alpha2_pred_mag=alpha2_pred_mag, alpha_medido=complex(alpha_medido), F_cat=F_cat,
         min_eig=min_eig)
print("\nGuardado: tarea12_resultados.npz")
