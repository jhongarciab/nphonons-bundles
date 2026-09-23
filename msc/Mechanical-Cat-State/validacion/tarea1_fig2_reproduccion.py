# -*- coding: utf-8 -*-
"""
Tarea 1 (sanidad): reproduce fig2_git_v1.py tal cual (SIN modificar el original).

Modelo completo:   qubit + oscilador, drive Omega = 4*g   (sigma_x)
Modelo efectivo:    solo oscilador, squeezing con eps = 2*g

Tras cada mesolve se verifica, para el estado reducido del oscilador
(ptrace del modelo completo, y estado del modelo efectivo):
    - traza == 1
    - hermiticidad: || rho - rho^dagger ||
    - positividad: autovalor minimo > -1e-10

Guarda <n>(t) y fidelidad(t) en tarea1_fig2_resultados.npz
"""

import numpy as np
from qutip import (destroy, thermal_dm, mesolve, tensor, qeye,
                    sigmam, sigmap, sigmaz, sigmax, Options, ptrace)
from qutip.metrics import fidelity as state_fidelity

# ------------------------------------------------------------
# 0) Ajustes de simulacion (identicos a fig2_git_v1.py)
# ------------------------------------------------------------
tau_max = 39.0
n_steps = 120
tau = np.linspace(0, tau_max, n_steps)
options = Options(nsteps=1_000_000, store_states=True)

# ------------------------------------------------------------
# 1) MODELO EFECTIVO (solo oscilador)
# ------------------------------------------------------------
N_eff = 50
a = destroy(N_eff)
adag = a.dag()

gz = 2 * np.pi * 6e6
r = 0.1
gx = r * gz
omega_m = 2 * np.pi * 100e6
omega_q = 2 * omega_m
kappa = 2 * np.pi * 100e3
gamma_m = 2 * np.pi * 15
n_th = 0

g = gz * gx / omega_m          # acoplamiento efectivo de dos fonones
eps = 2 * g                    # drive efectivo (paper: eps = 2g)
chi = -2j * eps * g / kappa

x = kappa / 2
D1m, D1p = omega_q - omega_m, omega_q + omega_m
D2m, D2p = omega_q - 2 * omega_m, omega_q + 2 * omega_m

G1m = 2 * gx**2 * x / (x**2 + D1m**2)
G1p = 2 * gx**2 * x / (x**2 + D1p**2)
Gp = n_th * gamma_m + G1p

G2m = 2 * g**2 * x / (x**2 + D2m**2)
G2p = 2 * g**2 * x / (x**2 + D2p**2)

dk = g**2 * (-D2m / (x**2 + D2m**2) - D2p / (x**2 + D2p**2))

from qutip import lindblad_dissipator

H_eff = chi.conjugate() * adag**2 + chi * a**2 + dk * (adag * a)**2
diss_eff = [G1m * lindblad_dissipator(a),
            G1p * lindblad_dissipator(adag),
            G2m * lindblad_dissipator(a**2),
            G2p * lindblad_dissipator(adag**2)]

rho0_eff = thermal_dm(N_eff, 0)
tlist_eff = tau / kappa
res_eff = mesolve(H_eff, rho0_eff, tlist_eff, diss_eff, [adag * a], options=options)
nb_eff = res_eff.expect[0]
states_eff = res_eff.states

# ------------------------------------------------------------
# 2) MODELO COMPLETO (qubit + oscilador), Omega = 4g
# ------------------------------------------------------------
Na, Nb = 2, 50
b = tensor(qeye(Na), destroy(Nb))
bd = b.dag()

gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
gz_d, om_m, Gam_m = gz_org / kappa_org, omega_m_org / kappa_org, Gamma_m_org / kappa_org
kap = 1.0

gx_dim = r * gz_d
g_dim = gz_d * gx_dim / om_m
Omega = 4 * g_dim               # drive REAL del modelo completo: Omega = 4g

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
    [Omega * sx, lambda t, _: np.exp(+1j * omega_d_dim * t)],
    [Omega * sx, lambda t, _: np.exp(-1j * omega_d_dim * t)],
]

diss_full = [kap * lindblad_dissipator(sm),
             (n_th + 1) * Gam_m * lindblad_dissipator(b),
             n_th * Gam_m * lindblad_dissipator(bd)]

rho0_full = tensor(thermal_dm(Na, 0), thermal_dm(Nb, 0))
tlist_full = tau
res_full = mesolve(H, rho0_full, tlist_full, diss_full, [bd * b], options=options)
nb_full = res_full.expect[0]
states_full = res_full.states
states_full_red = [ptrace(rho, 1) for rho in states_full]

# ------------------------------------------------------------
# 3) VALIDACION: traza, hermiticidad, positividad
# ------------------------------------------------------------
def validar_estado(rho, nombre, idx):
    tr = rho.tr()
    herm_err = (rho - rho.dag()).norm()
    eigs = rho.eigenenergies()
    min_eig = np.min(eigs.real)
    ok_tr = abs(tr - 1) < 1e-6
    ok_herm = herm_err < 1e-8
    ok_pos = min_eig > -1e-10
    if not (ok_tr and ok_herm and ok_pos):
        print(f"  [FALLO] {nombre}[{idx}]: tr={tr:.3e} herm_err={herm_err:.3e} min_eig={min_eig:.3e}")
    return tr, herm_err, min_eig, (ok_tr and ok_herm and ok_pos)

print("Validando estados reducidos del modelo completo (ptrace oscilador)...")
resumen_full = [validar_estado(r, "full_red", i) for i, r in enumerate(states_full_red)]
print("Validando estados del modelo efectivo...")
resumen_eff = [validar_estado(r, "eff", i) for i, r in enumerate(states_eff)]

trs_full = np.array([x[0] for x in resumen_full])
herm_full = np.array([x[1] for x in resumen_full])
mineig_full = np.array([x[2] for x in resumen_full])
ok_full = all(x[3] for x in resumen_full)

trs_eff = np.array([x[0] for x in resumen_eff])
herm_eff = np.array([x[1] for x in resumen_eff])
mineig_eff = np.array([x[2] for x in resumen_eff])
ok_eff = all(x[3] for x in resumen_eff)

print("\n=== RESUMEN VALIDACION ===")
print(f"Full  (reducido): max|tr-1|={np.max(np.abs(trs_full-1)):.3e}  "
      f"max herm_err={np.max(herm_full):.3e}  min autovalor={np.min(mineig_full):.3e}  "
      f"{'OK' if ok_full else 'FALLO'}")
print(f"Eff             : max|tr-1|={np.max(np.abs(trs_eff-1)):.3e}  "
      f"max herm_err={np.max(herm_eff):.3e}  min autovalor={np.min(mineig_eff):.3e}  "
      f"{'OK' if ok_eff else 'FALLO'}")

# ------------------------------------------------------------
# 4) Fidelidad vs tiempo
# ------------------------------------------------------------
fidelity_t = np.array([state_fidelity(states_full_red[i], states_eff[i]) for i in range(len(tau))])

print("\n=== RESULTADOS FISICOS ===")
print(f"<n>_full(0)  = {nb_full[0]:.4f}   <n>_eff(0)  = {nb_eff[0]:.4f}")
print(f"<n>_full(fin)= {nb_full[-1]:.4f}   <n>_eff(fin)= {nb_eff[-1]:.4f}")
print(f"Fidelidad minima = {fidelity_t.min():.6f}  (en kappa t = {tau[np.argmin(fidelity_t)]:.2f})")
print(f"Fidelidad final (kappa t = {tau[-1]:.2f}) = {fidelity_t[-1]:.6f}")

# ------------------------------------------------------------
# 5) Guardar resultados
# ------------------------------------------------------------
np.savez(
    "tarea1_fig2_resultados.npz",
    tau=tau,
    nb_full=nb_full,
    nb_eff=nb_eff,
    fidelity_t=fidelity_t,
    trs_full=trs_full, herm_full=herm_full, mineig_full=mineig_full,
    trs_eff=trs_eff, herm_eff=herm_eff, mineig_eff=mineig_eff,
)
print("\nGuardado: tarea1_fig2_resultados.npz")
