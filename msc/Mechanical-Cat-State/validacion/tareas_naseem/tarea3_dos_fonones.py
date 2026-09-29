# -*- coding: utf-8 -*-
"""
Tarea 3: aislar el termino de dos fonones.

Se simula el modelo COMPLETO (qubit + oscilador) SIN drive (Omega = 0),
partiendo de |g> tensor |n=4>, y se mide la tasa de decaimiento de <n>
inducida por el proceso virtual de dos fonones (mediado por el qubit y su
disipacion kappa), SIN ningun termino de disipacion de dos fonones puesto
"a mano": en el modelo completo no existe un operador a^2 explicito, el
efecto de dos fonones emerge solo de la dinamica coherente qubit-oscilador
mas la disipacion del qubit (kappa) y la del oscilador (gamma_m, pequena).

Se compara la tasa medida contra la formula analitica de la Ec. efectiva
    Gamma2_minus(g_eff) = 2 * g_eff^2 * Re S_{2-},   Re S_{2-} = x/(x^2+Delta2_minus^2)
para g_eff = g (definicion original del codigo) y g_eff = 2g (hipotesis del
factor 2 de la Tarea 2), para determinar directamente cual g_eff es
consistente con la dinamica completa.

NO se modifica ningun script original; este es un script nuevo en ./validacion/.
"""

import numpy as np
from qutip import (destroy, basis, tensor, qeye, mesolve, expect,
                    sigmam, sigmaz, sigmax, Options, ptrace, lindblad_dissipator)

# ------------------------------------------------------------
# 0) Parametros fisicos (mismas unidades de kappa que fig2/tarea1/tarea2)
# ------------------------------------------------------------
r = 0.1
gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
gz_d, om_m, Gam_m = gz_org / kappa_org, omega_m_org / kappa_org, Gamma_m_org / kappa_org
kap = 1.0
gx_dim = r * gz_d
g_dim = gz_d * gx_dim / om_m          # g del codigo (unidades de kappa)
n_th = 0

Na, Nb = 2, 18         # margen de seguridad: hay oscilaciones coherentes por encima de n0=4
omega_q_dim = 2 * om_m  # resonancia de dos fonones (igual que en fig1..fig5)

x = kap / 2.0
D2m = omega_q_dim - 2 * om_m            # = 0 (resonancia exacta)
ReS2m = x / (x**2 + D2m**2)             # = 1/x = 2

Gamma2_g   = 2 * g_dim**2 * ReS2m       # formula con g_eff = g
Gamma2_2g  = 2 * (2 * g_dim)**2 * ReS2m  # formula con g_eff = 2g  (4x mayor)

print(f"g (codigo)      = {g_dim:.6f}  (unidades kappa)")
print(f"Gamma2_minus(g)  = {Gamma2_g:.6f}")
print(f"Gamma2_minus(2g) = {Gamma2_2g:.6f}")

# ------------------------------------------------------------
# 1) Hamiltoniano COMPLETO, SIN drive (Omega = 0)
# ------------------------------------------------------------
b = tensor(qeye(Na), destroy(Nb))
bd = b.dag()
n_op = bd * b
sm = tensor(sigmam(), qeye(Nb))
sz = tensor(sigmaz(), qeye(Nb))
sx = tensor(sigmax(), qeye(Nb))

H0 = 0.5 * omega_q_dim * sz
H = [
    H0,
    [gx_dim * sx * b,  lambda t, _: np.exp(-1j * om_m * t)],
    [gx_dim * sx * bd, lambda t, _: np.exp(+1j * om_m * t)],
    [gz_d * sz * b,    lambda t, _: np.exp(-1j * om_m * t)],
    [gz_d * sz * bd,   lambda t, _: np.exp(+1j * om_m * t)],
    # SIN termino de drive Omega
]

diss_full = [kap * lindblad_dissipator(sm),
             (n_th + 1) * Gam_m * lindblad_dissipator(b),
             n_th * Gam_m * lindblad_dissipator(bd)]

# ------------------------------------------------------------
# 2) Estado inicial |g> tensor |n=4>, evolucion corta
# ------------------------------------------------------------
n0 = 4
psi0 = tensor(basis(Na, 0), basis(Nb, n0))   # |g> (convencion del codigo) x |4>

tau_max = 15.0
n_steps = 600
tlist = np.linspace(0, tau_max, n_steps)
options = Options(nsteps=1_000_000, store_states=True)

print("\nCorriendo modelo COMPLETO sin drive, |g>x|4>...")
res = mesolve(H, psi0, tlist, diss_full, [n_op], options=options)
nb_t = res.expect[0]
print("... listo.")

# ------------------------------------------------------------
# 3) Validacion (traza, hermiticidad, positividad) del estado
#    reducido del oscilador
# ------------------------------------------------------------
def validar_estado(rho, idx):
    tr = rho.tr()
    herm_err = (rho - rho.dag()).norm()
    min_eig = np.min(rho.eigenenergies().real)
    ok = abs(tr - 1) < 1e-6 and herm_err < 1e-8 and min_eig > -1e-10
    return tr, herm_err, min_eig, ok

states_red = [ptrace(rho, 1) for rho in res.states]
resumen = [validar_estado(r_, i) for i, r_ in enumerate(states_red)]
trs = np.array([r_[0] for r_ in resumen])
herms = np.array([r_[1] for r_ in resumen])
mineigs = np.array([r_[2] for r_ in resumen])
ok_all = all(r_[3] for r_ in resumen)

print("\n=== VALIDACION (estado reducido del oscilador) ===")
print(f"max|tr-1|   = {np.max(np.abs(trs-1)):.3e}")
print(f"max herm_err= {np.max(herms):.3e}")
print(f"min autoval = {np.min(mineigs):.3e}")
print("Resultado:", "OK" if ok_all else "FALLA umbral 1e-10 (ver nota en tarea1)")

# ------------------------------------------------------------
# 4) Extraer tasa de decaimiento Gamma2_minus "medida" de <n>(t)
#    dn/dt = -2 * Gamma2_measured(t) * <n>*(<n>-1)   (ansatz tipo D[a^2])
# ------------------------------------------------------------
dndt = np.gradient(nb_t, tlist)
denom = 2 * nb_t * (nb_t - 1)
# evitar division por valores pequenos/negativos cerca de n=0 o n=1
mask = np.abs(denom) > 0.5
Gamma2_measured = np.full_like(nb_t, np.nan)
Gamma2_measured[mask] = -dndt[mask] / denom[mask]

print("\n=== <n>(t) y Gamma2_minus 'instantanea' medida ===")
print(f"{'kappa*t':>8} {'<n>(t)':>10} {'dn/dt':>12} {'Gamma2_med':>12}")
for tt in [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    idx = np.argmin(np.abs(tlist - tt))
    print(f"{tlist[idx]:8.3f} {nb_t[idx]:10.4f} {dndt[idx]:12.5f} {Gamma2_measured[idx]:12.5f}")

# La dinamica bruta oscila coherentemente (no es un decaimiento markoviano
# limpio): se suaviza <n>(t) con un promedio movil (ventana ~ 1 unidad de
# kappa^-1) para extraer la tendencia SECULAR y de ahi la tasa de dos fonones.
dt = tlist[1] - tlist[0]
win = max(1, int(round(1.0 / dt)))   # ventana ~ 1 unidad de kappa*t
kernel = np.ones(win) / win
nb_smooth = np.convolve(nb_t, kernel, mode="same")
dndt_smooth = np.gradient(nb_smooth, tlist)
denom_smooth = 2 * nb_smooth * (nb_smooth - 1)
mask_s = np.abs(denom_smooth) > 0.5
Gamma2_smooth = np.full_like(nb_smooth, np.nan)
Gamma2_smooth[mask_s] = -dndt_smooth[mask_s] / denom_smooth[mask_s]

print("\n=== <n>(t) SUAVIZADO (tendencia secular) ===")
print(f"{'kappa*t':>8} {'<n>_smooth':>12} {'Gamma2_smooth':>14}")
for tt in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]:
    idx = np.argmin(np.abs(tlist - tt))
    print(f"{tlist[idx]:8.3f} {nb_smooth[idx]:12.4f} {Gamma2_smooth[idx]:14.5f}")

# excluir bordes (medio kernel) donde el promedio movil no es fiable
edge = win
interior = slice(edge, len(tlist) - edge)
plateau_mask = mask_s.copy()
plateau_mask[:edge] = False
plateau_mask[-edge:] = False
Gamma2_meas_plateau = np.nanmedian(Gamma2_smooth[plateau_mask])

print(f"\nGamma2_minus medido (mediana de la tendencia suavizada, excluyendo bordes) "
      f"= {Gamma2_meas_plateau:.6f}")
print(f"Comparacion con formula g_eff=g   : {Gamma2_g:.6f}  "
      f"(razon medido/formula = {Gamma2_meas_plateau/Gamma2_g:.3f})")
print(f"Comparacion con formula g_eff=2g  : {Gamma2_2g:.6f}  "
      f"(razon medido/formula = {Gamma2_meas_plateau/Gamma2_2g:.3f})")

# ------------------------------------------------------------
# 5) Guardar
# ------------------------------------------------------------
np.savez(
    "tarea3_dos_fonones_resultados.npz",
    tlist=tlist, nb_t=nb_t, dndt=dndt, Gamma2_measured=Gamma2_measured,
    Gamma2_meas_plateau=Gamma2_meas_plateau,
    Gamma2_g=Gamma2_g, Gamma2_2g=Gamma2_2g, g_dim=g_dim,
    trs=trs, herms=herms, mineigs=mineigs,
)
print("\nGuardado: tarea3_dos_fonones_resultados.npz")
