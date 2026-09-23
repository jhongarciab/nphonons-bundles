# -*- coding: utf-8 -*-
"""
Tarea 2 (hipotesis del factor 2).

La Ec. (9) del paper da   -2i g sigma_y (a^2 - a^dagger^2)  =  -2g(sigma_+ - sigma_-)(a^dagger^2 - a^2)
pero la Ec. (11), usada para derivar el modelo efectivo, emplea +g(sigma_+ - sigma_-)(a^dagger^2 - a^2).
Es decir, hay un posible factor 2 perdido en el acoplamiento de dos fonones g_eff.

Comparamos el modelo COMPLETO (Omega = 4g, drive real tal como esta codificado en
fig2_git_v1.py) contra tres modelos EFECTIVOS:

  (A) el del paper/codigo original:  g_eff = g        , eps = 2g
  (B) coupling corregido + drive consistente con el completo:
                                      g_eff = 2g       , eps = Omega_full = 4g
  (C) drive consistente pero SIN corregir g:
                                      g_eff = g        , eps = 4g

Para cada modelo se reporta:
    - <n>(t)
    - tiempo de subida t_90 (90% del valor estacionario, tomado como promedio
      de los ultimos 5 puntos de <n>(t))
    - fidelidad de Uhlmann vs el modelo completo, en todo t y en kappa*t = 39

NO se modifica fig2_git_v1.py; este script es una copia/derivacion en ./validacion/.
"""

import numpy as np
from qutip import (destroy, thermal_dm, mesolve, tensor, qeye,
                    sigmam, sigmaz, sigmax, Options, ptrace, lindblad_dissipator)
from qutip.metrics import fidelity as state_fidelity

# ------------------------------------------------------------
# 0) Ajustes de simulacion (identicos a fig2_git_v1.py / tarea1)
# ------------------------------------------------------------
tau_max = 39.0
n_steps = 120
tau = np.linspace(0, tau_max, n_steps)
options = Options(nsteps=1_000_000, store_states=True)

N_eff = 50
gz = 2 * np.pi * 6e6
r = 0.1
gx = r * gz
omega_m = 2 * np.pi * 100e6
omega_q = 2 * omega_m
kappa = 2 * np.pi * 100e3
gamma_m = 2 * np.pi * 15
n_th = 0

g_paper = gz * gx / omega_m     # g tal como lo define el codigo original

x = kappa / 2
D1m, D1p = omega_q - omega_m, omega_q + omega_m
D2m, D2p = omega_q - 2 * omega_m, omega_q + 2 * omega_m

# Tasas de un foton: dependen de g_x (transversal), no del g de dos fotones -> fijas
G1m = 2 * gx**2 * x / (x**2 + D1m**2)
G1p = 2 * gx**2 * x / (x**2 + D1p**2)

# ------------------------------------------------------------
# 1) MODELO COMPLETO (identico a fig2_git_v1.py, Omega = 4g)
# ------------------------------------------------------------
Na, Nb = 2, 50
b = tensor(qeye(Na), destroy(Nb))
bd = b.dag()

gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
gz_d, om_m, Gam_m = gz_org / kappa_org, omega_m_org / kappa_org, Gamma_m_org / kappa_org
kap = 1.0

gx_dim = r * gz_d
g_dim = gz_d * gx_dim / om_m      # = g_paper en unidades de kappa
Omega_full = 4 * g_dim            # drive REAL usado en el modelo completo

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

diss_full = [kap * lindblad_dissipator(sm),
             (n_th + 1) * Gam_m * lindblad_dissipator(b),
             n_th * Gam_m * lindblad_dissipator(bd)]

rho0_full = tensor(thermal_dm(Na, 0), thermal_dm(Nb, 0))
print("Corriendo modelo COMPLETO (Omega = 4g)...")
res_full = mesolve(H, rho0_full, tau, diss_full, [bd * b], options=options)
nb_full = res_full.expect[0]
states_full_red = [ptrace(rho, 1) for rho in res_full.states]
print("... completo listo.")

# ------------------------------------------------------------
# 2) Funcion generica para construir y correr un modelo efectivo
# ------------------------------------------------------------
def correr_modelo_efectivo(g_eff, eps, nombre):
    a = destroy(N_eff)
    adag = a.dag()

    chi = -2j * eps * g_eff / kappa
    G2m = 2 * g_eff**2 * x / (x**2 + D2m**2)
    G2p = 2 * g_eff**2 * x / (x**2 + D2p**2)
    dk = g_eff**2 * (-D2m / (x**2 + D2m**2) - D2p / (x**2 + D2p**2))

    H_eff = chi.conjugate() * adag**2 + chi * a**2 + dk * (adag * a)**2
    diss_eff = [G1m * lindblad_dissipator(a),
                G1p * lindblad_dissipator(adag),
                G2m * lindblad_dissipator(a**2),
                G2p * lindblad_dissipator(adag**2)]

    rho0_eff = thermal_dm(N_eff, 0)
    tlist_eff = tau / kappa   # kappa=1 en estas unidades, pero se deja explicito
    print(f"Corriendo modelo EFECTIVO {nombre} (g_eff={g_eff:.4g}, eps={eps:.4g})...")
    res = mesolve(H_eff, rho0_eff, tlist_eff, diss_eff, [adag * a], options=options)
    print(f"... {nombre} listo.")
    return res.expect[0], res.states


def tiempo_subida_90(t, n):
    n_ss = np.mean(n[-5:])
    umbral = 0.9 * n_ss
    idx = np.argmax(n >= umbral)
    if n[idx] < umbral:
        return np.nan, n_ss   # nunca llega al 90%
    if idx == 0:
        return t[0], n_ss
    # interpolacion lineal entre idx-1 e idx
    t0, t1 = t[idx - 1], t[idx]
    n0, n1 = n[idx - 1], n[idx]
    t90 = t0 + (umbral - n0) * (t1 - t0) / (n1 - n0)
    return t90, n_ss


# ------------------------------------------------------------
# 3) Tres modelos efectivos
# ------------------------------------------------------------
modelos = {
    "A_paper":        dict(g_eff=g_paper,     eps=2 * g_paper),
    "B_corregido":    dict(g_eff=2 * g_paper, eps=4 * g_paper),
    "C_drive_4g":     dict(g_eff=g_paper,     eps=4 * g_paper),
}

resultados = {}
for nombre, params in modelos.items():
    nb, states = correr_modelo_efectivo(params["g_eff"], params["eps"], nombre)
    fidelity_t = np.array([state_fidelity(states_full_red[i], states[i]) for i in range(len(tau))])
    t90, n_ss = tiempo_subida_90(tau, nb)
    resultados[nombre] = dict(
        nb=nb, fidelity_t=fidelity_t, t90=t90, n_ss=n_ss,
        fidelity_final=fidelity_t[-1],
        g_eff=params["g_eff"], eps=params["eps"],
    )

# ------------------------------------------------------------
# 4) Reporte
# ------------------------------------------------------------
print("\n=== TABLA RESUMEN (Tarea 2) ===")
print(f"{'Modelo':<12} {'g_eff/g':>8} {'eps/g':>8} {'n_ss':>8} {'t90':>8} "
      f"{'F_min':>8} {'F(kt=39)':>10}")
for nombre, r_ in resultados.items():
    f_min = r_["fidelity_t"].min()
    print(f"{nombre:<12} {r_['g_eff']/g_paper:>8.3f} {r_['eps']/g_paper:>8.3f} "
          f"{r_['n_ss']:>8.4f} {r_['t90']:>8.3f} {f_min:>8.4f} {r_['fidelity_final']:>10.6f}")

print(f"\n<n>_full(kt=39) = {nb_full[-1]:.4f}")
n_ss_full = np.mean(nb_full[-5:])
t90_full, _ = tiempo_subida_90(tau, nb_full)
print(f"n_ss_full = {n_ss_full:.4f}   t90_full = {t90_full:.3f}")

# ------------------------------------------------------------
# 5) Guardar
# ------------------------------------------------------------
np.savez(
    "tarea2_factor2_resultados.npz",
    tau=tau,
    nb_full=nb_full,
    n_ss_full=n_ss_full, t90_full=t90_full,
    **{f"{k}_nb": v["nb"] for k, v in resultados.items()},
    **{f"{k}_fidelity_t": v["fidelity_t"] for k, v in resultados.items()},
    **{f"{k}_t90": v["t90"] for k, v in resultados.items()},
    **{f"{k}_n_ss": v["n_ss"] for k, v in resultados.items()},
)
print("\nGuardado: tarea2_factor2_resultados.npz")
