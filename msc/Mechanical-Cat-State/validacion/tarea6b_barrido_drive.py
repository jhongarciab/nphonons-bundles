# -*- coding: utf-8 -*-
"""
Tarea 6b: barrido en eps/kappa (drive real del qubit) comparando el modelo
COMPLETO contra el modelo EFECTIVO con g_eff=g y g_eff=2g.

Para cada eps se reporta: p_e_ss (completo), n_ss (completo), t90 (completo)
vs n_ss y t90 de los dos modelos efectivos.

Alcance reducido respecto a una corrida "ideal" para que el barrido sea
factible en tiempo razonable, MANTENIENDO atol=1e-10, rtol=1e-8 (requisito
no negociable de esta ronda de validacion):
  - Nb = 20 (en vez de 50): suficiente margen para n_ss esperado (<10) en
    todo el rango de eps escaneado.
  - tau_max = 30, n_steps = 80 (en vez de 60/150): ~15x el tiempo de
    relajacion esperado (1/Gamma2_minus ~ 1.93 kappa^-1), que no depende
    de eps (Gamma2_minus solo depende de g).

Se reporta explicitamente dn/dt final de cada corrida para que se pueda
juzgar si el "estado estacionario" esta realmente alcanzado.

NO se modifica ningun script original.
"""

import numpy as np
from qutip import (tensor, qeye, destroy, thermal_dm, mesolve, lindblad_dissipator,
                    sigmam, sigmaz, sigmax, Options)

# ------------------------------------------------------------
# Parametros fisicos comunes
# ------------------------------------------------------------
r = 0.1
gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
gz_d, om_m, Gam_m = gz_org / kappa_org, omega_m_org / kappa_org, Gamma_m_org / kappa_org
kap = 1.0
n_th = 0
gx_dim = r * gz_d
g_dim = gz_d * gx_dim / om_m

x = kap / 2.0
D1m, D1p = 2 * om_m - om_m, 2 * om_m + om_m
D2m, D2p = 0.0, 4 * om_m   # omega_q = 2 om_m => D2m=0 exacto
ReS2m = x / (x**2 + D2m**2)
ReS2p = x / (x**2 + D2p**2)
ImS2m = -D2m / (x**2 + D2m**2)
ImS2p = -D2p / (x**2 + D2p**2)

G1m_gx = 2 * gx_dim**2 * (x / (x**2 + D1m**2))
G1p_gx = 2 * gx_dim**2 * (x / (x**2 + D1p**2))

Na, Nb_full = 2, 20
N_eff = 40

omega_q_dim = 2 * om_m
omega_d_dim = 2 * om_m

options = Options(nsteps=2_000_000, atol=1e-10, rtol=1e-8)


def correr_full(eps):
    b = tensor(qeye(Na), destroy(Nb_full))
    bd = b.dag()
    sm = tensor(sigmam(), qeye(Nb_full))
    sz = tensor(sigmaz(), qeye(Nb_full))
    sx = tensor(sigmax(), qeye(Nb_full))

    H0 = 0.5 * omega_q_dim * sz
    H = [
        H0,
        [gx_dim * sx * b, lambda t, _: np.exp(-1j * om_m * t)],
        [gx_dim * sx * bd, lambda t, _: np.exp(+1j * om_m * t)],
        [gz_d * sz * b, lambda t, _: np.exp(-1j * om_m * t)],
        [gz_d * sz * bd, lambda t, _: np.exp(+1j * om_m * t)],
        [eps * sx, lambda t, _: np.exp(+1j * omega_d_dim * t)],
        [eps * sx, lambda t, _: np.exp(-1j * omega_d_dim * t)],
    ]
    diss = [kap * lindblad_dissipator(sm),
            (n_th + 1) * Gam_m * lindblad_dissipator(b),
            n_th * Gam_m * lindblad_dissipator(bd)]
    rho0 = tensor(thermal_dm(Na, 0), thermal_dm(Nb_full, 0))

    tau_max, n_steps = 30.0, 80
    tlist = np.linspace(0, tau_max, n_steps)
    res = mesolve(H, rho0, tlist, diss, [sz, bd * b], options=options)
    sz_t, n_t = res.expect[0], res.expect[1]
    pe_t = (1 + sz_t.real) / 2

    dndt = np.gradient(n_t, tlist)
    n_ss = np.mean(n_t[-8:])
    pe_ss = np.mean(pe_t[-8:])
    t90, _ = tiempo_subida_90(tlist, n_t)
    return dict(n_ss=n_ss, pe_ss=pe_ss, t90=t90, dndt_final=dndt[-1], n_t=n_t, tlist=tlist)


def correr_eff(eps, g_eff):
    a = destroy(N_eff)
    adag = a.dag()
    G2m = 2 * g_eff**2 * ReS2m
    G2p = 2 * g_eff**2 * ReS2p
    dk = g_eff**2 * (ImS2m + ImS2p)
    chi = -2j * eps * g_eff / kap

    H_eff = chi.conjugate() * adag**2 + chi * a**2 + dk * (adag * a)**2
    diss = [G1m_gx * lindblad_dissipator(a),
            G1p_gx * lindblad_dissipator(adag),
            G2m * lindblad_dissipator(a**2),
            G2p * lindblad_dissipator(adag**2)]
    rho0 = thermal_dm(N_eff, 0)

    tau_max, n_steps = 30.0, 80
    tlist = np.linspace(0, tau_max, n_steps)
    res = mesolve(H_eff, rho0, tlist, diss, [adag * a], options=options)
    n_t = res.expect[0]
    n_ss = np.mean(n_t[-8:])
    t90, _ = tiempo_subida_90(tlist, n_t)
    return dict(n_ss=n_ss, t90=t90)


def tiempo_subida_90(t, n):
    n_ss = np.mean(n[-8:])
    if n_ss < 1e-6:
        return 0.0, n_ss
    umbral = 0.9 * n_ss
    idx = np.argmax(n >= umbral)
    if n[idx] < umbral:
        return np.nan, n_ss
    if idx == 0:
        return t[0], n_ss
    t0, t1, n0, n1 = t[idx - 1], t[idx], n[idx - 1], n[idx]
    return t0 + (umbral - n0) * (t1 - t0) / (n1 - n0), n_ss


eps_list = [0.02, 0.05, 0.1, 0.2, 0.4, 0.72, 1.44]

resultados = []
for eps in eps_list:
    print(f"\n=== eps/kappa = {eps} ===")
    full = correr_full(eps)
    effA = correr_eff(eps, g_dim)         # g_eff = g
    effB = correr_eff(eps, 2 * g_dim)     # g_eff = 2g
    print(f"  FULL : n_ss={full['n_ss']:.4f}  p_e_ss={full['pe_ss']:.5f}  "
          f"t90={full['t90']:.3f}  dn/dt_final={full['dndt_final']:.2e}")
    print(f"  A(g) : n_ss={effA['n_ss']:.4f}  t90={effA['t90']:.3f}")
    print(f"  B(2g): n_ss={effB['n_ss']:.4f}  t90={effB['t90']:.3f}")
    resultados.append((eps, full["n_ss"], full["pe_ss"], full["t90"], full["dndt_final"],
                        effA["n_ss"], effA["t90"], effB["n_ss"], effB["t90"]))

print("\n=== TABLA RESUMEN (Tarea 6b) ===")
hdr = (f"{'eps/k':>7} {'pe_ss':>8} {'n_full':>8} {'t90_full':>9} {'dndt_f':>9} | "
       f"{'n_A(g)':>8} {'t90_A':>7} | {'n_B(2g)':>8} {'t90_B':>7}")
print(hdr)
for row in resultados:
    eps, nf, pef, t90f, dndtf, nA, t90A, nB, t90B = row
    print(f"{eps:>7.2f} {pef:>8.5f} {nf:>8.4f} {t90f:>9.3f} {dndtf:>9.2e} | "
          f"{nA:>8.4f} {t90A:>7.3f} | {nB:>8.4f} {t90B:>7.3f}")

np.savez("tarea6b_resultados.npz", eps_list=np.array(eps_list), resultados=np.array(resultados),
         g_dim=g_dim)
print("\nGuardado: tarea6b_resultados.npz")
