# -*- coding: utf-8 -*-
"""
Tarea 9, Paso B: barrido en eps/kappa (igual que Tarea 6b), tau_max=60,
muestreo ESTROBOSCOPICO (tlist en multiplos exactos de T_m=2*pi/omega_m),
comparando el modelo completo contra el modelo B (g_eff=2g).

Version liviana (Nb=16) por limitaciones de memoria de la maquina; un solo
mesolve por corrida (sin duplicar para validacion) para minimizar costo.
"""

import numpy as np
from qutip import (tensor, qeye, destroy, thermal_dm, mesolve, lindblad_dissipator,
                    sigmam, sigmaz, sigmax, Options)

r = 0.1
gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
gz_d, om_m, Gam_m = gz_org / kappa_org, omega_m_org / kappa_org, Gamma_m_org / kappa_org
kap = 1.0
n_th = 0
gx_dim = r * gz_d
g_dim = gz_d * gx_dim / om_m
g_eff_2g = 2 * g_dim

x = kap / 2.0
D2m, D2p = 0.0, 4 * om_m
ReS2m = x / (x**2 + D2m**2)
ReS2p = x / (x**2 + D2p**2)
ImS2m = -D2m / (x**2 + D2m**2)
ImS2p = -D2p / (x**2 + D2p**2)
D1m, D1p = om_m, 3 * om_m
G1m_gx = 2 * gx_dim**2 * (x / (x**2 + D1m**2))
G1p_gx = 2 * gx_dim**2 * (x / (x**2 + D1p**2))

Na, Nb_full = 2, 16
N_eff = 30
omega_q_dim = 2 * om_m
omega_d_dim = 2 * om_m
T_m = 2 * np.pi / om_m

options = Options(nsteps=2_000_000, atol=1e-10, rtol=1e-8)


def tlist_estrobo(tau_max, n_puntos=100):
    K_total = int(round(tau_max / T_m))
    stride = max(1, K_total // n_puntos)
    m = np.arange(0, K_total + 1, stride)
    return m * T_m


def correr_full(eps, tau_max):
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
    diss = [kap * lindblad_dissipator(sm),
            (n_th + 1) * Gam_m * lindblad_dissipator(b),
            n_th * Gam_m * lindblad_dissipator(bd)]
    rho0 = tensor(thermal_dm(Na, 0), thermal_dm(Nb_full, 0))
    tlist = tlist_estrobo(tau_max)
    res = mesolve(H, rho0, tlist, diss, [sz, bd * b], options=options)
    sz_t, n_t = res.expect[0], res.expect[1]
    pe_t = (1 + sz_t.real) / 2
    dndt = np.gradient(n_t, tlist)
    return tlist, n_t, pe_t, dndt


def correr_eff(eps, g_eff, tau_max):
    a = destroy(N_eff)
    adag = a.dag()
    G2m = 2 * g_eff**2 * ReS2m
    G2p = 2 * g_eff**2 * ReS2p
    dk = g_eff**2 * (ImS2m + ImS2p)
    chi = -2j * eps * g_eff / kap
    H_eff = chi.conjugate() * adag**2 + chi * a**2 + dk * (adag * a)**2
    diss = [G1m_gx * lindblad_dissipator(a), G1p_gx * lindblad_dissipator(adag),
            G2m * lindblad_dissipator(a**2), G2p * lindblad_dissipator(adag**2)]
    rho0 = thermal_dm(N_eff, 0)
    tlist = tlist_estrobo(tau_max)
    res = mesolve(H_eff, rho0, tlist, diss, [adag * a], options=options)
    return tlist, res.expect[0]


def tiempo_subida_90(t, n):
    n_ss = np.mean(n[-8:])
    if n_ss < 1e-6:
        return 0.0
    umbral = 0.9 * n_ss
    idx = np.argmax(n >= umbral)
    if n[idx] < umbral or idx == 0:
        return np.nan if n[idx] < umbral else t[0]
    t0, t1, n0, n1 = t[idx - 1], t[idx], n[idx - 1], n[idx]
    return t0 + (umbral - n0) * (t1 - t0) / (n1 - n0)


eps_list = [0.02, 0.05, 0.1, 0.2, 0.4, 0.72, 1.44]
resultados = []
for eps in eps_list:
    print(f"\n--- eps/kappa = {eps} ---")
    tlist, n_t, pe_t, dndt = correr_full(eps, tau_max=60.0)
    tlist_e, n_eff_t = correr_eff(eps, g_eff_2g, tau_max=60.0)

    n_ss_full = np.mean(n_t[-8:])
    n_ss_effB = np.mean(n_eff_t[-8:])
    t90_full = tiempo_subida_90(tlist, n_t)
    t90_effB = tiempo_subida_90(tlist_e, n_eff_t)
    dndt_final = dndt[-1]
    pe_ss = np.mean(pe_t[-8:])
    razon = n_ss_full / n_ss_effB if n_ss_effB > 1e-9 else float("nan")

    print(f"  FULL: n_ss={n_ss_full:.5f}  t90={t90_full:.3f}  dn/dt_final={dndt_final:.3e}  pe_ss={pe_ss:.5f}")
    print(f"  B(2g): n_ss={n_ss_effB:.5f}  t90={t90_effB:.3f}")
    print(f"  razon full/B = {razon:.4f}")

    resultados.append((eps, n_ss_full, t90_full, dndt_final, pe_ss, n_ss_effB, t90_effB, razon))

print("\n=== TABLA RESUMEN (Tarea 9, Paso B, estroboscopico) ===")
hdr = (f"{'eps/k':>7} {'n_full':>9} {'t90_full':>9} {'dndt_f':>10} {'pe_ss':>8} | "
       f"{'n_B':>9} {'t90_B':>7} | {'full/B':>8}")
print(hdr)
for row in resultados:
    eps, nf, t90f, dndtf, pef, nB, t90B, razon = row
    print(f"{eps:>7.2f} {nf:>9.5f} {t90f:>9.3f} {dndtf:>10.2e} {pef:>8.5f} | "
          f"{nB:>9.5f} {t90B:>7.3f} | {razon:>8.4f}")

np.savez("tarea9_pasoB.npz", eps_list=np.array(eps_list), resultados=np.array(resultados))
print("\nGuardado: tarea9_pasoB.npz")
