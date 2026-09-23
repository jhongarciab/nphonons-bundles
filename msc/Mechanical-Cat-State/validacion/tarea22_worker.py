# -*- coding: utf-8 -*-
"""
Worker de la Tarea 22: calcula UNA sola celda (gz_scale, alpha2, modelo,
compensar) en un proceso fresco (para evitar acumulacion de memoria/
degradacion observada al correr todo en un unico proceso largo), y
guarda el resultado en un .npz individual.

Uso: python tarea22_worker.py <gz_scale> <alpha2> <modelo:full|eff> <compensar:0|1> <outfile>
"""

import sys
import numpy as np
from qutip import (tensor, qeye, destroy, sigmam, sigmaz, sigmax, Options, propagator,
                    vector_to_operator, Qobj, liouvillian)

gz_scale = float(sys.argv[1])
alpha2 = float(sys.argv[2])
modelo = sys.argv[3]
compensar = bool(int(sys.argv[4]))
outfile = sys.argv[5]

r = 0.1
gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
gz_baseline = gz_org / kappa_org
om_m = omega_m_org / kappa_org
Gam_m = Gamma_m_org / kappa_org
kap = 1.0
n_th = 0
gx_fijo = r * gz_baseline
omega_q_dim = 2 * om_m
T_m = 2 * np.pi / om_m
Na = 2

x = kap / 2.0
D2m0, D2p0 = 0.0, 4 * om_m
ReS2m0 = x / (x**2 + D2m0**2)
ReS2p0 = x / (x**2 + D2p0**2)
ImS2m0 = -D2m0 / (x**2 + D2m0**2)
ImS2p0 = -D2p0 / (x**2 + D2p0**2)
D1m, D1p = om_m, 3 * om_m
ImS1m = -D1m / (x**2 + D1m**2)
ImS1p = -D1p / (x**2 + D1p**2)
G1m_gx = 2 * gx_fijo**2 * (x / (x**2 + D1m**2))
G1p_gx = 2 * gx_fijo**2 * (x / (x**2 + D1p**2))
delta_1 = gx_fijo**2 * (ImS1m + ImS1p)

opts = Options(atol=1e-12, rtol=1e-10, nsteps=2_000_000)

Nb = max(20, int(4 * alpha2 + 12))

gz = gz_scale * gz_baseline
gx = gx_fijo
g_dim = gz * gx / om_m
g_eff = 2 * g_dim
eps = alpha2 * g_eff
Gamma2 = 4 * g_eff**2 / kap


def clasificar_y_extraer(evals, Xks, Nb, es_completo):
    P_osc = Qobj(np.diag((-1.0) ** np.arange(Nb)))
    n_op = destroy(Nb).dag() * destroy(Nb)
    a_op = destroy(Nb)
    if es_completo:
        P_ref, n_ref, a_ref = tensor(qeye(Na), P_osc), tensor(qeye(Na), n_op), tensor(qeye(Na), a_op)
    else:
        P_ref, n_ref, a_ref = P_osc, n_op, a_op

    modos = []
    for k, (mu, Xk) in enumerate(zip(evals, Xks)):
        if k == 0:
            continue
        if es_completo:
            lam = -np.log(mu) / T_m
        else:
            lam = -mu
        normXk = Xk.norm()
        ov_P = abs((P_ref.dag() * Xk).tr()) / normXk
        ov_n = abs((n_ref.dag() * Xk).tr()) / normXk
        ov_a = abs((a_ref.dag() * Xk).tr()) / normXk
        modos.append(dict(k=k, lam=lam, ov_P=ov_P, ov_n=ov_n, ov_a=ov_a))

    cand_P = [m for m in modos if m['ov_P'] > m['ov_n'] and m['ov_P'] > m['ov_a']]
    gamma_pf = min((m['lam'].real for m in cand_P), default=float('nan'))
    k_pf = min(cand_P, key=lambda m: m['lam'].real)['k'] if cand_P else None

    cand_a = [m for m in modos if m['ov_a'] > m['ov_P'] and m['ov_a'] > m['ov_n']]
    if cand_a:
        modo_bf = min(cand_a, key=lambda m: m['lam'].real)
        gamma_bf = modo_bf['lam'].real
        freq_bf = modo_bf['lam'].imag
        k_bf = modo_bf['k']
    else:
        gamma_bf, freq_bf, k_bf = float('nan'), float('nan'), None

    cand_n = [m for m in modos if m['ov_n'] > m['ov_P'] and m['ov_n'] > m['ov_a']
              and m['k'] not in (k_pf, k_bf)]
    gamma_conf = min((m['lam'].real for m in cand_n), default=float('nan'))

    return gamma_pf, gamma_bf, freq_bf, gamma_conf


if modelo == "full":
    if compensar:
        oq = 2 * (om_m + delta_1)
    else:
        oq = omega_q_dim
    od = oq

    b = tensor(qeye(Na), destroy(Nb))
    bd = b.dag()
    sm = tensor(sigmam(), qeye(Nb))
    sz = tensor(sigmaz(), qeye(Nb))
    sx = tensor(sigmax(), qeye(Nb))
    H0 = 0.5 * oq * sz
    H = [H0,
         [gx * sx * b, lambda t, _: np.exp(-1j * om_m * t)],
         [gx * sx * bd, lambda t, _: np.exp(+1j * om_m * t)],
         [gz * sz * b, lambda t, _: np.exp(-1j * om_m * t)],
         [gz * sz * bd, lambda t, _: np.exp(+1j * om_m * t)],
         [eps * sx, lambda t, _: np.exp(+1j * od * t)],
         [eps * sx, lambda t, _: np.exp(-1j * od * t)]]
    c_ops = [np.sqrt(kap) * sm, np.sqrt((n_th + 1) * Gam_m) * b, np.sqrt(n_th * Gam_m) * bd]

    U = propagator(H, T_m, c_ops, options=opts)
    evals, evecs = U.eigenstates()
    orden = np.argsort(-np.abs(evals))
    n_modos = min(12, len(evals))
    evals_top = evals[orden][:n_modos]
    Xks = [vector_to_operator(evecs[orden[i]]) for i in range(n_modos)]
    gamma_pf, gamma_bf, freq_bf, gamma_conf = clasificar_y_extraer(evals_top, Xks, Nb, True)

else:  # efectivo
    a = destroy(Nb)
    adag = a.dag()
    G2m = 2 * g_eff**2 * ReS2m0
    G2p = 2 * g_eff**2 * ReS2p0
    dk = g_eff**2 * (ImS2m0 + ImS2p0)
    chi = -2j * eps * g_eff / kap
    H_eff = chi.conjugate() * adag**2 + chi * a**2 + dk * (adag * a)**2
    if not compensar:
        H_eff = H_eff + delta_1 * (adag * a)
    c_ops = [np.sqrt(G1m_gx) * a, np.sqrt(G1p_gx) * adag,
             np.sqrt(G2m) * (a * a), np.sqrt(G2p) * (adag * adag),
             np.sqrt((n_th + 1) * Gam_m) * a, np.sqrt(n_th * Gam_m) * adag]

    L = liouvillian(H_eff, c_ops)
    evals, evecs = L.eigenstates(sparse=False)
    orden = np.argsort(-evals.real)
    n_modos = min(12, len(evals))
    evals_top = evals[orden][:n_modos]
    Xks = [vector_to_operator(evecs[orden[i]]) for i in range(n_modos)]
    gamma_pf, gamma_bf, freq_bf, gamma_conf = clasificar_y_extraer(evals_top, Xks, Nb, False)

np.savez(outfile, gz_scale=gz_scale, alpha2=alpha2, modelo=modelo, compensar=int(compensar),
         Nb=Nb, Gamma2=Gamma2, gamma_pf=gamma_pf, gamma_bf=gamma_bf, freq_bf=freq_bf,
         gamma_conf=gamma_conf)
print(f"OK gz={gz_scale} a2={alpha2} modelo={modelo} comp={compensar} Nb={Nb} "
      f"Gamma2={Gamma2:.5f} pf={gamma_pf:.4e} bf={gamma_bf:.4e} conf={gamma_conf:.4e}")
