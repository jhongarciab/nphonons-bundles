# -*- coding: utf-8 -*-
"""
Worker de la Tarea 25 (a)+(b): en un punto de resonancia vestida dado
(delta_m, Delta_q) para un (gz_scale, alpha2), calcula:
  - modelo COMPLETO (marco conmensurable w_r) en ese punto: gamma_bf,
    gamma_pf, brecha de confinamiento, y para el modo de confinamiento
    sus overlaps (P, n, a) e Im(lambda).
  - modelo EFECTIVO RESONANTE (g_eff=2g, SIN delta_1*a^dagger a,
    Delta_2-=0 -- misma variante "compensar" del efectivo de la Tarea 22,
    que si es valida): las mismas cantidades.

Uso: python tarea25_ab_worker.py <gz_scale> <alpha2> <delta_m> <Delta_q> <outfile>
"""

import sys
import numpy as np
from qutip import (tensor, qeye, destroy, sigmam, sigmap, sigmaz, Options, propagator,
                    vector_to_operator, Qobj, liouvillian, lindblad_dissipator)

gz_scale = float(sys.argv[1])
alpha2 = float(sys.argv[2])
delta_m = float(sys.argv[3])
Delta_q = float(sys.argv[4])
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
Na = 2

gz = gz_scale * gz_baseline
gx = gx_fijo
g_dim = gz * gx / om_m
g_eff = 2 * g_dim
eps = alpha2 * g_eff
Gamma2 = 4 * g_eff**2 / kap

Nb = max(20, int(4 * alpha2 + 12))
wr = om_m
T_r = 2 * np.pi / wr

x = kap / 2.0
D2m0, D2p0 = 0.0, 4 * om_m   # efectivo resonante: Delta_2-=0
ReS2m0 = x / (x**2 + D2m0**2)
ReS2p0 = x / (x**2 + D2p0**2)
ImS2m0 = -D2m0 / (x**2 + D2m0**2)
ImS2p0 = -D2p0 / (x**2 + D2p0**2)
D1m, D1p = om_m, 3 * om_m
G1m_gx = 2 * gx**2 * (x / (x**2 + D1m**2))
G1p_gx = 2 * gx**2 * (x / (x**2 + D1p**2))

opts = Options(atol=1e-12, rtol=1e-10, nsteps=2_000_000)


def clasificar(evals, Xks, Nb, es_completo, T):
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
        lam = (-np.log(mu) / T) if es_completo else (-mu)
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
        gamma_bf, k_bf = modo_bf['lam'].real, modo_bf['k']
    else:
        gamma_bf, k_bf = float('nan'), None

    cand_n = [m for m in modos if m['ov_n'] > m['ov_P'] and m['ov_n'] > m['ov_a']
              and m['k'] not in (k_pf, k_bf)]
    if cand_n:
        modo_conf = min(cand_n, key=lambda m: m['lam'].real)
        gamma_conf = modo_conf['lam'].real
        im_conf = modo_conf['lam'].imag
        ov_conf = (modo_conf['ov_P'], modo_conf['ov_n'], modo_conf['ov_a'])
    else:
        gamma_conf, im_conf, ov_conf = float('nan'), float('nan'), (np.nan, np.nan, np.nan)

    return gamma_pf, gamma_bf, gamma_conf, im_conf, ov_conf


# --- COMPLETO, marco conmensurable, en (delta_m, Delta_q) ---
b = tensor(qeye(Na), destroy(Nb))
bd = b.dag()
sm = tensor(sigmam(), qeye(Nb))
sp = tensor(sigmap(), qeye(Nb))
sz = tensor(sigmaz(), qeye(Nb))

H_static = delta_m * bd * b + (Delta_q / 2) * sz + eps * (sp + sm)
H = [
    H_static,
    [gx * sp * b, lambda t, _: np.exp(1j * wr * t)],
    [gx * sp * bd, lambda t, _: np.exp(3j * wr * t)],
    [gx * sm * b, lambda t, _: np.exp(-3j * wr * t)],
    [gx * sm * bd, lambda t, _: np.exp(-1j * wr * t)],
    [gz * sz * b, lambda t, _: np.exp(-1j * wr * t)],
    [gz * sz * bd, lambda t, _: np.exp(1j * wr * t)],
    [eps * sp, lambda t, _: np.exp(4j * wr * t)],
    [eps * sm, lambda t, _: np.exp(-4j * wr * t)],
]
c_ops = [np.sqrt(kap) * sm, np.sqrt((n_th + 1) * Gam_m) * b, np.sqrt(n_th * Gam_m) * bd]

U = propagator(H, T_r, c_ops, options=opts)
evals, evecs = U.eigenstates()
orden = np.argsort(-np.abs(evals))
n_modos = min(12, len(evals))
evals_top = evals[orden][:n_modos]
Xks = [vector_to_operator(evecs[orden[i]]) for i in range(n_modos)]
pf_full, bf_full, conf_full, im_conf_full, ov_conf_full = clasificar(evals_top, Xks, Nb, True, T_r)

# --- EFECTIVO RESONANTE (sin delta_1, Delta_2-=0) ---
a = destroy(Nb)
adag = a.dag()
G2m = 2 * g_eff**2 * ReS2m0
G2p = 2 * g_eff**2 * ReS2p0
dk = g_eff**2 * (ImS2m0 + ImS2p0)
chi = -2j * eps * g_eff / kap
H_eff = chi.conjugate() * adag**2 + chi * a**2 + dk * (adag * a)**2   # SIN delta_1
c_ops_eff = [np.sqrt(G1m_gx) * a, np.sqrt(G1p_gx) * adag,
             np.sqrt(G2m) * (a * a), np.sqrt(G2p) * (adag * adag),
             np.sqrt((n_th + 1) * Gam_m) * a, np.sqrt(n_th * Gam_m) * adag]

L = liouvillian(H_eff, c_ops_eff)
evals_e, evecs_e = L.eigenstates(sparse=False)
orden_e = np.argsort(-evals_e.real)
n_modos_e = min(12, len(evals_e))
evals_e_top = evals_e[orden_e][:n_modos_e]
Xks_e = [vector_to_operator(evecs_e[orden_e[i]]) for i in range(n_modos_e)]
pf_eff, bf_eff, conf_eff, im_conf_eff, ov_conf_eff = clasificar(evals_e_top, Xks_e, Nb, False, None)

np.savez(outfile, gz_scale=gz_scale, alpha2=alpha2, delta_m=delta_m, Delta_q=Delta_q,
         Gamma2=Gamma2, Nb=Nb,
         pf_full=pf_full, bf_full=bf_full, conf_full=conf_full, im_conf_full=im_conf_full,
         ov_conf_full=ov_conf_full,
         pf_eff=pf_eff, bf_eff=bf_eff, conf_eff=conf_eff, im_conf_eff=im_conf_eff,
         ov_conf_eff=ov_conf_eff)
print(f"OK gz={gz_scale} a2={alpha2} dm={delta_m:.5f} Dq={Delta_q:.5f} Gamma2={Gamma2:.5f} | "
      f"FULL pf={pf_full:.4e} bf={bf_full:.4e} conf={conf_full:.4e} im_conf={im_conf_full:.4e} | "
      f"EFF pf={pf_eff:.4e} bf={bf_eff:.4e} conf={conf_eff:.4e} im_conf={im_conf_eff:.4e}")
