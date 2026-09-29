# -*- coding: utf-8 -*-
"""
Worker de la Tarea 23: para un Gamma2/kappa objetivo dado (con
|alpha|^2=2 fijo), calcula la brecha de confinamiento del modelo
COMPLETO (Floquet) y del EFECTIVO (Liouvilliano, g_eff=2g+delta_1,
con amortiguamiento intrinseco siempre incluido).

Uso: python tarea23_worker.py <Gamma2_target> <outfile>
"""

import sys
import numpy as np
from qutip import (tensor, qeye, destroy, sigmam, sigmaz, sigmax, Options, propagator,
                    vector_to_operator, Qobj, liouvillian)

Gamma2_target = float(sys.argv[1])
outfile = sys.argv[2]

r = 0.1
gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
om_m = omega_m_org / kappa_org
Gam_m = Gamma_m_org / kappa_org
kap = 1.0
n_th = 0
gx_fijo = r * (gz_org / kappa_org)
omega_q_dim = 2 * om_m
T_m = 2 * np.pi / om_m
Na = 2
Nb = 20
alpha2 = 2.0

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

g_eff = np.sqrt(Gamma2_target / 4.0)
gz = g_eff * om_m / (2 * gx_fijo)
eps = alpha2 * g_eff

opts = Options(atol=1e-12, rtol=1e-10, nsteps=2_000_000)


def confinamiento(evals, Xks, Nb, es_completo):
    P_osc = Qobj(np.diag((-1.0) ** np.arange(Nb)))
    n_op = destroy(Nb).dag() * destroy(Nb)
    a_op = destroy(Nb)
    if es_completo:
        P_ref, n_ref, a_ref = tensor(qeye(Na), P_osc), tensor(qeye(Na), n_op), tensor(qeye(Na), a_op)
    else:
        P_ref, n_ref, a_ref = P_osc, n_op, a_op

    cand = []
    for k, (mu, Xk) in enumerate(zip(evals, Xks)):
        if k == 0:
            continue
        lam = (-np.log(mu) / T_m) if es_completo else (-mu)
        normXk = Xk.norm()
        ov_P = abs((P_ref.dag() * Xk).tr()) / normXk
        ov_n = abs((n_ref.dag() * Xk).tr()) / normXk
        ov_a = abs((a_ref.dag() * Xk).tr()) / normXk
        if ov_n > ov_P and ov_n > ov_a:
            cand.append(lam.real)
    return min(cand) if cand else float('nan')


# --- COMPLETO ---
b = tensor(qeye(Na), destroy(Nb))
bd = b.dag()
sm = tensor(sigmam(), qeye(Nb))
sz = tensor(sigmaz(), qeye(Nb))
sx = tensor(sigmax(), qeye(Nb))
H0 = 0.5 * omega_q_dim * sz
H = [H0,
     [gx_fijo * sx * b, lambda t, _: np.exp(-1j * om_m * t)],
     [gx_fijo * sx * bd, lambda t, _: np.exp(+1j * om_m * t)],
     [gz * sz * b, lambda t, _: np.exp(-1j * om_m * t)],
     [gz * sz * bd, lambda t, _: np.exp(+1j * om_m * t)],
     [eps * sx, lambda t, _: np.exp(+1j * omega_q_dim * t)],
     [eps * sx, lambda t, _: np.exp(-1j * omega_q_dim * t)]]
c_ops = [np.sqrt(kap) * sm, np.sqrt((n_th + 1) * Gam_m) * b, np.sqrt(n_th * Gam_m) * bd]

U = propagator(H, T_m, c_ops, options=opts)
evals, evecs = U.eigenstates()
orden = np.argsort(-np.abs(evals))
n_modos = min(10, len(evals))
evals_top = evals[orden][:n_modos]
Xks = [vector_to_operator(evecs[orden[i]]) for i in range(n_modos)]
conf_full = confinamiento(evals_top, Xks, Nb, True)

# --- EFECTIVO ---
a = destroy(Nb)
adag = a.dag()
G2m = 2 * g_eff**2 * ReS2m0
G2p = 2 * g_eff**2 * ReS2p0
dk = g_eff**2 * (ImS2m0 + ImS2p0)
chi = -2j * eps * g_eff / kap
H_eff = chi.conjugate() * adag**2 + chi * a**2 + dk * (adag * a)**2 + delta_1 * (adag * a)
c_ops_eff = [np.sqrt(G1m_gx) * a, np.sqrt(G1p_gx) * adag,
             np.sqrt(G2m) * (a * a), np.sqrt(G2p) * (adag * adag),
             np.sqrt((n_th + 1) * Gam_m) * a, np.sqrt(n_th * Gam_m) * adag]
L = liouvillian(H_eff, c_ops_eff)
evals_e, evecs_e = L.eigenstates(sparse=False)
orden_e = np.argsort(-evals_e.real)
n_modos_e = min(10, len(evals_e))
evals_e_top = evals_e[orden_e][:n_modos_e]
Xks_e = [vector_to_operator(evecs_e[orden_e[i]]) for i in range(n_modos_e)]
conf_eff = confinamiento(evals_e_top, Xks_e, Nb, False)

np.savez(outfile, Gamma2_target=Gamma2_target, gz=gz, g_eff=g_eff, eps=eps,
         conf_full=conf_full, conf_eff=conf_eff)
print(f"OK Gamma2={Gamma2_target:.5f} gz={gz:.3f} conf_full={conf_full:.5e} conf_eff={conf_eff:.5e}")
