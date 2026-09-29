# -*- coding: utf-8 -*-
"""
Worker de la Tarea 24: calcula UNA celda (gz_scale, alpha2, delta_m,
Delta_q) del modelo COMPLETO en el marco conmensurable de frecuencia
unica w_r=omega_m, y reporta gamma_bf, Im(lambda_bf), gamma_pf y la
brecha de confinamiento.

Uso: python tarea24_worker.py <gz_scale> <alpha2> <delta_m> <Delta_q> <outfile>
"""

import sys
import numpy as np
from qutip import (tensor, qeye, destroy, sigmam, sigmap, sigmaz, Options, propagator,
                    vector_to_operator, Qobj)

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

opts = Options(atol=1e-12, rtol=1e-10, nsteps=2_000_000)


def clasificar_y_extraer(evals, Xks, Nb):
    P_osc = Qobj(np.diag((-1.0) ** np.arange(Nb)))
    n_op = destroy(Nb).dag() * destroy(Nb)
    a_op = destroy(Nb)
    P_ref, n_ref, a_ref = tensor(qeye(Na), P_osc), tensor(qeye(Na), n_op), tensor(qeye(Na), a_op)

    modos = []
    for k, (mu, Xk) in enumerate(zip(evals, Xks)):
        if k == 0:
            continue
        lam = -np.log(mu) / T_r
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
        im_bf = modo_bf['lam'].imag
        k_bf = modo_bf['k']
    else:
        gamma_bf, im_bf, k_bf = float('nan'), float('nan'), None

    cand_n = [m for m in modos if m['ov_n'] > m['ov_P'] and m['ov_n'] > m['ov_a']
              and m['k'] not in (k_pf, k_bf)]
    gamma_conf = min((m['lam'].real for m in cand_n), default=float('nan'))

    return gamma_pf, gamma_bf, im_bf, gamma_conf


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

gamma_pf, gamma_bf, im_bf, gamma_conf = clasificar_y_extraer(evals_top, Xks, Nb)

np.savez(outfile, gz_scale=gz_scale, alpha2=alpha2, delta_m=delta_m, Delta_q=Delta_q,
         Gamma2=Gamma2, Nb=Nb, gamma_pf=gamma_pf, gamma_bf=gamma_bf, im_bf=im_bf,
         gamma_conf=gamma_conf)
print(f"OK dm={delta_m:.5f} Dq={Delta_q:.5f} Nb={Nb} Gamma2={Gamma2:.5f} "
      f"pf={gamma_pf:.4e} bf={gamma_bf:.4e} im_bf={im_bf:.4e} conf={gamma_conf:.4e}")
