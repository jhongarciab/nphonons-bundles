# -*- coding: utf-8 -*-
"""Modelo completo (marco conmensurable, Tarea 24) y efectivo, compartido por Tareas 26-27.
Qubit en basis(2,1) (estado base fisico) -- no se usa estado inicial aqui (solo espectro)."""
import numpy as np
from qutip import (tensor, qeye, destroy, sigmam, sigmap, sigmaz, Options, propagator,
                   vector_to_operator, Qobj, liouvillian)

r = 0.1
gz_org, omega_m_org, Gamma_m_org, kappa_org = (2 * np.pi * 6e6, 2 * np.pi * 100e6,
                                                2 * np.pi * 15, 2 * np.pi * 100e3)
gz_baseline = gz_org / kappa_org
om_m = omega_m_org / kappa_org
Gam_m = Gamma_m_org / kappa_org
kap, n_th, Na = 1.0, 0, 2
gx = r * gz_baseline
wr = om_m
T_r = 2 * np.pi / wr


def gz_scale_from_Gamma2(G2):
    """Gamma2/kappa = 4 g_eff^2 ; g_eff = 2 gz gx/om_m  (calibrado: 0.5 <-> 0.5184)."""
    return 0.5 * np.sqrt(G2 / 0.5184)


def params(gz_scale, alpha2):
    gz = gz_scale * gz_baseline
    g_eff = 2 * gz * gx / om_m
    return dict(gz=gz, g_eff=g_eff, eps=alpha2 * g_eff, Gamma2=4 * g_eff**2 / kap,
                Nb=max(20, int(4 * alpha2 + 12)))


def full_propagator(gz_scale, alpha2, delta_m, Delta_q, atol=1e-12, rtol=1e-10):
    p = params(gz_scale, alpha2); Nb, gz, eps = p['Nb'], p['gz'], p['eps']
    b = tensor(qeye(Na), destroy(Nb)); bd = b.dag()
    sm = tensor(sigmam(), qeye(Nb)); sp = tensor(sigmap(), qeye(Nb)); sz = tensor(sigmaz(), qeye(Nb))
    H = [delta_m * bd * b + (Delta_q / 2) * sz + eps * (sp + sm),
         [gx * sp * b, lambda t, _: np.exp(1j * wr * t)],
         [gx * sp * bd, lambda t, _: np.exp(3j * wr * t)],
         [gx * sm * b, lambda t, _: np.exp(-3j * wr * t)],
         [gx * sm * bd, lambda t, _: np.exp(-1j * wr * t)],
         [gz * sz * b, lambda t, _: np.exp(-1j * wr * t)],
         [gz * sz * bd, lambda t, _: np.exp(1j * wr * t)],
         [eps * sp, lambda t, _: np.exp(4j * wr * t)],
         [eps * sm, lambda t, _: np.exp(-4j * wr * t)]]
    c_ops = [np.sqrt(kap) * sm, np.sqrt((n_th + 1) * Gam_m) * b, np.sqrt(n_th * Gam_m) * bd]
    opts = Options(atol=atol, rtol=rtol, nsteps=2_000_000)
    return propagator(H, T_r, c_ops, options=opts), p


def effective_liouvillian(gz_scale, alpha2, Delta_2m):
    """Efectivo SIN delta_1 a^dag a (compensado por delta_m=-delta_1), con Delta_2- = Delta_2m.
    Delta_2+ = 4 om_m + Delta_2m."""
    p = params(gz_scale, alpha2); Nb, g_eff, eps = p['Nb'], p['g_eff'], p['eps']
    x = kap / 2.0
    D2m, D2p = Delta_2m, 4 * om_m + Delta_2m
    ReS = lambda D: x / (x**2 + D**2)
    ImS = lambda D: -D / (x**2 + D**2)
    D1m, D1p = om_m, 3 * om_m
    G1m = 2 * gx**2 * ReS(D1m); G1p = 2 * gx**2 * ReS(D1p)
    a = destroy(Nb); ad = a.dag()
    G2m, G2p = 2 * g_eff**2 * ReS(D2m), 2 * g_eff**2 * ReS(D2p)
    dk = g_eff**2 * (ImS(D2m) + ImS(D2p))
    chi = -2j * eps * g_eff / kap
    H_eff = chi.conjugate() * ad**2 + chi * a**2 + dk * (ad * a)**2
    c = [np.sqrt(G1m) * a, np.sqrt(G1p) * ad, np.sqrt(G2m) * (a * a), np.sqrt(G2p) * (ad * ad),
         np.sqrt((n_th + 1) * Gam_m) * a, np.sqrt(n_th * Gam_m) * ad]
    return liouvillian(H_eff, c), p


def modos_ordenados(evals, evecs, Nb, es_completo, T_eff, n=12):
    """Devuelve lista de modos ordenados por Re(lambda) ascendente (k=0 estacionario)."""
    if es_completo:
        lam_all = -np.log(evals.astype(complex)) / T_eff
    else:
        lam_all = -evals
    orden = np.argsort(lam_all.real)[:n]
    P = Qobj(np.diag((-1.0) ** np.arange(Nb))); nop = destroy(Nb).dag() * destroy(Nb); aop = destroy(Nb)
    if es_completo:
        P, nop, aop = tensor(qeye(Na), P), tensor(qeye(Na), nop), tensor(qeye(Na), aop)
    out = []
    for k, i in enumerate(orden):
        X = vector_to_operator(evecs[i]); nx = X.norm()
        out.append(dict(k=k, lam=lam_all[i], mu=(evals[i] if es_completo else np.nan),
                        ov_P=abs((P.dag() * X).tr()) / nx, ov_n=abs((nop.dag() * X).tr()) / nx,
                        ov_a=abs((aop.dag() * X).tr()) / nx))
    return out


def bf_de_modos(modos):
    """gamma_bf: min Re(lambda) entre modos k>=1 con overlap 'a' dominante (misma regla que Tarea 25)."""
    c = [m for m in modos[1:] if m['ov_a'] > m['ov_P'] and m['ov_a'] > m['ov_n']]
    return min(c, key=lambda m: m['lam'].real) if c else None
