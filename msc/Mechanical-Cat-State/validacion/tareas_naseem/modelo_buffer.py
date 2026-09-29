# -*- coding: utf-8 -*-
"""Tarea 29-bis: buffer de dos niveles (A) vs armonico (B). g=1.
H = g (a^2 - alpha^2) X^dag + h.c. ; A: X=sigma_- , B: X=b (Nb_buffer=6).  Disipacion: kappa D[X] + kappa1 D[a]."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, scipy.linalg as sl
from qutip import destroy, qeye, tensor, sigmam, liouvillian

KAPPA1 = 1e-3   # en unidades de g


def build(model, alpha2, kappa, Nb_buf=6):
    Nc = int(np.ceil(alpha2 + 6 * np.sqrt(alpha2))) + 4
    a = tensor(destroy(Nc), qeye(2 if model == 'A' else Nb_buf))
    X = tensor(qeye(Nc), sigmam() if model == 'A' else destroy(Nb_buf))
    core = a * a - alpha2 * tensor(qeye(Nc), qeye(2 if model == 'A' else Nb_buf))
    H = core * X.dag() + (core * X.dag()).dag()
    L = liouvillian(H, [np.sqrt(kappa) * X, np.sqrt(KAPPA1) * a])
    return L, Nc, (2 if model == 'A' else Nb_buf)


def _vec2mat(v, d):
    return v.reshape(d, d, order='F')


def brecha_union(M, slow4, omegas=range(0, 13), k=24, tol_slow=1e-3):
    """Autovalor con menor tasa fuera de los 4 modos logicos. Shift-invert cerca de 0 pierde modos
    con |Im| grande, y ARPACK 'LR' converge mal con kappa/g chico: se sondea sigma=0.05+i*omega
    (omega=0..12) y se une (con conjugados)."""
    al = []
    for w in omegas:
        e = sla.eigs(M, k=k, sigma=0.05 + 1j * w, which='LM', tol=1e-10, ncv=2 * k + 1, return_eigenvectors=False)
        al += list(e) + list(np.conj(e))
    u = []
    for x in np.array(al)[np.argsort(-np.array(al).real)]:
        if not any(abs(x - y) < 1e-6 for y in u): u.append(x)
    u = list(u)
    for s4 in slow4:                       # quita cada modo logico por proximidad
        if u:
            j = int(np.argmin([abs(x - s4) for x in u]))
            if abs(u[j] - s4) < tol_slow: u.pop(j)
    return max(u, key=lambda x: x.real)


def espectro_bajo(L, dense_max=3000):
    """Devuelve (rates[0..7] ordenadas, ims, matrices de los 4 primeros modos (k=0..3), gap, gap_im).
    Denso si cabe; si no, shift-invert cerca de 0 para k=0..3 (precision en tasas ~1e-12) y
    ARPACK 'LR' (mayor Re) para el 5o modo: el shift-invert pierde modos con |Im| grande."""
    n = L.shape[0]; d = int(round(np.sqrt(n)))
    if n <= dense_max:
        w, v = sl.eig(L.full()); o = np.argsort(-w.real)[:8]; w, v = w[o], v[:, o]
        return w, [_vec2mat(v[:, i], d) for i in range(4)]
    M = L.data.tocsc()
    w1, v1 = sla.eigs(M, k=8, sigma=1e-3, which='LM', tol=1e-14, ncv=48)
    o = np.argsort(-w1.real); w1, v1 = w1[o], v1[:, o]
    gap = brecha_union(M, w1[:4])
    w = np.concatenate([w1[:4], [gap]])   # k=0..3 precisos; k=4 = 5o por Re (union de shift-invert complejos)
    return w, [_vec2mat(v1[:, i], d) for i in range(4)]


def analizar(L, Nc, dq):
    w, X = espectro_bajo(L)
    Pop = np.kron(np.diag((-1.0) ** np.arange(Nc)), np.eye(dq))
    aop = np.kron(np.diag(np.sqrt(np.arange(1, Nc)), 1), np.eye(dq))
    ov = lambda O, M: abs(np.trace(O.conj().T @ M)) / np.linalg.norm(M)
    rate = -w.real
    modos = [dict(rate=rate[i], im=w[i].imag, P=ov(Pop, X[i]), a=ov(aop, X[i])) for i in range(4)]
    lg = modos[1:4]
    pf = max(lg, key=lambda m: m['P']); bf = max(lg, key=lambda m: m['a'])
    ok = pf is not bf and bf['a'] > pf['a'] and pf['P'] > bf['P']
    return dict(gap=rate[4], gap_im=w[4].imag, gamma_pf=pf['rate'], gamma_bf=bf['rate'],
                stationary=modos[0]['rate'], ok=ok)
