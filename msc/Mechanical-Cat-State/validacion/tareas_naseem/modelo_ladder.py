# -*- coding: utf-8 -*-
"""Tareas 31-32: modelo minimo con buffer de dos niveles explicito y escalera de ingredientes.
H = G[(a^2-alpha^2) s+ + h.c.], L = sqrt(kappa) s-, con Gamma2 = 4 G^2/kappa (kappa=1).
Los valores de cada ingrediente salen de las mismas formulas que modelo_comun.effective_liouvillian
(gx, om_m, Gam_m, delta_m, Delta_q, ...), evaluadas con g_eff = G."""
import numpy as np, scipy.linalg as sl
from qutip import destroy, qeye, tensor, sigmam, sigmap, basis, coherent, liouvillian, steadystate
import modelo_comun as mc

DM, DQ = 0.048, 0.144
x = mc.kap / 2.0
ReS = lambda D: x / (x**2 + D**2)
ImS = lambda D: -D / (x**2 + D**2)

VARIANTS = {   # dq, dm, lamb, oneph, kerr(None|'p'|'full'), nonres
    'min': dict(),
    's1': dict(dq=1, dm=1),
    's2': dict(dq=1, dm=1, lamb=1),
    's3': dict(dq=1, dm=1, lamb=1, oneph=1),
    's4': dict(dq=1, dm=1, lamb=1, oneph=1, kerr='p'),
    's5': dict(dq=1, dm=1, lamb=1, oneph=1, kerr='p', nonres=1),
    's4L': dict(dq=1, dm=1, lamb=1, oneph=1, kerr='full'),
    's5L': dict(dq=1, dm=1, lamb=1, oneph=1, kerr='full', nonres=1),
    # quitar UN ingrediente a s5
    'no_dq': dict(dm=1, lamb=1, oneph=1, kerr='p', nonres=1),
    'no_dmlamb': dict(dq=1, oneph=1, kerr='p', nonres=1),
    'no_oneph': dict(dq=1, dm=1, lamb=1, kerr='p', nonres=1),
    'no_nonres': dict(dq=1, dm=1, lamb=1, oneph=1, kerr='p'),
    'no_kerrp': dict(dq=1, dm=1, lamb=1, oneph=1, nonres=1),
}


def valores(G2):
    G = 0.5 * np.sqrt(G2 * mc.kap)            # Gamma2 = 4 G^2/kappa
    D1m, D1p, D2m, D2p = mc.om_m, 3 * mc.om_m, DQ, 4 * mc.om_m + DQ
    return dict(G=G, dq=DQ, dm=DM, lamb=mc.gx**2 * (ImS(D1m) + ImS(D1p)),
                G1m=2 * mc.gx**2 * ReS(D1m), G1p=2 * mc.gx**2 * ReS(D1p), gam_m=mc.Gam_m,
                dk_p=G**2 * ImS(D2p), dk_full=G**2 * (ImS(D2m) + ImS(D2p)), G2p=2 * G**2 * ReS(D2p))


def build(G2, alpha2, N, var):
    f = VARIANTS[var]; v = valores(G2)
    a = tensor(destroy(N), qeye(2)); sm = tensor(qeye(N), sigmam()); sp = sm.dag()
    c = a * a - alpha2 * tensor(qeye(N), qeye(2))
    H = v['G'] * (c * sp + (c * sp).dag())
    n = a.dag() * a
    if f.get('dq'): H = H + v['dq'] * sp * sm
    if f.get('dm'): H = H + v['dm'] * n
    if f.get('lamb'): H = H + v['lamb'] * n
    if f.get('kerr') == 'p': H = H + v['dk_p'] * n * n
    if f.get('kerr') == 'full': H = H + v['dk_full'] * n * n
    cops = [np.sqrt(mc.kap) * sm]
    if f.get('oneph'):
        cops += [np.sqrt(v['G1m'] + v['gam_m']) * a, np.sqrt(v['G1p']) * a.dag()]
    if f.get('nonres'): cops.append(np.sqrt(v['G2p']) * a.dag() * a.dag())
    return liouvillian(H, cops)


def analizar(L, N, alpha2, k=12):
    """Devuelve dict: gap robusta (5o por Re, como Tarea 27), gap simple (excluye 4 modos del espacio
    del codigo por peso en el), tasas, validaciones del estacionario."""
    n = L.shape[0]; d = int(round(np.sqrt(n)))
    w, v = sl.eig(L.full()); o = np.argsort(-w.real)[:k]; w, v = w[o], v[:, o]
    X = [v[:, i].reshape(d, d, order='F') for i in range(k)]
    rate = -w.real
    P = np.kron(np.diag((-1.0) ** np.arange(N)), np.eye(2)); A = np.kron(np.diag(np.sqrt(np.arange(1, N)), 1), np.eye(2))
    ov = lambda O, M: abs(np.trace(O.conj().T @ M)) / np.linalg.norm(M)
    # proyector del espacio del codigo (oscilador) x qubit en el estado base basis(2,1)
    if alpha2 > 0:
        al = np.sqrt(alpha2); kets = [coherent(N, s * al).full()[:, 0] for s in (1, -1)]
    else:
        kets = [np.eye(N)[:, 0], np.eye(N)[:, 1]]
    Q, _ = np.linalg.qr(np.array(kets).T); Pi = np.kron(Q @ Q.conj().T, np.diag([0.0, 1.0]))
    wcode = np.array([np.linalg.norm(Pi @ M @ Pi) / np.linalg.norm(M) for M in X])
    code4 = set(np.argsort(-wcode)[:4]); resto = [i for i in range(k) if i not in code4]
    gap_simple = min(rate[i] for i in resto); gap_rob = rate[4]
    lg = [1, 2, 3]; pf = max(lg, key=lambda i: ov(P, X[i])); bf = max(lg, key=lambda i: ov(A, X[i]))
    # validaciones del estado estacionario (solo si es unico)
    unico = rate[1] > 1e-11
    val = dict(unico=unico, trace_err=np.nan, herm=np.nan, min_eig=np.nan, purity=np.nan, a2=np.nan, pexc=np.nan)
    if unico:
        rho = steadystate(L, method='direct').full()   # resuelve L rho = 0, Tr rho = 1 (mas preciso que el autovector)
        val['trace_err'] = abs(np.trace(rho) - 1); val['herm'] = np.linalg.norm(rho - rho.conj().T)
        rh = (rho + rho.conj().T) / 2; val['min_eig'] = np.linalg.eigvalsh(rh).min()
        val['purity'] = np.real(np.trace(rh @ rh)); a_full = np.kron(np.diag(np.sqrt(np.arange(1, N)), 1), np.eye(2))
        val['a2'] = abs(np.trace(rh @ a_full @ a_full))
        val['pexc'] = np.real(np.trace(rh @ np.kron(np.eye(N), np.diag([1.0, 0.0]))))
    return dict(gap_rob=gap_rob, gap_simple=gap_simple, gap_im=w[4].imag, rates=rate, ims=w.imag,
                gamma_pf=rate[pf], gamma_bf=rate[bf], **{f"v_{a}": b for a, b in val.items()},
                wcode=wcode, code4=np.array(sorted(code4)))
