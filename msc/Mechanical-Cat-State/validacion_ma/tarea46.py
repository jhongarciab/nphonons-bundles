# -*- coding: utf-8 -*-
"""Tarea 46: esquema de Liu et al. (arXiv:2501.08675), Ec. (9) en la base vestida. Unidades 2*pi*MHz = 1.
H = nu/2 sz~ + Dm m^dag m + gz (m+m^dag) sz~ + gx (m+m^dag) sx~ + eps_p cos(wp t) sx~.
(i) gamma D[s~-]; (ii) gamma D[s-] con s- bare expresado en la base vestida (theta=pi/4):
    s- = ((cos t -1)/2) s~+ + ((cos t +1)/2) s~- + (sin t /2) s~z.
Se simula en el marco de laboratorio (mesolve, string coeff) y se rota el oscilador a wp/2 para P_c, F."""
import sys, numpy as np, qutip as qt
from qutip import destroy, qeye, tensor, sigmam, sigmap, sigmaz, sigmax, basis, coherent

nu, G, epsp, gam = 35.4, 10.0, 3.53, 16.0
gx = gz = np.sqrt(2) * G / 4; Dm = nu / 2; th = np.pi / 4
xk = gam / 2
ReS = lambda D: xk / (xk**2 + D**2); ImS = lambda D: -D / (xk**2 + D**2)
w = Dm                                   # frecuencia del oscilador
G1m, G1p = 2 * gx**2 * ReS(w), 2 * gx**2 * ReS(3 * w)                 # con kappa^2/4 en los denominadores
G1m0, G1p0 = 2 * gx**2 * (xk / w**2), 2 * gx**2 * (xk / (3 * w)**2)     # sin el termino kappa^2/4
delta_osc = gx**2 * (ImS(w) + ImS(3 * w))
Gexch = 2 * gx * gz / w; alpha2_an = (epsp / 2) / abs(Gexch)


def simulate(version, wp, N, gts):
    a = tensor(qeye(2), destroy(N)); sz = tensor(sigmaz(), qeye(N)); sxq = tensor(sigmax(), qeye(N)); n = a.dag() * a
    H0 = nu / 2 * sz + Dm * n + gz * (a + a.dag()) * sz + gx * (a + a.dag()) * sxq
    H = [H0, [epsp * sxq, 'cos(wp*t)']]
    sm = tensor(sigmam(), qeye(N)); sp = sm.dag()
    L = np.sqrt(gam) * sm if version == 'i' else np.sqrt(gam) * (((np.cos(th) - 1) / 2) * sp + ((np.cos(th) + 1) / 2) * sm + (np.sin(th) / 2) * sz)
    psi0 = tensor(basis(2, 1), basis(N, 0)); ts = np.asarray(gts) / gam
    r = qt.mesolve(H, psi0, ts, [L], args=dict(wp=wp), options={'atol': 1e-10, 'rtol': 1e-8, 'nsteps': 1000000, 'progress_bar': False})
    return ts, r.states


def analyze(ts, states, wp, N, alpha=None):
    nop = np.arange(N); a = destroy(N).full(); par = np.diag((-1.0) ** nop)
    out = dict(par=[], Pe=[], a2=[], herm=[], trace=[], mineig=[], rho=[])
    rhos = []
    for t, s in zip(ts, states):
        rho = s.full() if s.isoper else np.outer(s.full()[:, 0], s.full()[:, 0].conj()); rhos.append(rho)
        tr = np.trace(rho); r = ((rho / tr) + (rho / tr).conj().T) / 2; q = r.reshape(2, N, 2, N); ro = np.einsum('iaib->ab', q)
        out['par'].append(np.real(np.trace(par @ ro))); out['Pe'].append(np.real(np.einsum('aiai->', q[:1, :, :1, :])))
        out['a2'].append(np.trace(a @ a @ ro) * np.exp(1j * wp * t)); out['herm'].append(np.linalg.norm(rho - rho.conj().T)); out['trace'].append(abs(tr - 1)); out['mineig'].append(np.linalg.eigvalsh(r).min())
    a2c = np.array(out['a2']); late = a2c[-6:].mean()
    if alpha is None: alpha = np.sqrt(late + 0j)                      # cat objetivo autoconsistente (rotante)
    kp, km = coherent(N, alpha).full()[:, 0], coherent(N, -alpha).full()[:, 0]; even = kp + km; even = even / np.linalg.norm(even)
    Q, _ = np.linalg.qr(np.array([kp, km]).T); Pi = Q @ Q.conj().T
    Pc, F = [], []
    for t, rho in zip(ts, rhos):
        tr = np.trace(rho); q = (rho / tr).reshape(2, N, 2, N); ro = np.einsum('iaib->ab', q)
        R = np.diag(np.exp(1j * wp / 2 * t * nop)); rr = R @ ro @ R.conj().T          # oscilador en el marco rotante a wp/2
        Pc.append(np.real(np.trace(Pi @ rr))); F.append(np.real(even.conj() @ rr @ even))
    res = {k: np.array(v) for k, v in out.items() if k != 'rho'}; res['Pc'] = np.array(Pc); res['F'] = np.array(F); res['alpha'] = alpha
    return res
