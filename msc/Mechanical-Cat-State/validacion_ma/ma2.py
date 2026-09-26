# -*- coding: utf-8 -*-
"""Modelo de Ma (marco rotante exacto) con parametros libres, re-sintonizado (Tarea 42) y con baño filtrado opcional (Tarea 43).
Orden de subsistemas: (qubit, oscilador[, filtro]). Qubit: basis(2,0)=|e>, basis(2,1)=|g>."""
import numpy as np, qutip as qt
from qutip import destroy, qeye, tensor, sigmam, sigmap, sigmaz, basis, coherent

ALPHA = 2j


def params(w=6.0, gx=0.05, gz=0.15, kappa=0.03, alpha2=4.0, retune=True, wp=None):
    """gx<0 y gz>0 (convencion de Ma). Om=|alpha|^2 G, G=2 gx gz/w. Re-sintonia: wp=2(w-4gx^2/(3w)), d=wp."""
    gx = -abs(gx); gz = abs(gz); G = 2 * gx * gz / w
    wp = 2 * (w - 4 * gx**2 / (3 * w)) if wp is None else wp
    return dict(w=w, gx=gx, gz=gz, kappa=kappa, alpha2=alpha2, G=G, Om=alpha2 * abs(G), wp=wp, d=wp,
                kappa2=4 * G**2 / kappa)


def hamiltonian(p, N, filt=None, counter=True, gz_direct=True, pair=False):
    """filt=None: kappa D[s-]. filt=dict(kf, J, Nf): filtro f (2w) acoplado J(s+ f + s- f^dag), decae kf; qubit sin decaimiento directo."""
    Nf = 1 if filt is None else filt['Nf']; ids = qeye(Nf)
    a = tensor(qeye(2), destroy(N), ids); sp = tensor(sigmap(), qeye(N), ids); sm = sp.dag(); sz = tensor(sigmaz(), qeye(N), ids); n = a.dag() * a
    wp, w, d = p['wp'], p['w'], p['d']; gx, gz = p['gx'], p['gz']
    H0 = (w - wp / 2) * n + 0.5 * (d - wp) * sz + p['Om'] * (sp + sm)
    if pair: H0 = H0 - p['G'] * (sp * a * a + sm * a.dag() * a.dag())     # intercambio de pares explicito (estatico en el marco rotante)
    H = [H0, [gx * sp * a, 'exp(1j*wp*t/2)'], [gx * sm * a.dag(), 'exp(-1j*wp*t/2)']]
    if counter: H += [[gx * sp * a.dag(), 'exp(3j*wp*t/2)'], [gx * sm * a, 'exp(-3j*wp*t/2)']]
    if gz_direct: H += [[gz * sz * a, 'exp(-1j*wp*t/2)'], [gz * sz * a.dag(), 'exp(1j*wp*t/2)']]
    if filt is None: c = [np.sqrt(p['kappa']) * sm]
    else:
        f = tensor(qeye(2), qeye(N), destroy(Nf))
        H[0] = H0 + (2 * w - wp) * f.dag() * f + filt['J'] * (sp * f + sm * f.dag())
        c = [np.sqrt(filt['kf']) * f]
    return H, c, dict(wp=wp)


def floquet(p, N, filt=None, atol=1e-13, rtol=1e-11, **hopts):
    H, c, args = hamiltonian(p, N, filt, **hopts); T = 4 * np.pi / p['wp']
    return qt.propagator(H, T, c, args=args, options={'atol': atol, 'rtol': rtol, 'nsteps': 500000}).full(), T


def ops(N, Nf=1, alpha2=4.0):
    al = 1j * np.sqrt(alpha2)     # cat objetivo alpha = i |alpha|
    a = destroy(N).full(); kp, km = coherent(N, al).full()[:, 0], coherent(N, -al).full()[:, 0]
    even = kp + km; even /= np.linalg.norm(even); Q, _ = np.linalg.qr(np.array([kp, km]).T)
    return dict(a2=a @ a, par=np.diag(np.exp(1j * np.pi * np.arange(N))), Pi=Q @ Q.conj().T, even=even, N=N, Nf=Nf)


def initial(N, Nf=1):
    k = tensor(basis(2, 1), basis(N, 0), basis(Nf, 0)).full()[:, 0]; return np.outer(k, k.conj())


def measure(rho, o):
    N, Nf = o['N'], o['Nf']; tr = np.trace(rho); herm = np.linalg.norm(rho - rho.conj().T); rh = (rho / tr + (rho / tr).conj().T) / 2
    r = rh.reshape(2, N, Nf, 2, N, Nf); ro = np.einsum('iafibf->ab', r)
    return dict(F=np.real(o['even'].conj() @ ro @ o['even']), Pc=np.real(np.trace(o['Pi'] @ ro)), par=np.real(np.trace(o['par'] @ ro)),
                Pe=np.real(np.einsum('iafiaf->', r[:1, :, :, :1, :, :])), a2=abs(np.trace(o['a2'] @ ro)),
                trace_err=abs(tr - 1), herm=herm, mineig=np.linalg.eigvalsh(rh).min())


class Evolver:
    def __init__(self, U, kmax=20):
        self.P = [U]
        for _ in range(kmax): self.P.append(self.P[-1] @ self.P[-1])
    def vec(self, n, v0):
        v = v0.copy(); k = 0
        while n:
            if n & 1: v = self.P[k] @ v
            n >>= 1; k += 1
        return v

