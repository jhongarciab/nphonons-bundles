# -*- coding: utf-8 -*-
"""Modelo completo de Ma, Xie y Li (PRA 99, 022302) en el marco rotante exacto (oscilador a wp/2, qubit a wp).
Unidades 2*pi*GHz. QuTiP 5. Qubit: basis(2,0)=|e>, basis(2,1)=|g>.
H_R(t) = (w-wp/2) n + (d-wp)/2 sz + Om (s+ + s-) + gx[s+ a e^{i wp t/2} + s+ a^dag e^{3i wp t/2} + h.c.]
         + gz sz (a e^{-i wp t/2} + h.c.),  L = sqrt(kappa) s-,  kappa = 2 Gam.
El H es periodico con T = 4 pi/wp; en t=nT el marco rotante coincide con el de laboratorio."""
import numpy as np, qutip as qt
from qutip import destroy, qeye, tensor, sigmam, sigmap, sigmaz, basis, coherent

w, d, g, th, Om, Gam = 6.0, 12.0, 0.3, np.pi / 4, 0.06, 0.015
gx, gz, kappa = -g * np.sin(th), g * np.cos(th), 2 * Gam
ALPHA = 2j


def hamiltonian(wp, N):
    a = tensor(qeye(2), destroy(N)); sp = tensor(sigmap(), qeye(N)); sm = sp.dag(); sz = tensor(sigmaz(), qeye(N)); n = a.dag() * a
    H0 = (w - wp / 2) * n + 0.5 * (d - wp) * sz + Om * (sp + sm)
    H = [H0,
         [gx * sp * a, 'exp(1j*wp*t/2)'], [gx * sm * a.dag(), 'exp(-1j*wp*t/2)'],
         [gx * sp * a.dag(), 'exp(3j*wp*t/2)'], [gx * sm * a, 'exp(-3j*wp*t/2)'],
         [gz * sz * a, 'exp(-1j*wp*t/2)'], [gz * sz * a.dag(), 'exp(1j*wp*t/2)']]
    return H, [np.sqrt(kappa) * sm], dict(wp=wp)


def floquet(wp, N, atol=1e-12, rtol=1e-10):
    H, c, args = hamiltonian(wp, N); T = 4 * np.pi / wp
    return qt.propagator(H, T, c, args=args, options={'atol': atol, 'rtol': rtol, 'nsteps': 200000}).full(), T


def observables_ops(N):
    a = destroy(N).full(); ncat = None
    kp, km = coherent(N, ALPHA).full()[:, 0], coherent(N, -ALPHA).full()[:, 0]
    even = (kp + km); even /= np.linalg.norm(even)
    Q, _ = np.linalg.qr(np.array([kp, km]).T)
    return dict(a2=a @ a, par=np.diag(np.exp(1j * np.pi * np.arange(N))), Pi=Q @ Q.conj().T, even=even)


def measure(rho, N, ops):
    """rho: (2N x 2N) con orden (qubit x oscilador). Devuelve F, P_c, paridad, P_e, |<a^2>| y validaciones."""
    tr = np.trace(rho); herm = np.linalg.norm(rho - rho.conj().T); rh = (rho / tr + (rho / tr).conj().T) / 2
    mineig = np.linalg.eigvalsh(rh).min()
    r = rh.reshape(2, N, 2, N); ro = np.einsum('iaib->ab', r)
    Pe = np.real(r[0, :, 0, :].trace())        # qubit indice 0 = |e>
    return dict(F=np.real(ops['even'].conj() @ ro @ ops['even']), Pc=np.real(np.trace(ops['Pi'] @ ro)),
                par=np.real(np.trace(ops['par'] @ ro)), Pe=Pe, a2=abs(np.trace(ops['a2'] @ ro)),
                trace_err=abs(tr - 1), herm=herm, mineig=mineig)


def initial(N):
    k = tensor(basis(2, 1), basis(N, 0)).full()[:, 0]        # |0>|g>
    return np.outer(k, k.conj())


class Evolver:
    """Evoluciona con U^n por cuadrados sucesivos (P[k]=U^(2^k))."""
    def __init__(self, U, N, kmax=17):
        self.N, self.d = N, U.shape[0]; self.P = [U]
        for _ in range(kmax): self.P.append(self.P[-1] @ self.P[-1])
    def rho(self, n, v0):
        v = v0.copy(); k = 0
        while n:
            if n & 1: v = self.P[k] @ v
            n >>= 1; k += 1
        dd = int(round(np.sqrt(len(v)))); return v.reshape(dd, dd, order='F')
