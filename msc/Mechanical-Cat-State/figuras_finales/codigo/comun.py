"""Utilidades comunes de las figuras finales (modelo completo en el marco de laboratorio).

H(t) = w a†a + (wq/2)σz + (a+a†)(gx σx + gz σz) + Ω(σ+ e^{-i wp t} + h.c.), κ D[σ-].
Convención de estados: |e> = basis(2,0), |g> = basis(2,1), σz|g> = -|g>.
Floquet: propagador del superoperador sobre T_p = 2π/wp; muestreo estroboscópico t = nT_p (fase 0 del drive).
Base polarónica (C6): con el qubit en |g> el oscilador está desplazado en +gz/w (real) en el laboratorio;
en t = nT_p el código es {D(gz/w)|±α>}.
"""
import os, time
import numpy as np
import qutip as qt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # figuras_finales/
AQUI = os.path.join(RAIZ, 'figuras')                                        # salida de figuras (PDF/PNG)
DATA = os.path.join(RAIZ, 'data')
OPTS = dict(atol=1e-12, rtol=1e-10, nsteps=10**6)


def ops(N):
    a = qt.tensor(qt.destroy(N), qt.qeye(2))
    sm = qt.tensor(qt.qeye(N), qt.sigmam())
    sz = qt.tensor(qt.qeye(N), qt.sigmaz())
    sx = qt.tensor(qt.qeye(N), qt.sigmax())
    return a, sm, sz, sx


def hamiltoniano(N, w, wq, gx, gz, Om, wp):
    a, sm, sz, sx = ops(N)
    H0 = w * a.dag() * a + 0.5 * wq * sz + (a + a.dag()) * (gx * sx + gz * sz)
    return [H0, [Om * sm.dag(), lambda t: np.exp(-1j * wp * t)], [Om * sm, lambda t: np.exp(1j * wp * t)]]


def propagador(N, w, wq, gx, gz, Om, wp, kap):
    a, sm, sz, sx = ops(N)
    H = hamiltoniano(N, w, wq, gx, gz, Om, wp)
    t0 = time.time()
    U = qt.propagator(H, 2 * np.pi / wp, [np.sqrt(kap) * sm], options=OPTS).full()
    return U, time.time() - t0


def estacionario(U, N):
    """Autovector de λ≈1 del propagador (estado estroboscópico asintótico)."""
    lam, R = np.linalg.eig(U)
    k = np.argmin(abs(lam - 1))
    D = 2 * N
    M = R[:, k].reshape(D, D, order='F')
    M = M / np.trace(M); M = (M + M.conj().T) / 2
    return M, lam


def validar(M):
    return (abs(np.trace(M) - 1), np.linalg.norm(M - M.conj().T), np.linalg.eigvalsh((M + M.conj().T) / 2).min())


def proyector_codigo(N, alpha, d=0.0):
    """Proyector sobre span{D(d)|α>, D(d)|-α>} ⊗ 1_qubit."""
    Dd = qt.displace(N, d)
    ca, cb = Dd * qt.coherent(N, alpha), Dd * qt.coherent(N, -alpha)
    return qt.tensor((ca + cb).unit().proj() + (ca - cb).unit().proj(), qt.qeye(2)).full()


def Pc(M, N, alpha, d=0.0):
    return np.real(np.trace(proyector_codigo(N, alpha, d) @ M))
