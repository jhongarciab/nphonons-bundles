"""Memoria de fonones con estados oscuros: modelo reducido autocontenido (1 exciton + modo), carga/lectura ideal.
Unidades: omega_b=1. Compara 3 puntos (articulo) y 4 puntos en rectangulo (propuesta A2) bajo ruido de sitio con correlacion chi."""
import numpy as np, scipy.linalg as sl, qutip as qt
from scipy.optimize import minimize
NC = 3
def h3(G1=.35, G2=.6):
    e1 = -2*G2/3; e2 = -G1/2 - G2/6; j12 = G2/3; j23 = G1/2 - G2/6
    return np.array([[e1, j12, j12], [j12, e2, j23], [j12, j23, e2]])
def h4(a=.12, ap=.05, b=.03):
    J = np.array([[0, a, b, ap], [a, 0, ap, b], [b, ap, 0, a], [ap, b, a, 0]], float)
    w = np.linalg.eigvalsh(J); return J - w.max()*np.eye(4)   # brillante en E=0
class Sys:
    def __init__(s, h, lam, slots, kappa=1e-3, gam=2e-4, gph=4e-4, chi=1.0):
        s.n = len(h); s.h = h; s.lam = np.array(lam, float); s.kappa, s.gam, s.gph, s.chi = kappa, gam, gph, chi
        w, V = np.linalg.eigh(h); s.w, s.V = w, V
        s.B = V[:, np.argmax(w)]; s.B = s.B*np.sign(s.B.sum())
        s.D = {}
        for k, idx in enumerate(slots): s.D[k] = V[:, idx]
        s.E = {k: w[i] for k, i in enumerate(slots)}
        s.Lam = np.diag(s.lam)
    # --- espacio exc x modo (cierre) ---
    def Hclosed(s, wl, Om):
        n, d = s.n, s.n + 1
        He = np.zeros((d, d)); He[1:, 1:] = s.h - wl*np.eye(n)
        Hl = np.zeros((d, d)); Hl[1:, 0] = Om/np.sqrt(n); Hl += Hl.T
        Lm = np.zeros((d, d)); Lm[1:, 1:] = s.Lam
        a = np.diag(np.sqrt(np.arange(1, NC)), 1)
        return np.kron(He + Hl, np.eye(NC)) + np.kron(np.eye(d), a.T@a) + np.kron(Lm, a + a.T)
    def ket(s, vec, m):
        d = s.n + 1; e = np.zeros(d); e[1:] = vec; f = np.zeros(NC); f[m] = 1; return np.kron(e, f)
    def pulse(s, s_idx, Om, dur, d2):
        wl = s.E[s_idx] - 1.0 - d2          # resonancia de dos fotones |G,1> <-> |D_s,0>
        U = sl.expm(-1j*s.Hclosed(wl, Om)*dur); return wl, U
    def transfer(s, s_idx, p):
        Om, dur, d2 = p; wl, U = s.pulse(s_idx, Om, dur, d2)
        psi = U@s.ket(np.zeros(s.n), 1)*0 if False else U@np.kron(np.eye(s.n+1)[0], np.eye(NC)[1])
        return abs(s.ket(s.D[s_idx], 0)@psi)**2
    def optimize(s, s_idx, guess):
        best = None
        for sc in (1, .8, 1.2):
            g = np.array(guess)*np.array([sc, sc, 1])
            r = minimize(lambda p: 1 - s.transfer(s_idx, p), g, method='Nelder-Mead', options=dict(xatol=1e-7, fatol=1e-10, maxiter=4000))
            if best is None or r.fun < best.fun: best = r
        return best.x, 1 - best.fun
    # --- sistema abierto: propagadores exactos (superoperador) en [exc, modo, R]; A2 solo en el ultimo tramo ---
    def build(s):
        n = s.n; d = n + 1; s.d = d
        s.dims = [d, NC, 2]
        def op(x, pos, dims=None):
            dims = dims or s.dims
            return qt.tensor(*[x if k == pos else qt.qeye(dm) for k, dm in enumerate(dims)])
        s.op = op
        s.b = op(qt.destroy(NC), 1)
        cs = [np.sqrt(s.kappa)*s.b]
        Lg = np.zeros((d, d)); Lg[0, 1:] = np.sqrt(2*s.gam/n); cs.append(op(qt.Qobj(Lg), 0))
        K = (1 - s.chi)*np.eye(n) + s.chi*np.ones((n, n)); mu, U = np.linalg.eigh(K)
        for a in range(n):
            if mu[a] < 1e-12: continue
            L = np.zeros((d, d)); L[1:, 1:] = np.diag(U[:, a]); cs.append(op(qt.Qobj(np.sqrt(s.gph*mu[a])*L), 0))
        s.cs = cs
    def Hop(s, wl, Om):
        return qt.Qobj(np.kron(s.Hclosed(wl, Om), np.eye(2)), dims=[s.dims, s.dims])
    def prop(s, H, t):
        return (qt.liouvillian(H, s.cs)*t).expm()
    def run(s, T, pulses, tau12=100.0, tol=1e-9):
        s.build(); op = s.op
        (O1, d1, e1), (O2, d2, e2) = pulses
        R0, R1 = qt.basis(2, 0), qt.basis(2, 1)
        wl1, _ = s.pulse(0, O1, d1, e1); wl2, _ = s.pulse(1, O2, d2, e2)
        H1 = s.Hop(wl1, O1); H2 = s.Hop(wl2, O2); Hoff = s.Hop(0.0, 0.0)
        X = qt.Qobj(np.eye(NC)[[1, 0] + list(range(2, NC))])
        def load(r, k):
            Pr = R0*R0.dag() if k == 0 else R1*R1.dag(); Qr = qt.qeye(2) - Pr
            U = op(X, 1)*op(Pr, 2) + op(qt.qeye(NC), 1)*op(Qr, 2); return U*r*U.dag()
        psi = qt.tensor(qt.basis(s.d, 0), qt.basis(NC, 0), (R0 + R1).unit()); r = psi*psi.dag()
        P1 = s.prop(H1, d1); P2 = s.prop(H2, d2)
        r = load(r, 0); r = P1(r); r = s.prop(Hoff, tau12)(r)
        r = load(r, 1); r = P2(r); r = s.prop(Hoff, T)(r)
        r = P2(r)
        dims4 = s.dims + [2]
        S = np.zeros((s.d*NC*2*2,)*2)
        for flat in range(S.shape[0]):
            idx = np.unravel_index(flat, dims4); m, a = idx[1], idx[3]; new = list(idx)
            if (m, a) == (1, 0): new[1], new[3] = 0, 1
            elif (m, a) == (0, 1): new[1], new[3] = 1, 0
            S[np.ravel_multi_index(new, dims4), flat] = 1
        U = qt.Qobj(S, dims=[dims4, dims4])
        r4 = U*qt.tensor(r, qt.fock_dm(2, 0))*U.dag()
        H1_4 = qt.tensor(H1, qt.qeye(2)); cs4 = [qt.tensor(c, qt.qeye(2)) for c in s.cs]
        r4 = (qt.liouvillian(H1_4, cs4)*d1).expm()(r4)
        rho = r4.ptrace([1, 2, 3])
        ket = lambda m, R, a2: qt.tensor(qt.basis(NC, m), qt.basis(2, R), qt.basis(2, a2))
        k0, k1 = ket(1, 0, 0), ket(0, 1, 1)
        a = float(np.real(qt.expect(k0*k0.dag(), rho))); b = float(np.real(qt.expect(k1*k1.dag(), rho))); c = abs(complex(k0.dag()*rho*k1))
        sv = sum(float(np.real(qt.expect(ket(x, R, 1-x)*ket(x, R, 1-x).dag(), rho))) for R in (0, 1) for x in (0, 1))
        F = .5*(a + b) + c
        return dict(a=a, b=b, c=c, s=sv, F=F, Fh=F/sv, V=2*c/(a+b) if a+b > 0 else 0)
