"""Reoptimiza el control en el modelo Raman COMPLETO para el estado D1 (gauss Tp=100): parametros (escala de G, retraso, desintonia constante).
Evalua una rejilla en paralelo y luego refina con Nelder-Mead sobre la mejor celda."""
import sys, time, numpy as np, qutip as qt, itertools
from multiprocessing import Pool
from mem import Sys, h3
from captura import pulse, optimize_G, ideal_G
NC = 3; ke, ki, Gmax, dt = 1e-2, 1e-3, 0.012, 2.0; Tp = 100.0; eps = 0.02; gam, gph = 2e-4, 4e-4
S = Sys(h3(), [.2, .14, 0], slots=[1, 0], kappa=ki, gam=gam, gph=gph, chi=1.0)
g1 = abs(S.B@S.Lam@S.D[0]); E1 = S.E[0]; DI = 1.0 - E1
C = sum((S.V[:, c]@S.Lam@S.D[0])**2/(1 + S.w[c] - E1) for c in range(3))
t = np.arange(-9*Tp, 9*Tp, dt); u = pulse('gauss', Tp, t, 0.0); Gi, _ = ideal_G(u, t, ke+ki); G, eta_eff = optimize_G(u, t, ke, ki, Gmax, G0=Gi)
u = u*np.sqrt(1-eps); gs = u**2/np.clip(1-np.cumsum(u**2)*dt, 1e-6, None)
d = 4
def kr(A, B, Cc): return qt.Qobj(np.kron(np.kron(A, B), Cc), dims=[[d, NC, 2], [d, NC, 2]])
Ie, Im, Is = np.eye(d), np.eye(NC), np.eye(2); a_m = np.diag(np.sqrt(np.arange(1, NC)), 1); s_s = np.array([[0, 1], [0, 0.]])
b = kr(Ie, a_m, Is); asrc = kr(Ie, Im, s_s); Nexc = kr(np.diag([0, 1, 1, 1.]), Im, Is)
Lam_blk = np.zeros((d, d)); Lam_blk[1:, 1:] = S.Lam; He = np.zeros((d, d)); He[1:, 1:] = S.h
H0 = kr(He, Im, Is) + b.dag()*b + asrc.dag()*asrc + kr(Lam_blk, a_m + a_m.T, Is)
Hl = np.zeros((d, d)); Hl[1:, 0] = 1/np.sqrt(3); Hl = Hl + Hl.T; Hlaser = kr(Hl, Im, Is)
D1 = np.zeros((d, d)); D1[1:, 1:] = np.outer(S.D[0], S.D[0]); PD1 = kr(D1, Im, Is)
psi0 = qt.tensor(qt.basis(d, 0), qt.basis(NC, 0), qt.basis(2, 1)); rho0 = psi0*psi0.dag()
def evalcfg(cfg):
    scale, delay, doff = cfg
    Om = scale*np.interp(t-delay, t, G, left=0, right=0)*DI/g1; wl = E1 - 1.0 - (-Om**2/DI + C) + doff
    f_om = lambda tt, args=None: np.interp(tt, t, Om); f_wl = lambda tt, args=None: np.interp(tt, t, wl)
    f_gs = lambda tt, args=None: np.sqrt(np.interp(tt, t, gs)); f_hc = lambda tt, args=None: -0.5*np.sqrt(np.interp(tt, t, gs)*ke)
    Hc = b.dag()*asrc - asrc.dag()*b
    H = qt.QobjEvo([H0, [Hlaser, f_om], [Nexc, lambda tt, args=None: -f_wl(tt)], [1j*Hc, f_hc]])
    radiative = np.zeros((d, d)); radiative[0, 1:] = np.sqrt(2*gam/3)
    cs = [qt.QobjEvo([[asrc, f_gs], np.sqrt(ke)*b]), np.sqrt(ki)*b, kr(radiative, Im, Is), np.sqrt(gph)*Nexc]
    r = qt.mesolve(H, rho0, np.array([t[0], t[-1]]), cs, options=dict(atol=1e-7, rtol=1e-7, nsteps=10**7)).states[-1]
    return cfg, float(np.real(qt.expect(PD1, r)))/(1-eps)
if __name__ == '__main__':
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    grid = list(itertools.product((0.9, 1.0, 1.1), (-20.0, 0.0, 20.0), (-0.004, 0.0, 0.004)))
    print('eta efectivo=%.3f ; rejilla completa (%d puntos), 3 procesos'%(eta_eff, len(grid)), flush=True); best = (0, None); t0 = time.time()
    with Pool(nproc) as p:
        for cfg, e in p.imap_unordered(evalcfg, grid):
            print('  escala=%.2f retraso=%+5.0f desint=%+.3f  eta=%.3f  (%.0fs)'%(*cfg, e, time.time()-t0), flush=True)
            if e > best[0]: best = (e, cfg)
    print('MEJOR en rejilla: eta=%.3f en %s'%best, flush=True)
