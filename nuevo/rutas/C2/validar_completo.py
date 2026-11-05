"""Punto 1: modelo Raman COMPLETO (exc (G,s1..s3) x modo x fuente en cascada) para un paquete gaussiano hacia D1, con el control G(t) optimizado del modelo efectivo.
Se mapea G(t) -> Om(t) = G(t) Delta_I / g1 y se compensa el corrimiento Stark con chirp ideal: wl(t) = E_D1 - w - d2*(t), d2*(t) = -Om(t)^2/Delta_I + C (C = corrimiento 2o orden estatico de Eq. (stark) calculado numericamente).
Eficiencia = poblacion de D1 al final / norma del paquete."""
import sys, time, numpy as np, qutip as qt, scipy.linalg as sl
from mem import Sys, h3, NC as NC0
from captura import pulse, optimize_G, ideal_G
NC = 3
ke, ki, Gmax, dt = 1e-2, 1e-3, 0.012, 2.0; Tp = 100.0; eps = 0.02
gam, gph = 2e-4, 4e-4
S = Sys(h3(), [.2, .14, 0], slots=[1, 0], kappa=ki, gam=gam, gph=gph, chi=1.0)
g1 = abs(S.B@S.Lam@S.D[0]); E1 = S.E[0]; DI = 1.0 + (0.0 - E1)    # Delta_I = w + (E_B - E_D) = 1 + 0.35
# corrimiento estatico (2o orden) de |D1,0>: C = sum_c |L_c1|^2/(1+E_c-E_1), c en {B,D1,D2}
Lm = S.Lam; w_all, V_all = S.w, S.V
C = sum((V_all[:, c]@Lm@S.D[0])**2/(1 + w_all[c] - E1) for c in range(3))
print('g1=%.4f Delta_I=%.3f C=%.4f'%(g1, DI, C), flush=True)
# control del modelo efectivo
t = np.arange(-9*Tp, 9*Tp, dt); u = pulse('gauss', Tp, t, 0.0); Gi, _ = ideal_G(u, t, ke+ki); G, eta_eff = optimize_G(u, t, ke, ki, Gmax, G0=Gi)
print('eta efectivo = %.3f'%eta_eff, flush=True)
u = u*np.sqrt(1-eps); cum = np.cumsum(u**2)*dt; gs = u**2/np.clip(1-cum, 1e-6, None)
# operadores: exc (4) x modo (NC) x fuente (2)
d = 4
def kr(A, B, Cc): return qt.Qobj(np.kron(np.kron(A, B), Cc), dims=[[d, NC, 2], [d, NC, 2]])
Ie, Im, Is = np.eye(d), np.eye(NC), np.eye(2)
a_m = np.diag(np.sqrt(np.arange(1, NC)), 1); s_s = np.array([[0, 1], [0, 0.]])
b = kr(Ie, a_m, Is); asrc = kr(Ie, Im, s_s); nexc = np.diag([0, 1, 1, 1.]); Nexc = kr(nexc, Im, Is)
Lam_blk = np.zeros((d, d)); Lam_blk[1:, 1:] = S.Lam
Hstat = kr(np.zeros((d, d)), Im, Is)*0
He = np.zeros((d, d)); He[1:, 1:] = S.h
H0 = kr(He, Im, Is) + b.dag()*b + asrc.dag()*asrc + kr(Lam_blk, a_m + a_m.T, Is)
Hl = np.zeros((d, d)); Hl[1:, 0] = 1/np.sqrt(3); Hl = Hl + Hl.T; Hlaser = kr(Hl, Im, Is)
def make(scale):
    Om = scale*np.interp(t, t, G)*DI/g1
    wl = E1 - 1.0 - (-Om**2/DI + C)          # chirp ideal
    f_om = lambda tt, args=None: np.interp(tt, t, Om); f_wl = lambda tt, args=None: np.interp(tt, t, wl)
    f_gs = lambda tt, args=None: np.sqrt(np.interp(tt, t, gs)); f_hc = lambda tt, args=None: -0.5*np.sqrt(np.interp(tt, t, gs)*ke)
    Hc = b.dag()*asrc - asrc.dag()*b
    H = qt.QobjEvo([H0, [Hlaser, f_om], [Nexc, lambda tt, args=None: -f_wl(tt)], [1j*Hc, f_hc]])   # -(i/2) sqrt(gs ke)(b+a_s - a_s+ b) = i*(-1/2..)*(b+a_s - a_s+ b)
    cs = [qt.QobjEvo([[asrc, f_gs], np.sqrt(ke)*b]), np.sqrt(ki)*b]
    radiative = np.zeros((d, d)); radiative[0, 1:] = np.sqrt(2*gam/3); cs.append(kr(radiative, Im, Is))
    P = np.zeros((d, d))
    for kk in range(3):
        L = np.zeros((d, d)); L[1:, 1:] = np.sqrt(gph)*np.diag([1, 1, 1.]) if False else 0; 
    # desfase comun (chi=1): operador proporcional al numero de excitones
    cs.append(np.sqrt(gph)*Nexc)
    return H, cs
psi0 = qt.tensor(qt.basis(d, 0), qt.basis(NC, 0), qt.basis(2, 1)); rho0 = psi0*psi0.dag()
D1 = np.zeros((d, d)); D1[1:, 1:] = np.outer(S.D[0], S.D[0]); PD1 = kr(D1, Im, Is)
Pcav = b.dag()*b
for scale in [float(x) for x in sys.argv[1:]] or [1.0]:
    t0 = time.time(); H, cs = make(scale)
    r = qt.mesolve(H, rho0, np.array([t[0], t[-1]]), cs, options=dict(atol=1e-7, rtol=1e-7, nsteps=10**7)).states[-1]
    pD1 = float(np.real(qt.expect(PD1, r)))/(1-eps); pL = float(np.real(qt.expect(asrc.dag()*asrc, r)))
    print('escala G x%.2f: eta completo = %.3f (D1) ; fuente residual %.3f ; <b+b>=%.4f  (%.0fs)'%(scale, pD1, pL, float(np.real(qt.expect(Pcav, r))), time.time()-t0), flush=True)
