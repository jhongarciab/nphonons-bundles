"""Verificación de la referencia pasiva: dinámica abierta de dos modos acústicos independientes con los mismos puertos
(carga en cascada, captura en cavidad sumidero) frente a la expresión factorizada p1, p2, F_pas del manuscrito.
Uso: python passive_two_modes.py T tau12   (ωb=1; κ=1e-3, κ_in=κ_out=10κ, tail=0.02, eps_cap=0.02; rtol 1e-9)."""
import sys, json, numpy as np, qutip as qt
OPT = {"atol": 1e-11, "rtol": 1e-9, "nsteps": 3_000_000}
kappa, kin_frac, kout_frac, tail, eps = 1e-3, 10.0, 10.0, 0.02, 0.02
kin, kout = kappa*kin_frac, kappa*kout_frac
dims = [2, 2, 2, 2, 2, 2]   # m1, m2, R, A1, A2, fuente
def op(x, pos): return qt.tensor(*[x if k == pos else qt.qeye(d) for k, d in enumerate(dims)])
m = [op(qt.destroy(2), 0), op(qt.destroy(2), 1)]; A = [op(qt.destroy(2), 3), op(qt.destroy(2), 4)]; src = op(qt.destroy(2), 5)
R0, R1 = qt.basis(2, 0), qt.basis(2, 1); PR = [op(R0*R0.dag(), 2), op(R1*R1.dag(), 2)]
sx = qt.Qobj(np.array([[0, 1], [1, 0]]))
def hold(r, t):
    if t <= 0: return r
    return qt.mesolve(0*m[0], r, [0, t], [np.sqrt(kappa)*m[0], np.sqrt(kappa)*m[1]], options=OPT).states[-1]
def load(r, k):
    """Ventana de carga k (0: modo 1, 1: modo 2): fuente excitada si R=k; el otro modo solo pierde κ."""
    Id = qt.tensor(*[qt.qeye(d) for d in dims]); U = op(sx, 5)*PR[k] + (Id - PR[k])
    r = U*r*U.dag()
    TL = 4.0/kin; tend = TL + np.log(1 - tail)/kin
    g = lambda t, args=None: np.sqrt(kin*np.exp(kin*(t - TL))/(1 - np.exp(kin*(t - TL))))
    b = m[k]; other = m[1 - k]
    Hc = qt.QobjEvo([[-(1j/2)*np.sqrt(kin)*(b.dag()*src - src.dag()*b), g]])
    Lt = qt.QobjEvo([[src, g], np.sqrt(kin)*b])
    r = qt.mesolve(Hc, r, [0, tend], [Lt, np.sqrt(kappa)*other], options=OPT).states[-1]
    pl = float(np.real(qt.expect(PR[k]*op(qt.fock_dm(2, 1), k), r)))*2
    keep = [0, 1, 2, 3, 4]
    return qt.tensor(r.ptrace(keep), qt.fock_dm(2, 0)), pl
def emit(r, k):
    tE = 4.0/kout; a = A[k]; b = m[k]; other = m[1 - k]
    gc = lambda t, args=None: np.sqrt(kout*np.exp(-kout*t)/(eps + 1 - np.exp(-kout*t)))
    Hc = qt.QobjEvo([[-(1j/2)*np.sqrt(kout)*(a.dag()*b - b.dag()*a), gc]])
    Lt = qt.QobjEvo([[a, gc], np.sqrt(kout)*b])
    return qt.mesolve(Hc, r, [0, tE], [Lt, np.sqrt(kappa)*other], options=OPT).states[-1]
def eta_R():
    a_ = qt.tensor(qt.destroy(2), qt.qeye(2)); sk = qt.tensor(qt.qeye(2), qt.destroy(2))
    gc = lambda t, args=None: np.sqrt(kout*np.exp(-kout*t)/(eps + 1 - np.exp(-kout*t)))
    Hc = qt.QobjEvo([[-(1j/2)*np.sqrt(kout)*(sk.dag()*a_ - a_.dag()*sk), gc]])
    L2 = qt.QobjEvo([[sk, gc], np.sqrt(kout)*a_])
    p = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))
    res = qt.mesolve(Hc, p*p.dag(), [0, 4.0/kout], [L2], e_ops=[sk.dag()*sk], options=OPT)
    return float(res.expect[0][-1])
def run(T, tau12):
    psi = qt.tensor(qt.basis(2, 0), qt.basis(2, 0), (R0 + R1).unit(), qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0))
    r = psi*psi.dag()
    r, pl1 = load(r, 0)
    r = hold(r, tau12)
    r, pl2 = load(r, 1)
    r = hold(r, T)
    r = emit(r, 1)
    r = emit(r, 0)
    rho = r.ptrace([2, 3, 4])
    ket = lambda R, a1, a2: qt.tensor(qt.basis(2, R), qt.basis(2, a1), qt.basis(2, a2))
    k0, k1 = ket(0, 1, 0), ket(1, 0, 1)
    a = float(np.real(qt.expect(k0*k0.dag(), rho))); bq = float(np.real(qt.expect(k1*k1.dag(), rho))); c = abs(complex(k0.dag()*rho*k1))
    s = sum(float(np.real(qt.expect(ket(R, x, 1 - x)*ket(R, x, 1 - x).dag(), rho))) for R in (0, 1) for x in (0, 1))
    F = 0.5*(a + bq) + c
    ev = np.linalg.eigvalsh(0.5*(rho.full() + rho.full().conj().T))
    etaR = eta_R(); tL2 = 4.0/kin
    p1 = pl1*etaR*np.exp(-kappa*(tL2 + T + 4.0/kout)); p2 = pl2*etaR*np.exp(-kappa*T)
    p1t = p1*np.exp(-kappa*tau12)   # incluye la espera tau12 entre las dos ventanas de carga
    Fp = lambda x, y: (x + y)/4 + np.sqrt(x*y)/2
    return dict(T=T, tau12=tau12, mineig=float(ev.min()), tr=float(np.real(rho.tr())), a=a, b=bq, c=c, F=F, s=s, F_h=F/s,
                etaL=[pl1, pl2], etaR=etaR, formula=dict(F=Fp(p1, p2), s=(p1 + p2)/2, F_h=Fp(p1, p2)/((p1 + p2)/2)),
                formula_tau=dict(F=Fp(p1t, p2), s=(p1t + p2)/2, F_h=Fp(p1t, p2)/((p1t + p2)/2)))
if __name__ == "__main__":
    print(json.dumps(run(float(sys.argv[1]), float(sys.argv[2]))), flush=True)
