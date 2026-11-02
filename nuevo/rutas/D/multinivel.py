"""Correccion por multinivel: transmon de L niveles (acoplo longitudinal k*lam por nivel k, Potts Eq.2) vs qubit de 2 niveles, MISMA maestra
(marco de laboratorio, salto local sqrt(kap) b) y misma optimizacion de la desintonia. Se reporta cociente J_L/J_2 y poblacion de niveles >=2."""
import numpy as np, sys
sys.path.insert(0,'/home/user/nphonons-bundles/nuevo/bundles/src'); import bundles_core as bc
import qutip as qt
from scipy.optimize import minimize_scalar
def flux(L, N, Delta, alpha, lam, Om, kap, gam, gph):
    a = qt.tensor(qt.destroy(L), qt.qeye(N)); b = qt.tensor(qt.qeye(L), qt.destroy(N)); nq = a.dag()*a
    H = Delta*nq + 0.5*alpha*nq*(nq-1) + b.dag()*b + lam*nq*(b+b.dag()) + Om*(a+a.dag())
    c = [np.sqrt(kap)*b, np.sqrt(gam)*a] + ([np.sqrt(gph)*nq] if gph>0 else [])
    rho = qt.steadystate(H, c)
    return float(np.real(qt.expect(b.dag()*b, rho))*kap), float(np.real(qt.expect(nq*(nq-1)/2*(nq>=2) if False else (nq>=2)*1.0 if False else nq, rho)))
def best(L, N, alpha, lam, Om, kap, gam, gph, n=2, w=4e-3, npt=41):
    D0 = bc.delta_res(n, lam, Om)
    f = lambda D: flux(L, N, D, alpha, lam, Om, kap, gam, gph)[0]
    Ds = D0 + np.linspace(-w, w, npt); Js = [f(D) for D in Ds]; i = int(np.argmax(Js))
    r = minimize_scalar(lambda D: -f(D), bounds=(Ds[max(i-1,0)], Ds[min(i+1,npt-1)]), method='bounded', options={'xatol':1e-7})
    return -r.fun, r.x, D0
if __name__ == '__main__':
    cases = {
     'A Potts, Om=0.03, kap=1e-4 (alpha/wb=-0.089)': dict(alpha=-388/4347, lam=11.9/4347, Om=0.03, kap=1e-4, gam=0.2397/4347, gph=1.114/4347, w=2e-3),
     'C transmon@100MHz alpha=-2.5wb, Om=0.15, kap=6.3e-3': dict(alpha=-2.5, lam=0.12, Om=0.15, kap=6.3e-3, gam=1e-3, gph=2e-3, w=8e-3),
     'C transmon@100MHz alpha=-2.0wb (resonancia 1->2 con 2 fonones extra)': dict(alpha=-2.0, lam=0.12, Om=0.15, kap=6.3e-3, gam=1e-3, gph=2e-3, w=8e-3),
     'C alpha=-10wb (anarmonicidad grande)': dict(alpha=-10., lam=0.12, Om=0.15, kap=6.3e-3, gam=1e-3, gph=2e-3, w=8e-3),
    }
    for k, p in cases.items():
        w = p.pop('w'); N = 8
        J2, D2, D0 = best(2, N, **p, w=w)
        J3, D3, _ = best(3, N, **p, w=w)
        J4, D4, _ = best(4, N, **p, w=w)
        print(f"{k}\n   J(2niv)={J2:.3e} J(3niv)={J3:.3e} J(4niv)={J4:.3e}  cociente 3/2={J3/J2:.3f} 4/2={J4/J2:.3f}  (D2={D2:.5f}, D3={D3:.5f})", flush=True)
