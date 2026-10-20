"""F19: modelo completo de 3 puntos (8 estados electronicos, con biexciton) con carga Y lectura reales por cascada.
Reutiliza M de f16_full3dot (carga real) y la lectura por cascada de f17 (captura en la cavidad sumidero A_k, eps_cap=0.02).
Convencion igual a f17: kin = kout = 10 kappa, la perdida intrinseca se omite durante los puertos. Pulsos P1, T=2000, chi=1.
Uso: F14_TOL=1e-9 python f19_full_real_ports.py DXX [T]   (F16_PROJ=1: control proyectado al sector N<=1 = reducido)"""
import os, sys, json, numpy as np, qutip as qt
os.environ.setdefault("F14_TOL", "1e-9")
import f16_full3dot as f16
from f16_full3dot import M, OPT, NC, vD1, vD2, swap_perm

P1 = (0.197, 170, -0.02, 0.29, 150, -0.025); G1, G2 = 0.35, 0.6

def run(dxx, T=2000.0, kappa=1e-3, kin_frac=10.0, kout_frac=10.0, tau12=100.0, eps_cap=0.02, tail=0.02, chi=1.0, tw=0.0):
    om1, d1, s1, om2, d2, s2 = P1
    m = M(True, G1, G2, dxx, kappa=kappa, kin=kappa*kin_frac, chi=chi)
    b = m.b; kin = m.kin; kout = kappa*kout_frac; tE = 4.0/kout
    wL1, wL2 = m.E["D1"] - 1.0 - s1, m.E["D2"] - 1.0 - s2
    H1i, H2i = m.H(wL1), m.H(wL2); H1p, H2p = m.H(wL1, om1), m.H(wL2, om2)
    c = m.cops()
    def st(r, H, t, cc=c):
        if t <= 0: return r
        return qt.mesolve(H, r, [0, t], cc, options=OPT).states[-1]
    R0, R1 = qt.basis(2, 0), qt.basis(2, 1)
    gnd = qt.tensor(qt.basis(8, 0), qt.basis(NC, 0), (R0 + R1).unit(), qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0))
    r = gnd*gnd.dag()
    sx = qt.Qobj(np.array([[0, 1], [1, 0]])); s_op = m.op(qt.destroy(2), 5)
    def load(r, k, Hi):
        Pr = (R0*R0.dag() if k == 0 else R1*R1.dag()); Qr = qt.qeye(2) - Pr
        U = m.op(sx, 5)*m.op(Pr, 2) + m.op(qt.qeye(2), 5)*m.op(Qr, 2); r = U*r*U.dag()
        TL = 4.0/kin; tend = TL + np.log(1.0 - tail)/kin
        g = lambda t, args=None: np.sqrt(kin*np.exp(kin*(t-TL))/(1-np.exp(kin*(t-TL))))
        Hc = qt.QobjEvo([[-(1j/2)*np.sqrt(kin)*(b.dag()*s_op - s_op.dag()*b), g]])
        Lt = qt.QobjEvo([[s_op, g], np.sqrt(kin)*b])
        ci = ([np.sqrt(max(kappa - kin, 0.0))*b] if kappa > kin else []) + list(m.c_static)
        res = qt.mesolve(qt.QobjEvo(Hi + s_op.dag()*s_op) + Hc, r, [0, tend], [Lt] + ci, options=OPT).states[-1]
        return qt.tensor(res.ptrace([0, 1, 2, 3, 4]), qt.fock_dm(2, 0))
    def emit(r, k, Hi):
        ak = m.op(qt.destroy(2), 2 + k)
        gc = lambda tt, args=None: np.sqrt(kout*np.exp(-kout*tt)/(eps_cap + 1 - np.exp(-kout*tt)))
        Hc = qt.QobjEvo([[-(1j/2)*np.sqrt(kout)*(ak.dag()*b - b.dag()*ak), gc]])
        Lt = qt.QobjEvo([[ak, gc], np.sqrt(kout)*b])
        return qt.mesolve(qt.QobjEvo(Hi + ak.dag()*ak) + Hc, r, [0, tE], [Lt] + list(m.c_static), options=OPT).states[-1]
    r = load(r, 0, H1i); print("load1", flush=True)
    r = st(r, H1p, d1); r = st(r, H1i, tau12); print("write1", flush=True)
    r = load(r, 1, H2i); print("load2", flush=True)
    r = st(r, H2p, d2); r = st(r, H2i, T); print("write2+wait", flush=True)
    r = st(r, H2p, d2); r = emit(r, 2, H2i); r = st(r, H1i, tw)
    r = st(r, H1p, d1); r = emit(r, 1, H1i)
    rho = r.ptrace([2, 3, 4])
    ket = lambda R, a1, a2: qt.tensor(qt.basis(2, R), qt.basis(2, a1), qt.basis(2, a2))
    k0, k1 = ket(0, 1, 0), ket(1, 0, 1)
    a = float(np.real(qt.expect(k0*k0.dag(), rho))); bq = float(np.real(qt.expect(k1*k1.dag(), rho))); cc = abs(complex(k0.dag()*rho*k1))
    s = sum(float(np.real(qt.expect(ket(R, x, 1-x)*ket(R, x, 1-x).dag(), rho))) for R in (0, 1) for x in (0, 1))
    F = 0.5*(a + bq) + cc
    ev = np.linalg.eigvalsh(0.5*(rho.full() + rho.full().conj().T))
    return dict(dxx=dxx, T=T, proj=bool(os.environ.get("F16_PROJ")), tol=float(os.environ["F14_TOL"]), mineig=float(ev.min()), tr=float(np.real(rho.tr())),
                a=a, b=bq, c=cc, ok_c2=bool(cc**2 <= a*bq + 1e-9), F=F, s=s, F_h=F/s)

if __name__ == "__main__":
    dxx = float(sys.argv[1]); T = float(sys.argv[2]) if len(sys.argv) > 2 else 2000.0
    print(json.dumps(run(dxx, T)), flush=True)
