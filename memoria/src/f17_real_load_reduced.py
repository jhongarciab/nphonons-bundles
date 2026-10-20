"""F15: carga real del fonón en el modelo reducido {G,B,D1,D2} (F9/F13): fuente en cascada (paquete exponencial creciente, óptimo) vs carga ideal.
Pulsos y brechas de F13 arranque 1, f_b=100 GHz (tasas F11), χ=1. κ_in = fracción del κ total acoplada al puerto de carga; el resto es pérdida intrínseca."""
import os, sys, json, numpy as np, qutip as qt
os.environ.setdefault("F9_NC", "2"); os.environ.setdefault("F9_TOL", "1e-6")
import f9_dual_rail_n3 as f9
from f13_reoptimize import rates
NC = f9.NC; OPT = f9.OPT
P1 = (0.197, 170, -0.02, 0.29, 150, -0.025); G1, G2 = 0.35, 0.6
dims = [4, NC, 2, 2, 2, 2]   # exc, modo, R, A1, A2, fuente
def op(x, pos): return qt.tensor(*[x if k == pos else qt.qeye(d) for k, d in enumerate(dims)])
def swap(k):
    D = int(np.prod(dims)); U = np.zeros((D, D))
    for flat in range(D):
        idx = np.unravel_index(flat, dims); n, a = idx[1], idx[2 + k]; new = list(idx)
        if (n, a) == (1, 0): new[1], new[2 + k] = 0, 1
        elif (n, a) == (0, 1): new[1], new[2 + k] = 1, 0
        U[np.ravel_multi_index(new, dims), flat] = 1
    return qt.Qobj(U, dims=[dims, dims])
def lift(x):  # operador de dims antiguas [4,NC,2,2,2] a dims con fuente
    return qt.tensor(x, qt.qeye(2))
CAP = []
def run(T, mode, kappa=1e-3, kin_frac=1.0, gph=4e-4, chi=1.0, gam=2e-4, tail=0.02, fb=100, tau12=100.0, upto=None, tw=0.0, readout="ideal", kout_frac=10.0, eps_cap=0.02):
    gc, gc3, gs = rates(fb, G1, G2)
    E, c0 = f9.model((-G1, -G2), kappa, gam, gph, gc, gs, chi=chi, gc3=gc3)
    c0 = [lift(x) for x in c0]
    om1, d1, s1, om2, d2, s2 = P1
    H1, Hd = f9.hamil(E, 2, s1, om1); H2, _ = f9.hamil(E, 3, s2, om2)
    H1, H2, Hd = lift(H1), lift(H2), lift(Hd)
    H1p, H2p = H1 + om1*Hd, H2 + om2*Hd
    b = lift(f9.bX); a_s = op(qt.destroy(2), 5)
    kin = kappa*kin_frac
    def st(r, H, t, cc=c0):
        if t <= 0: return r
        o = qt.mesolve(H, r, [0, t], cc, options=OPT).states[-1]; CAP.append(o); return o
    R0, R1 = qt.basis(2, 0), qt.basis(2, 1)
    psi = qt.tensor(f9.ket4[0], qt.basis(NC, 0), (R0 + R1).unit(), qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0))
    r = psi*psi.dag()
    X = qt.Qobj(np.eye(NC)[[1, 0] + list(range(2, NC))]); sx = qt.Qobj(np.array([[0, 1], [1, 0]]))
    info = []
    def load(r, k, Hi):
        Pr = (R0*R0.dag() if k == 0 else R1*R1.dag()); Qr = qt.qeye(2) - Pr
        if mode == "ideal":
            U = op(X, 1)*op(Pr, 2) + op(qt.qeye(NC), 1)*op(Qr, 2); return U*r*U.dag()
        U = op(sx, 5)*op(Pr, 2) + op(qt.qeye(2), 5)*op(Qr, 2); r = U*r*U.dag()
        TL = 4.0/kin; tend = TL + np.log(1 - tail)/kin   # e^{kin(t-TL)} = 1-tail
        g = lambda t, args=None: np.sqrt(kin*np.exp(kin*(t - TL))/(1 - np.exp(kin*(t - TL))))
        Hc = qt.QobjEvo([[-(1j/2)*np.sqrt(kin)*(b.dag()*a_s - a_s.dag()*b), g]])
        Lt = qt.QobjEvo([[a_s, g], np.sqrt(kin)*b])
        ci = [x for x in c0 if True]
        ci[0] = np.sqrt(max(kappa - kin, 0.0))*b if kappa > kin else 0*b   # c0[0]=κ b original; sustituir por pérdida intrínseca
        res = qt.mesolve(qt.QobjEvo(Hi + a_s.dag()*a_s) + Hc, r, [0, tend], [Lt] + ci, options=OPT).states[-1]
        pl = float(np.real(qt.expect(op(R0*R0.dag() if k == 0 else R1*R1.dag(), 2)*op(qt.fock_dm(NC, 1), 1), res)))*2
        info.append(pl)
        return qt.tensor(res.ptrace([0, 1, 2, 3, 4]), qt.fock_dm(2, 0))
    _l = load
    def load(r, k, Hi):
        o = _l(r, k, Hi); CAP.append(o); return o
    r = load(r, 0, H1)
    diag = {}
    PD1 = lift(f9.op(f9.P[2], 0)); PD2 = lift(f9.op(f9.P[3], 0)); nb = b.dag()*b
    PR0, PR1 = op(R0*R0.dag(), 2), op(R1*R1.dag(), 2)
    ex = lambda A, rr: float(np.real(qt.expect(A, rr)))
    diag["pb_R0_after_load1"] = 2*ex(PR0*nb, r)
    r = st(r, H1p, d1); diag["pD1_R0_after_write1"] = 2*ex(PR0*PD1, r); diag["pb_R0_after_write1"] = 2*ex(PR0*nb, r)
    r = st(r, H1, tau12)
    r = load(r, 1, H2)
    diag["pD1_R0_after_load2"] = 2*ex(PR0*PD1, r); diag["pb_R1_after_load2"] = 2*ex(PR1*nb, r); diag["pb_R0_after_load2"] = 2*ex(PR0*nb, r); diag["pD2_R1_after_load2"] = 2*ex(PR1*PD2, r)
    if upto == "load2": return diag
    r = st(r, H2p, d2); diag["pD2_R1_after_write2"] = 2*ex(PR1*PD2, r); diag["pD1_R0_after_write2"] = 2*ex(PR0*PD1, r)
    r = st(r, H2, T)
    kout = kappa*kout_frac; tE = 4.0/kout
    def emit(r, k, Hi):
        # modo -> guía -> cavidad sumidero A_k (captura óptima de un paquete exponencial decreciente, regularizada con eps_cap)
        ak = op(qt.destroy(2), 2 + k)
        gc = lambda tt, args=None: np.sqrt(kout*np.exp(-kout*tt)/(eps_cap + 1 - np.exp(-kout*tt)))
        Hc = qt.QobjEvo([[-(1j/2)*np.sqrt(kout)*(ak.dag()*b - b.dag()*ak), gc]])
        Lt = qt.QobjEvo([[ak, gc], np.sqrt(kout)*b])
        ci = list(c0[1:])
        return qt.mesolve(qt.QobjEvo(Hi + ak.dag()*ak) + Hc, r, [0, tE], [Lt] + ci, options=OPT).states[-1]
    if readout == "real":
        r = st(r, H2p, d2); r = emit(r, 2, H2)
        r = st(r, H1, tw)
        r = st(r, H1p, d1); r = emit(r, 1, H1)
    else:
        r = st(r, H2p, d2); U2 = swap(2); r = U2*r*U2.dag()
        r = st(r, H1, tw)
        r = st(r, H1p, d1); U1 = swap(1); r = U1*r*U1.dag()
    rho = r.ptrace([2, 3, 4])
    ket = lambda R, a1, a2: qt.tensor(qt.basis(2, R), qt.basis(2, a1), qt.basis(2, a2))
    k0, k1 = ket(0, 1, 0), ket(1, 0, 1)
    a = float(np.real(qt.expect(k0*k0.dag(), rho))); bq = float(np.real(qt.expect(k1*k1.dag(), rho))); cc = abs(complex(k0.dag()*rho*k1))
    s = sum(float(np.real(qt.expect(ket(R, x, 1-x)*ket(R, x, 1-x).dag(), rho))) for R in (0, 1) for x in (0, 1))
    F = 0.5*(a + bq) + cc
    ev = np.linalg.eigvalsh(0.5*(rho.full()+rho.full().conj().T)); out = dict(mineig=float(ev.min()), tr=float(np.real(rho.tr())), a=a, b=bq, c=cc, mode=mode, kin_frac=kin_frac, T=T, F=F, s=s, F_h=F/s if s else None, pload=info, tail=tail, diag=diag, readout=readout, kout_frac=kout_frac)
    if readout == "real":
        # eficiencia de captura de un fonón en la cavidad por la cascada de salida
        a_ = qt.tensor(qt.destroy(NC), qt.qeye(2)); sk = qt.tensor(qt.qeye(NC), qt.destroy(2))
        gcs = lambda tt, args=None: np.sqrt(kout*np.exp(-kout*tt)/(eps_cap + 1 - np.exp(-kout*tt)))
        Hc2 = qt.QobjEvo([[-(1j/2)*np.sqrt(kout)*(sk.dag()*a_ - a_.dag()*sk), gcs]])
        L2 = qt.QobjEvo([[sk, gcs], np.sqrt(kout)*a_])
        p1_ = qt.tensor(qt.basis(NC, 1), qt.basis(2, 0))
        res2 = qt.mesolve(qt.QobjEvo(1.0*(a_.dag()*a_ + sk.dag()*sk)) + Hc2, p1_*p1_.dag(), [0, tE], [L2], e_ops=[sk.dag()*sk], options=OPT)
        etaR = float(res2.expect[0][-1]); out["eta_R"] = etaR
        eL = info if info else [1.0, 1.0]
        tL2 = 4.0/(kappa*kin_frac) if mode == "real" else 0.0
        p1p = eL[0]*etaR*np.exp(-kappa*(tL2 + T + tE)); p2p = eL[-1]*etaR*np.exp(-kappa*T)
        out["pas_F"] = 0.5*(p1p/2 + p2p/2) + 0.5*np.sqrt(p1p*p2p); out["pas_s"] = 0.5*(p1p + p2p); out["pas_Fh"] = out["pas_F"]/out["pas_s"]
    return out
if __name__ == "__main__":
    mode, frac, T = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    tau = float(sys.argv[4]) if len(sys.argv) > 4 else 100.0
    chi = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    tw = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    ro = sys.argv[7] if len(sys.argv) > 7 else "ideal"
    kof = float(sys.argv[8]) if len(sys.argv) > 8 else 10.0
    if mode == "diag":
        print("DIAG f17", json.dumps(run(T, "real", kin_frac=frac, tau12=tau, chi=chi, upto="load2")), flush=True); sys.exit()
    r = run(T, mode, kin_frac=frac, tau12=tau, chi=chi, tw=tw, readout=ro, kout_frac=kof); r.update(tau12=tau, chi=chi, tw=tw); print(json.dumps(r), flush=True)
