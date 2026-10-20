"""F14: modelo completo de 3 puntos (8 estados electrónicos, con biexcitón/triexcitón) + carga real del fonón por cascada.
Hamiltoniano de un excitón construido directamente en la base de puntos: H1 = Σ_a E_a |a><a|, B=(1,1,1)/√3 (E=0), D1=(0,1,-1)/√2 (E=-g1), D2=(2,-1,-1)/√6 (E=-g2)
(hopping y energías de sitio = elementos de H1). Interacción de pares Δxx Σ_{i<j} n_i n_j. Fonón Σ λ_i n_i (b+b†), λ=(0.20,0.14,0). Láser Ω/√3 Σ_i(σ_i+σ_i†).
Disipación: κ b (o cascada), √(2γ/3)Σσ_i (η=1), desfase local √(γφ(1-χ)) n_i y común √(γφχ)Σ n_i, relajación por el continuo L_{lo<-hi}=Σ lo_i hi_j σ_i†σ_j (tasas de F11).
Carga: 'ideal' = volteo controlado por R del modo; 'real' = fuente en cascada (Kiilerich-Mølmer) con paquete exponencial creciente (óptimo) de duración TL=4/κ_in, hasta que quedan 2 % en la fuente.
Descarga: ideal (intercambio modo -> registro A_k). Qubit: (|0>_R|bin1>+|1>_R|bin2>)/√2; F = (a+b)/2+|c| con referencia R."""
import os, sys, json, numpy as np, qutip as qt
from f13_reoptimize import rates
NC = int(os.environ.get("F14_NC", 2)); TOL = float(os.environ.get("F14_TOL", 1e-6))
OPT = {"atol": TOL*1e-2, "rtol": TOL, "nsteps": 5_000_000}
LAMS = (0.20, 0.14, 0.0)
vB, vD1, vD2 = np.ones(3)/np.sqrt(3), np.array([0, 1, -1])/np.sqrt(2), np.array([2, -1, -1])/np.sqrt(6)

CAP = []

class M:
    def __init__(self, real, g1, g2, dxx, fb=100, kappa=1e-3, kin=None, gam=2e-4, gph=4e-4, chi=1.0):
        self.real = real; self.dims = [8, NC, 2, 2, 2] + ([2] if real else [])
        self.nd = len(self.dims); self.kappa = kappa; self.kin = kappa if kin is None else kin
        I2 = qt.qeye(2)
        sm = [qt.tensor(qt.destroy(2), I2, I2), qt.tensor(I2, qt.destroy(2), I2), qt.tensor(I2, I2, qt.destroy(2))]
        self.sm = [self.emb(x) for x in sm]; self.n = [s.dag()*s for s in self.sm]; self.N = sum(self.n)
        self.b = self.op(qt.destroy(NC), 1)
        E = {"B": 0.0, "D1": -g1, "D2": -g2}; self.E = E
        H1 = E["B"]*np.outer(vB, vB) + E["D1"]*np.outer(vD1, vD1) + E["D2"]*np.outer(vD2, vD2)
        Hd = sum(H1[i, j]*self.sm[i].dag()*self.sm[j] for i in range(3) for j in range(3))
        Hd = Hd + dxx*sum(self.n[i]*self.n[j] for i in range(3) for j in range(i+1, 3))
        self.H0 = self.b.dag()*self.b + Hd + sum(LAMS[i]*self.n[i]*(self.b + self.b.dag()) for i in range(3))
        self.Hlas = sum(s + s.dag() for s in self.sm)/np.sqrt(3)
        c = [np.sqrt(2*gam/3)*sum(self.sm)]
        c += [np.sqrt(gph*(1-chi))*x for x in self.n]
        if chi: c.append(np.sqrt(gph*chi)*self.N)
        gc, gc3, gs = rates(fb, g1, g2)
        hop = lambda lo, hi: sum(lo[i]*hi[j]*self.sm[i].dag()*self.sm[j] for i in range(3) for j in range(3))
        c += [np.sqrt(gc)*hop(vD1, vB), np.sqrt(gc3)*hop(vD2, vB), np.sqrt(gs)*hop(vD2, vD1)]
        if os.environ.get("F16_PROJ"):   # control: proyecta H y saltos al sector N<=1 (equivale al reducido; sin rigidez por dxx)
            d = np.real(np.diag(self.N.full())); import scipy.sparse as sps; Pn = qt.Qobj(sps.diags((d < 1.5).astype(float)).tocsr(), dims=self.N.dims)
            self.H0 = (Pn*self.H0*Pn).to('CSR'); self.Hlas = (Pn*self.Hlas*Pn).to('CSR'); c = [(Pn*x*Pn).to('CSR') for x in c]
        self.c_static = c
        self.rates = (gc, gc3, gs)
    def emb(self, x8):
        x8 = qt.Qobj(x8.full(), dims=[[8], [8]])
        return qt.tensor(x8, *[qt.qeye(d) for d in self.dims[1:]])
    def op(self, x, pos):
        return qt.tensor(*[x if k == pos else qt.qeye(d) for k, d in enumerate(self.dims)])
    def H(self, wL, om=0.0):
        return self.H0 - wL*self.N + om*self.Hlas
    def cops(self, with_cavity=True):
        return ([np.sqrt(self.kappa)*self.b] if with_cavity else []) + self.c_static

def swap_perm(dims, k):
    D = int(np.prod(dims)); U = np.zeros((D, D))
    for flat in range(D):
        idx = np.unravel_index(flat, dims); n, a = idx[1], idx[2 + k]; new = list(idx)
        if (n, a) == (1, 0): new[1], new[2 + k] = 0, 1
        elif (n, a) == (0, 1): new[1], new[2 + k] = 1, 0
        U[np.ravel_multi_index(new, dims), flat] = 1
    return qt.Qobj(U, dims=[dims, dims])

def run(p, g1, g2, dxx, T, real=False, fb=100, chi=1.0, kappa=1e-3, kin=None, gph=4e-4, tau12=100.0, verbose=False, upto=None):
    om1, d1, s1, om2, d2, s2 = p
    m = M(real, g1, g2, dxx, fb, kappa, kin, gph=gph, chi=chi)
    wL1, wL2 = m.E["D1"] - 1.0 - s1, m.E["D2"] - 1.0 - s2
    H1i, H2i = m.H(wL1), m.H(wL2); H1p, H2p = m.H(wL1, om1), m.H(wL2, om2)
    c = m.cops()
    def st(r, H, t, cc=c):
        if t <= 0: return r
        o = qt.mesolve(H, r, [0, t], cc, options=OPT).states[-1]; CAP.append(o); return o
    R0, R1 = qt.basis(2, 0), qt.basis(2, 1)
    gnd = qt.tensor(qt.basis(8, 0), qt.basis(NC, 0), (R0 + R1).unit(), qt.basis(2, 0), qt.basis(2, 0), *([qt.basis(2, 0)] if real else []))
    r = gnd*gnd.dag()
    X = qt.Qobj(np.eye(NC)[[1, 0] + list(range(2, NC))])
    def load_ideal(r, k):
        Pr = (R0*R0.dag() if k == 0 else R1*R1.dag()); Qr = qt.qeye(2) - Pr
        U = m.op(X, 1)*m.op(Pr, 2) + m.op(qt.qeye(NC), 1)*m.op(Qr, 2)
        return U*r*U.dag()
    pload = []
    def load_real(r, k, Hi):
        Pr = (R0*R0.dag() if k == 0 else R1*R1.dag()); Qr = qt.qeye(2) - Pr
        sx = qt.Qobj(np.array([[0, 1], [1, 0]]))
        U = m.op(sx, 5)*m.op(Pr, 2) + m.op(qt.qeye(2), 5)*m.op(Qr, 2)
        r = U*r*U.dag()
        s_op = m.op(qt.destroy(2), 5); kin = m.kin; TL = 4.0/kin
        tail = 0.02
        tend = TL + np.log(1.0 - tail)/kin
        g = lambda t, args=None: np.sqrt(kin*np.exp(kin*(t-TL))/(1-np.exp(kin*(t-TL))))   # √γ_s(t)
        Hc = qt.QobjEvo([[ -(1j/2)*np.sqrt(kin)*(m.b.dag()*s_op - s_op.dag()*m.b), g]])
        Ltot = qt.QobjEvo([[s_op, g], np.sqrt(kin)*m.b])
        ci = [np.sqrt(max(m.kappa - kin, 0.0))*m.b] + m.c_static if m.kappa > kin else list(m.c_static)
        res = qt.mesolve(qt.QobjEvo(Hi + s_op.dag()*s_op) + Hc, r, [0, tend], [Ltot] + ci, options=OPT).states[-1]
        keep = [0, 1, 2, 3, 4]
        rk = res.ptrace(keep); rn = qt.tensor(rk, qt.fock_dm(2, 0))
        return rn
    load = (lambda r, k, Hi: load_real(r, k, Hi)) if real else (lambda r, k, Hi: load_ideal(r, k))
    _l = load
    def load(r, k, Hi):
        o = _l(r, k, Hi); CAP.append(o); return o
    r = load(r, 0, H1i)
    diag = {}
    PD1 = sum(vD1[i]*vD1[j]*m.sm[i].dag()*m.sm[j] for i in range(3) for j in range(3)); PD2 = sum(vD2[i]*vD2[j]*m.sm[i].dag()*m.sm[j] for i in range(3) for j in range(3))
    nb = m.b.dag()*m.b; PR0, PR1 = m.op(R0*R0.dag(), 2), m.op(R1*R1.dag(), 2)
    ex = lambda A, rr: float(np.real(qt.expect(A, rr)))
    diag["pb_R0_after_load1"] = 2*ex(PR0*nb, r)
    if real:
        e = qt.tensor(qt.basis(8, 0), qt.basis(NC, 1)); pl = float(np.real(qt.expect(m.op(R0*R0.dag(), 2)*m.op(qt.fock_dm(NC, 1), 1), r)))*2
        pload.append(pl)
    r = st(r, H1p, d1); diag["pD1_R0_after_write1"] = 2*ex(PR0*PD1, r); diag["pb_R0_after_write1"] = 2*ex(PR0*nb, r)
    r = st(r, H1i, tau12)
    r = load(r, 1, H2i)
    diag["pD1_R0_after_load2"] = 2*ex(PR0*PD1, r); diag["pb_R1_after_load2"] = 2*ex(PR1*nb, r); diag["pb_R0_after_load2"] = 2*ex(PR0*nb, r); diag["pD2_R1_after_load2"] = 2*ex(PR1*PD2, r)
    if upto == "load2": return diag
    r = st(r, H2p, d2); diag["pD2_R1_after_write2"] = 2*ex(PR1*PD2, r); diag["pD1_R0_after_write2"] = 2*ex(PR0*PD1, r)
    r = st(r, H2i, T)
    r = st(r, H2p, d2); U2 = swap_perm(m.dims, 2); r = U2*r*U2.dag()
    r = st(r, H1p, d1); U1 = swap_perm(m.dims, 1); r = U1*r*U1.dag()
    rho = r.ptrace([2, 3, 4])
    ket = lambda R, a1, a2: qt.tensor(qt.basis(2, R), qt.basis(2, a1), qt.basis(2, a2))
    k0, k1 = ket(0, 1, 0), ket(1, 0, 1)
    a = float(np.real(qt.expect(k0*k0.dag(), rho))); b = float(np.real(qt.expect(k1*k1.dag(), rho))); cc = abs(complex(k0.dag()*rho*k1))
    s = sum(float(np.real(qt.expect(ket(R, x, 1-x)*ket(R, x, 1-x).dag(), rho))) for R in (0, 1) for x in (0, 1))
    F = 0.5*(a + b) + cc
    ev = np.linalg.eigvalsh(0.5*(rho.full()+rho.full().conj().T)); return dict(mineig=float(ev.min()), tr=float(np.real(rho.tr())), herm=float(np.abs(rho.full()-rho.full().conj().T).max()), F=F, s=s, F_h=F/s if s > 0 else float("nan"), a=a, b=b, c=cc, pload=pload, rates=m.rates, diag=diag)

if __name__ == "__main__":
    P1 = (0.197, 170, -0.02, 0.29, 150, -0.025); G1, G2 = 0.35, 0.6
    name = sys.argv[1]
    if name == "dxx":
        dxx = float(sys.argv[2]); r = run(P1, G1, G2, dxx, 2000)
        print(f"dxx={dxx:6.2f} ideal: F={r['F']:.3f} s={r['s']:.3f} F|h={r['F_h']:.3f} (a={r['a']:.3f} b={r['b']:.3f} c={r['c']:.3f})", flush=True)
    elif name == "diag":
        dxx, kin = float(sys.argv[2]), float(sys.argv[3]); print("DIAG f16", dxx, kin, json.dumps(run(P1, G1, G2, dxx, 2000, real=True, kin=kin, upto="load2")), flush=True)
    elif name == "real":
        dxx, kin = float(sys.argv[2]), float(sys.argv[3]); r = run(P1, G1, G2, dxx, 2000, real=True, kin=kin, gph=float(os.environ.get('F16_GPH', 4e-4)))
        print(f"REAL load dxx={dxx} kin={kin}: F={r['F']:.3f} s={r['s']:.3f} F|h={r['F_h']:.3f} pload={r['pload']} (a={r['a']:.3f} b={r['b']:.3f} c={r['c']:.3f}) mineig={r['mineig']:.2e} tr={r['tr']:.6f} herm={r['herm']:.1e} diag={json.dumps(r['diag'])}", flush=True)
