"""F9: N=3 emisores, dos ranuras oscuras D1,D2 (modelo reducido {G,B,D1,D2} x modo x R x A1 x A2). Qubit de doble carril de dos 'time-bins' de fonón:
   (|0>_R|bin1> + |1>_R|bin2>)/√2. Carga ideal controlada por R (el fonón llega a la cavidad), escritura Raman en la ranura k con el láser en la resonancia
   de dos fotones de esa ranura (selección por frecuencia, sin interruptor), espera T, lectura en orden inverso con descarga ideal del modo a los registros A2, A1.
Base de puntos (un solo excitón): B=(1,1,1)/√3, D1=(0,1,-1)/√2, D2=(2,-1,-1)/√6; energías E_B=0, E_D1=gap1, E_D2=gap2 (< 0: D por debajo de B).
Fonón: Σ λ_i n_i (b+b†) con λ=(0.20,0.14,0). Óptica: η=1 (solo B radía, √(2γ)|G><B|). Desfase local √γφ n_i. Relajación T=0 hacia abajo con Γc entre niveles.
Métrica: F = max_φ <Φ_φ|ρ_{R,A1,A2}|Φ_φ> = (a+b)/2+|c| (sin postselección) y F/s con s la prob. de que A1+A2=1 (heraldo por conteo de fonones).
Pasivo: dos modos de cavidad independientes con el mismo κ y los mismos tiempos de permanencia."""
import sys, json, itertools, numpy as np, qutip as qt
from multiprocessing import Pool
from pathlib import Path
import os
NC = int(os.environ.get("F9_NC", 3))
_tol = float(os.environ.get("F9_TOL", 1e-8))
OPT = {"atol": _tol*1e-2, "rtol": _tol, "nsteps": 3_000_000}
LAMS = (0.20, 0.14, 0.0)  # g(B-D1)=0.057, g(B-D2)=0.061
def dims(): return [4, NC, 2, 2, 2]
def op(x, pos):
    ops = [qt.qeye(d) for d in dims()]; ops[pos] = x; return qt.tensor(*ops)
ket4 = [qt.basis(4, i) for i in range(4)]
vB, vD1, vD2 = np.ones(3)/np.sqrt(3), np.array([0, 1, -1])/np.sqrt(2), np.array([2, -1, -1])/np.sqrt(6)
V = {1: vB, 2: vD1, 3: vD2}
def ndot(i):
    M = sum(V[a][i]*V[bb][i]*ket4[a]*ket4[bb].dag() for a in V for bb in V); return M
P = {k: ket4[k]*ket4[k].dag() for k in range(4)}
b = qt.destroy(NC); bX = op(b, 1)
R0 = qt.basis(2, 0); R1 = qt.basis(2, 1)

def model(gaps, kappa, gam, gph, gc, gs=None, glam=LAMS, chi=0.0, gc3=None, cov=None):
    gs = gc if gs is None else gs
    gc3 = gc if gc3 is None else gc3
    E = {1: 0.0, 2: gaps[0], 3: gaps[1]}
    c = [np.sqrt(kappa)*bX, np.sqrt(2*gam)*op(ket4[0]*ket4[1].dag(), 0)]
    if cov is None:
        for i in range(3): c.append(np.sqrt(gph*(1-chi))*op(ndot(i), 0))
        if chi: c.append(np.sqrt(gph*chi)*op(P[1] + P[2] + P[3], 0))
    else:
        cov = np.asarray(cov, dtype=float)
        if cov.shape != (3, 3) or not np.allclose(cov, cov.T):
            raise ValueError('cov must be a symmetric 3x3 matrix')
        vals, vecs = np.linalg.eigh(cov)
        if vals.min() < -1e-10: raise ValueError('cov is not positive semidefinite')
        for k, value in enumerate(vals):
            if value > 1e-12:
                L = sum(float(vecs[i, k])*op(ndot(i), 0) for i in range(3))
                c.append(np.sqrt(gph*value)*L)
    for hi, lo, rate in ((1, 2, gc), (1, 3, gc3), (2, 3, gs)):
        if rate: c.append(np.sqrt(rate)*op(ket4[lo]*ket4[hi].dag(), 0))
    return E, c

def hamil(E, slot, d2, om, glam=LAMS):
    """marco del láser para la ranura 'slot' (2 o 3): δ_slot = ω_b + d2."""
    wL = E[slot] - 1.0 - d2
    H = bX.dag()*bX
    for k in (1, 2, 3): H += (E[k] - wL)*op(P[k], 0)
    for i in range(3): H += glam[i]*op(ndot(i), 0)*(bX + bX.dag())
    Hd = op(ket4[0]*ket4[1].dag() + ket4[1]*ket4[0].dag(), 0)
    return H, Hd

def loadU(r):
    """si R=r: |0>_modo <-> |1>_modo (carga ideal de un fonón en la cavidad vacía); identidad en lo demás."""
    X = qt.Qobj(np.eye(NC)[[1, 0] + list(range(2, NC))])
    Pr = R0*R0.dag() if r == 0 else R1*R1.dag(); Qr = qt.qeye(2) - Pr
    return op(X, 1)*op(Pr, 2) + op(qt.qeye(NC), 1)*op(Qr, 2)

def swap_modo_registro(k):
    """construye el unitario por permutación en la base producto completa"""
    D = int(np.prod(dims())); U = np.zeros((D, D))
    for flat in range(D):
        idx = np.unravel_index(flat, dims())
        x, n, r, a1, a2 = idx; a = (a1, a2)[k-1]
        new = list(idx)
        if (n, a) == (1, 0): new[1], new[2 + k] = 0, 1
        elif (n, a) == (0, 1): new[1], new[2 + k] = 1, 0
        U[np.ravel_multi_index(new, dims()), flat] = 1
    return qt.Qobj(U, dims=[dims(), dims()])

def run(p, T, kappa=0.001, gam=2e-4, gph=4e-4, gc=0.0015, gaps=(-0.5, -1.0), tau12=100.0, gs=None, chi=0.0, gc3=None, cov=None):
    om1, d1, s1, om2, d2, s2 = p
    E, c = model(gaps, kappa, gam, gph, gc, gs, chi=chi, gc3=gc3, cov=cov)
    H1, Hd = hamil(E, 2, s1, om1); H2, _ = hamil(E, 3, s2, om2)
    Hp1, Hp2 = H1 + om1*Hd, H2 + om2*Hd
    st = lambda r, H, t: qt.mesolve(H, r, [0, t], c, options=OPT).states[-1] if t > 0 else r
    psi = qt.tensor(ket4[0], qt.basis(NC, 0), (R0 + R1).unit(), qt.basis(2, 0), qt.basis(2, 0))
    r = psi*psi.dag()
    r = loadU(0)*r*loadU(0).dag(); r = st(r, Hp1, d1); r = st(r, H1, tau12)
    r = loadU(1)*r*loadU(1).dag(); r = st(r, Hp2, d2); r = st(r, H2, T)
    r = st(r, Hp2, d2); U2 = swap_modo_registro(2); r = U2*r*U2.dag()
    r = st(r, Hp1, d1); U1 = swap_modo_registro(1); r = U1*r*U1.dag()
    rho = r.ptrace([2, 3, 4])   # R, A1, A2
    e = lambda R, a1, a2: qt.tensor(qt.basis(2, R), qt.basis(2, a1), qt.basis(2, a2))
    k0, k1 = e(0, 1, 0), e(1, 0, 1)
    a = float(np.real(qt.expect(k0*k0.dag(), rho))); bq = float(np.real(qt.expect(k1*k1.dag(), rho))); cc = abs(complex(k0.dag()*rho*k1))
    s = sum(float(np.real(qt.expect(e(R, x, 1-x)*e(R, x, 1-x).dag(), rho))) for R in (0, 1) for x in (0, 1))
    F = 0.5*(a + bq) + cc
    # pasivo (dos modos): tiempos de permanencia
    t_in1, t_in2 = 0.0, d1 + tau12
    t_out2 = t_in2 + d2 + T + d2; t_out1 = t_out2 + d1
    p1, p2 = np.exp(-kappa*(t_out1 - t_in1)), np.exp(-kappa*(t_out2 - t_in2))
    Fp = 0.5*(p1/2 + p2/2) + 0.5*np.sqrt(p1*p2)   # sin postselección: poblaciones (p/2) y coherencia √(p1p2)/2
    return dict(T=T, F=F, s=s, F_h=F/s if s > 0 else np.nan, a=a, b=bq, c=cc, pas_F=Fp, pas_s=0.5*(p1 + p2),
                pas_Fh=Fp/(0.5*(p1 + p2)), params=list(map(float, p)))

def obj(p):
    return run(p, 500)["F"]

if __name__ == "__main__":
    p0 = (0.12, 260, 0.0, 0.25, 180, -0.02)
    steps = (0.02, 40, 0.01, 0.04, 30, 0.01)
    cur = list(p0); best = obj(cur); print("inicio F=", round(best, 4), flush=True)
    for it in range(14):
        cand = []
        for i in range(6):
            for sgn in (-1, 1):
                q = list(cur); q[i] = max(q[i] + sgn*steps[i], 1e-3 if i != 2 and i != 5 else -1); cand.append(q)
        with Pool(12) as pool: sc = pool.map(obj, cand)
        j = int(np.argmax(sc))
        if sc[j] > best + 1e-4: best, cur = sc[j], cand[j]; print(f"it{it}: F={best:.4f} p={np.round(cur, 3).tolist()}", flush=True)
        else: steps = tuple(s*0.5 for s in steps); print(f"it{it}: sin mejora, paso/2", flush=True)
    res = []
    for T in (0, 500, 2000, 8000):
        r = run(cur, T); res.append(r)
        print(f"T={T:5d}  F={r['F']:.3f} s={r['s']:.3f} F|herald={r['F_h']:.3f} | pasivo(2 modos) F={r['pas_F']:.3f} s={r['pas_s']:.3f} F|herald={r['pas_Fh']:.3f}", flush=True)
    for gcv in (0.0, 0.005):
        r = run(cur, 2000, gc=gcv); print(f"Γc={gcv}: T=2000 F={r['F']:.3f} s={r['s']:.3f} F|h={r['F_h']:.3f}", flush=True)
    for gp in (1e-4, 2e-3):
        r = run(cur, 2000, gph=gp); print(f"γφ={gp}: T=2000 F={r['F']:.3f} s={r['s']:.3f} F|h={r['F_h']:.3f}", flush=True)
    Path(__file__).resolve().parents[1].joinpath("data", "f9_dual_rail_n3.json").write_text(json.dumps(dict(params=cur, results=res)))
