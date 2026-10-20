"""F7: memoria fonón -> excitón oscuro por transferencia Raman (N=2, modelo reducido {G,B,D} x modo).
Sin interruptor: el láser fija la resonancia de dos fotones |G,1> <-> |D,0>; en espera (láser apagado) D queda fuera de resonancia con |B,1>
si gap=E_D-E_B != ω_b. Barre gap, Ω, duración y desintonía de dos fotones δ2 (compensa el desplazamiento de luz).
Objetivo: score=(P*C*vac)^(1/3) tras espera T, κ y Γc dados; compara con el pasivo (e^{-κT}, e^{-κT/2}, 1).
Modelo: ω_b=1; H0 = b†b + δ_B P_B + δ_D P_D + (λ/2)(P_B+P_D+X)(b+b†)  [λ1=λ, λ2=0]; X=|B><D|+h.c.; η=1: salto √(2γ)|G><B|;
desfase local √γφ n_i; Γc: relajación T=0 hacia el nivel inferior entre B y D. Sin biexcitón (Δxx→-∞)."""
import sys, json, itertools, numpy as np, qutip as qt
from multiprocessing import Pool
from pathlib import Path
NC = 3
LAM, GAM, GPH, GC = 0.14, 2e-4, 4e-4, 0.0015
CHI = 0.0  # fracción del desfase que es ruido común (acopla a n1+n2)
G, B, D = (qt.basis(3, i) for i in range(3))
I3, b = qt.qeye(3), qt.destroy(NC)
ex = lambda x: qt.tensor(x, qt.qeye(NC))
PB, PD = ex(B*B.dag()), ex(D*D.dag())
X = ex(B*D.dag() + D*B.dag()); bb = qt.tensor(I3, b)
n1, n2 = (PB + PD + X)/2, (PB + PD - X)/2
OPT = {"atol": 1e-10, "rtol": 1e-8, "nsteps": 2_000_000}
g0 = qt.tensor(G, qt.basis(NC, 0)); g1 = qt.tensor(G, qt.basis(NC, 1))

def build(gap, d2, kappa, gc=GC):
    dD = 1.0 + d2; dB = dD - gap
    H0 = bb.dag()*bb + dB*PB + dD*PD + LAM/2*(PB + PD + X)*(bb + bb.dag())
    sGB = ex(G*B.dag())
    c = [np.sqrt(kappa)*bb, np.sqrt(2*GAM)*sGB, np.sqrt(GPH*(1-CHI))*n1, np.sqrt(GPH*(1-CHI))*n2]
    if CHI: c.append(np.sqrt(GPH*CHI)*(n1 + n2))
    if gc:
        c.append(np.sqrt(gc)*(ex(B*D.dag()) if gap > 0 else ex(D*B.dag())))
    return H0, sGB + sGB.dag(), c

def protocol(a):
    gap, om, dur, d2, kappa, wait, gc = a
    H0, Hd, c = build(gap, d2, kappa, gc)
    Hp = H0 + om*Hd
    st = lambda r, H, t: qt.mesolve(H, r, [0, t], c, options=OPT).states[-1] if t > 0 else r
    def run(ket):
        r = st(ket*ket.dag(), Hp, dur); r = st(r, H0, wait); return st(r, Hp, dur)
    r1, r0, rp = run(g1), run(g0), run((g0 + g1).unit())
    P = float(np.real(qt.expect(g1*g1.dag(), r1))); V = float(np.real(qt.expect(g0*g0.dag(), r0)))
    C = float(2*abs(g0.dag()*rp*g1)); 
    wr = st(g1*g1.dag(), Hp, dur)
    return dict(gap=gap, om=om, dur=dur, d2=d2, kappa=kappa, wait=wait, gc=gc, P=P, C=C, V=V,
                score=float((max(P,0)*max(C,0)*max(V,0))**(1/3)), written=float(np.real(qt.expect(ex(D*D.dag())*qt.tensor(qt.qeye(3), qt.qeye(NC)), wr))))

if __name__ == "__main__":
    kappa, wait = float(sys.argv[1]), float(sys.argv[2])
    gaps = (-1.5, -0.5, 0.5, 1.5); oms = (0.05, 0.08, 0.12, 0.18)
    durs = (20, 40, 60, 90, 130, 180, 260); d2s = np.round(np.linspace(-0.08, 0.08, 9), 3)
    jobs = [(g, o, d, s, kappa, wait, GC) for g in gaps for o in oms for d in durs for s in d2s]
    with Pool() as p: rows = p.map(protocol, jobs, chunksize=8)
    pas = dict(P=np.exp(-kappa*wait), C=np.exp(-kappa*wait/2), V=1.0)
    print(f"κ={kappa}, espera={wait}; pasivo P={pas['P']:.3f} C={pas['C']:.3f} score={(pas['P']*pas['C'])**(1/3):.3f}")
    for g in gaps:
        best = max((r for r in rows if r["gap"] == g), key=lambda r: r["score"])
        print(f"gap={g:5.1f}: score={best['score']:.3f} P={best['P']:.3f} C={best['C']:.3f} V={best['V']:.3f} | Ω={best['om']} dur={best['dur']} δ2={best['d2']}")
    Path(__file__).resolve().parents[1].joinpath("data", f"f7_raman_k{kappa}_w{int(wait)}.json").write_text(json.dumps(rows))
