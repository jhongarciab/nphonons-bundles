"""F13: reoptimiza pulsos (Ω,dur,δ2 de cada ranura) y brechas (g1,g2 = |E_D-E_B|) del doble carril N=3 con las tasas de regla de oro (F11) recalculadas
para cada par de brechas. Objetivo: F a T=2000, κ=0.001, γ=2e-4, γφ=4e-4, χ=1, f_b=100 GHz. Búsqueda por patrones con paso decreciente. Uso: f13_reoptimize.py <inicio 0|1|2>"""
import sys, json, numpy as np
from multiprocessing import Pool
from pathlib import Path
from f9_dual_rail_n3 import run
HB, EV = 1.054571817e-34, 1.602176634e-19
DE, DH, RHO, C = 7.0*EV, -3.5*EV, 5370.0, 5110.0
AE = 3e-9; AH = AE/1.15
V = {"B": np.ones(3)/np.sqrt(3), "D1": np.array([0, 1, -1])/np.sqrt(2), "D2": np.array([2, -1, -1])/np.sqrt(6)}
def gam(a, b, w, d):
    q = w/C; m = V[a]*V[b]
    F = DE*np.exp(-(q*AE)**2/4) - DH*np.exp(-(q*AH)**2/4)
    Jw = F**2*w**3/(4*np.pi**2*HB*RHO*C**5); s = 0.0
    for i in range(3):
        for j in range(3):
            x = q*d; s += m[i]*m[j]*(1.0 if i == j else np.sin(x)/x)
    return 2*np.pi*Jw*s
def rates(fb, g1, g2):
    w0 = 2*np.pi*fb*1e9; d = 6e-9*(0.13/(g1/3*HB*w0/EV*1e3))**(1/3)
    return gam("B", "D1", g1*w0, d)/w0, gam("B", "D2", g2*w0, d)/w0, gam("D1", "D2", (g2-g1)*w0, d)/w0
STEPS = np.array([0.02, 40, 0.01, 0.04, 30, 0.01, 0.1, 0.1])
LOW = np.array([0.02, 20, -0.2, 0.02, 20, -0.2, 0.15, 0.3])
def evalp(x, fb=100, T=2000, chi=1.0, kappa=0.001, gph=4e-4):
    p, g1, g2 = x[:6], x[6], x[7]
    if g2 < g1 + 0.15: return -1.0
    gc, gc3, gs = rates(fb, g1, g2)
    return run(tuple(p), T, kappa=kappa, gph=gph, gc=gc, gc3=gc3, gs=gs, gaps=(-g1, -g2), chi=chi)
def obj(x):
    r = evalp(x); return -1.0 if isinstance(r, float) else r["F"]
STARTS = [np.array([0.12, 300, 0.0, 0.29, 180, -0.01, 0.5, 1.0]),
          np.array([0.18, 130, -0.02, 0.25, 180, -0.02, 0.3, 1.0]),
          np.array([0.12, 260, 0.0, 0.25, 180, -0.02, 0.5, 1.5])]
if __name__ == "__main__":
    k = int(sys.argv[1]); cur = STARTS[k].copy(); steps = STEPS.copy()
    best = obj(cur); print(f"[start {k}] F0={best:.4f} x={np.round(cur, 3).tolist()}", flush=True)
    with Pool(4) as pool:
        for it in range(40):
            cand = []
            for i in range(8):
                for sgn in (-1, 1):
                    q = cur.copy(); q[i] = max(q[i] + sgn*steps[i], LOW[i]); cand.append(q)
            sc = pool.map(obj, cand); j = int(np.argmax(sc))
            if sc[j] > best + 1e-4: best, cur = sc[j], cand[j]; print(f"it{it}: F={best:.4f} x={np.round(cur, 3).tolist()}", flush=True)
            else:
                steps = steps*0.5; print(f"it{it}: sin mejora, paso/2", flush=True)
                if steps[1] < 3: break
    Path(__file__).resolve().parents[1].joinpath("data", f"f13_reopt_start{k}.json").write_text(json.dumps(dict(start=k, x=cur.tolist(), F=best, rates_100GHz=rates(100, cur[6], cur[7]))))
