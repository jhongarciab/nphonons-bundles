"""F18 (propuesta A5): ventaja de la memoria frente a la escala del acoplo fonónico. λ = S·(0.20,0.14,0). Pulsos (Ω,dur,δ2 por ranura) REOPTIMIZADOS para cada S
(objetivo: F a T=2000, E/S ideales, κ=1e-3, γφ=4e-4, χ=1, brechas 0.35/0.6, tasas de F11), Ω hasta 0.9. Luego evaluación con carga y lectura reales (κ_in=κ_out=10κ, T=2000)
y comparación con el pasivo. Uso: f18_lambda_scan.py <S>"""
import os, sys, json
os.environ.setdefault("F9_NC", "2"); os.environ.setdefault("F9_TOL", "1e-6")
import numpy as np
from pathlib import Path
import f9_dual_rail_n3 as f9
S = float(sys.argv[1])
LAM = tuple(np.array([0.20, 0.14, 0.0])*S)
f9.hamil.__defaults__ = (LAM,); f9.LAMS = LAM
import f13_reoptimize as f13
from multiprocessing import Pool
G1, G2 = 0.35, 0.60
vB, vD1, vD2 = np.ones(3)/np.sqrt(3), np.array([0, 1, -1])/np.sqrt(2), np.array([2, -1, -1])/np.sqrt(6)
lam = np.array(LAM)
def guess():
    p = []
    for v, g in ((vD1, G1), (vD2, G2)):
        gs = float(vB @ (lam*v)); Om = min(0.9, 0.197/S); dI = 1 + g
        tpi = np.pi*dI/(2*Om*abs(gs)); ell = float(v @ (lam*v))
        d2 = -Om**2/dI + gs**2/dI + ell**2/1.0
        p += [Om, min(3000.0, tpi), d2]
    return np.array(p)
LOW = np.array([0.02, 20, -0.3, 0.02, 20, -0.3]); HIGH = np.array([0.9, 3500, 0.3, 0.9, 3500, 0.3])
def obj(p):
    r = f13.evalp(np.array(list(p) + [G1, G2])); return -1.0 if isinstance(r, float) else r["F"]
if __name__ == "__main__":
    cur = guess()
    if os.environ.get("F18_WARM"):   # arranque en caliente desde un JSON previo (p de 6 parámetros)
        cur = np.array(json.load(open(os.environ["F18_WARM"]))["p"])
    steps = np.array([0.04, max(30, 0.1*cur[1]), 0.01, 0.04, max(30, 0.1*cur[4]), 0.01])*(0.5 if os.environ.get("F18_WARM") else 1.0)
    best = obj(cur); print(f"[S={S}] inicio F={best:.4f} p={np.round(cur, 4).tolist()}", flush=True)
    with Pool(2) as pool:
        for it in range(45):
            cand = []
            for i in range(6):
                for sg in (-1, 1):
                    q = cur.copy(); q[i] = float(np.clip(q[i] + sg*steps[i], LOW[i], HIGH[i])); cand.append(q)
            sc = pool.map(obj, cand); j = int(np.argmax(sc))
            if sc[j] > best + 1e-4: best, cur = sc[j], cand[j]; print(f"it{it}: F={best:.4f} p={np.round(cur, 4).tolist()}", flush=True)
            else:
                steps = steps*0.5; print(f"it{it}: sin mejora, paso/2", flush=True)
                if steps[0] < 0.002: break
    res = dict(S=S, lam=list(LAM), p=cur.tolist(), F_search=best)
    ev = []
    for T in (500, 2000, 8000):
        r = f13.evalp(np.array(list(cur) + [G1, G2]), T=T); ev.append(dict(T=T, F=r["F"], s=r["s"], F_h=r["F_h"], pas_F=r["pas_F"], pas_s=r["pas_s"]))
        print(f"[S={S}] ideal I/O T={T}: F={r['F']:.3f} s={r['s']:.3f} F|h={r['F_h']:.3f}", flush=True)
    r0 = f13.evalp(np.array(list(cur) + [G1, G2]), T=2000, chi=0.0); ev.append(dict(T=2000, chi=0.0, F=r0["F"], s=r0["s"], F_h=r0["F_h"]))
    res["ideal_io"] = ev
    (Path(__file__).resolve().parents[1]/"data"/f"f18_lambda_S{S}.json").write_text(json.dumps(res))
    import f17_real_load_reduced as f17
    f17.P1 = tuple(cur)
    for kin, kout in ((10.0, 10.0),):
        rr = f17.run(2000.0, "real", kin_frac=kin, readout="real", kout_frac=kout)
        res["real_io"] = dict(F=rr["F"], s=rr["s"], F_h=rr["F_h"], pas_F=rr["pas_F"], pas_s=rr["pas_s"], eta_R=rr["eta_R"], pload=rr["pload"])
        print(f"[S={S}] real I/O 10κ T=2000: F={rr['F']:.3f} s={rr['s']:.3f} F|h={rr['F_h']:.3f} | pasivo F={rr['pas_F']:.3f} s={rr['pas_s']:.3f}", flush=True)
    (Path(__file__).resolve().parents[1]/"data"/f"f18_lambda_S{S}.json").write_text(json.dumps(res))
