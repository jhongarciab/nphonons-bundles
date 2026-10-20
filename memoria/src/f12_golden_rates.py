"""F12: F9 con las tasas de la regla de oro (F11) por transición: B->D1, B->D2, D1->D2 (gaps -0.5,-1.0), T=0, χ∈{0,1}. Pulsos de F9 sin reoptimizar."""
import json
from multiprocessing import Pool
from f9_dual_rail_n3 import run
P9 = (0.12, 300, 0.0, 0.29, 180, -0.01)
RATES = {50: (1.28e-5, 4.00e-4, 6.40e-6), 100: (1.27e-4, 3.75e-3, 6.33e-5), 150: (4.76e-4, 1.30e-2, 2.38e-4)}
def job(a):
    fb, chi, T, k = a; gc, gc3, gs = RATES[fb]
    r = run(P9, T, kappa=k, gc=gc, gc3=gc3, gs=gs, chi=chi); r.update(fb=fb, chi=chi, kappa=k); return r
if __name__ == "__main__":
    jobs = [(fb, chi, T, k) for fb in RATES for chi in (0.0, 1.0) for T in (500, 2000, 8000) for k in (0.001, 0.0003)]
    with Pool(12) as p: rows = p.map(job, jobs)
    json.dump(rows, open("../data/f12_golden_rates.json", "w"))
    for r in rows:
        print(f"fb={r['fb']:3d}GHz κ={r['kappa']:.4f} χ={r['chi']:.0f} T={r['T']:5d}: F={r['F']:.3f} s={r['s']:.3f} F|h={r['F_h']:.3f} | pasivo F={r['pas_F']:.3f} s={r['pas_s']:.3f}")
