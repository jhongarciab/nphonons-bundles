"""F10: ruido de desfase común (correlacionado) vs local. chi = fracción del desfase que es común (acopla a n1+n2(+n3)).
N=2 un carril: pulso gap=-1.0 (Ω=.25, dur=180, δ2=-.02), memoria de |G,1>. N=3 doble carril: pulsos de F9. κ=0.001, γ=2e-4, Γc=0.0015 (Γs=0 y Γs=Γc)."""
import json, numpy as np
from multiprocessing import Pool
import f7_raman_memory_n2 as F7
from f9_dual_rail_n3 import run as run9
P9 = (0.12, 300, 0.0, 0.29, 180, -0.01)
def job(a):
    kind, chi, gph, T, gs = a
    if kind == "N2":
        F7.CHI = chi; F7.GPH = gph
        r = F7.protocol((-1.0, 0.25, 180, -0.02, 0.001, T, 0.0015)); return dict(kind=kind, chi=chi, gphi=gph, T=T, P=r["P"], C=r["C"], V=r["V"])
    r = run9(P9, T, gph=gph, gs=gs, chi=chi); return dict(kind=kind, chi=chi, gphi=gph, T=T, gs=gs, F=r["F"], s=r["s"], F_h=r["F_h"])
if __name__ == "__main__":
    jobs = []
    for gph in (4e-4, 2e-3):
        for chi in (0.0, 0.5, 0.9, 1.0):
            for T in (2000, 8000):
                jobs.append(("N2", chi, gph, T, None))
                for gs in (0.0, 0.0015): jobs.append(("N3", chi, gph, T, gs))
    with Pool(12) as p: rows = p.map(job, jobs)
    json.dump(rows, open("../data/f10_common_noise.json", "w"))
    for gph in (4e-4, 2e-3):
        print(f"\nγφ={gph}")
        for T in (2000, 8000):
            for chi in (0.0, 0.5, 0.9, 1.0):
                g = lambda k, gs=None: next(r for r in rows if r["kind"] == k and r["chi"] == chi and r["gphi"] == gph and r["T"] == T and r.get("gs") == gs)
                a, b, c = g("N2"), g("N3", 0.0), g("N3", 0.0015)
                print(f"T={T:5d} χ={chi:3.1f} | N=2 un carril: P={a['P']:.2f} C={a['C']:.2f} V={a['V']:.2f} | N=3 doble carril Γs=0: F={b['F']:.2f} s={b['s']:.2f} F|h={b['F_h']:.2f} | Γs=Γc: F={c['F']:.2f} F|h={c['F_h']:.2f}")
