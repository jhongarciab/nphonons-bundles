"""Reevalúa con tolerancia estricta los pulsos ya optimizados de F18 (JSON en data/old_tol1e6_f18/). Uso: F9_TOL=1e-9 f18_reeval.py <S>"""
import os, sys, json
os.environ.setdefault("F9_NC", "2"); os.environ.setdefault("F9_TOL", "1e-9")
import numpy as np
from pathlib import Path
import f9_dual_rail_n3 as f9
S = float(sys.argv[1]); LAM = tuple(np.array([0.20, 0.14, 0.0])*S)
f9.hamil.__defaults__ = (LAM,); f9.LAMS = LAM
import f13_reoptimize as f13, f17_real_load_reduced as f17
ROOT = Path(__file__).resolve().parents[1]
old = json.loads((ROOT/"data"/"old_tol1e6_f18"/f"f18_lambda_S{S}.json").read_text()); cur = np.array(old["p"]); G1, G2 = 0.35, 0.60
res = dict(S=S, p=cur.tolist(), tol=os.environ["F9_TOL"], ideal_io=[])
for T in (500, 2000, 8000):
    r = f13.evalp(np.array(list(cur)+[G1, G2]), T=T); res["ideal_io"].append(dict(T=T, F=r["F"], s=r["s"], F_h=r["F_h"])); print(f"[S={S}] ideal T={T}: {r['F']:.3f} {r['s']:.3f} {r['F_h']:.3f}", flush=True)
r0 = f13.evalp(np.array(list(cur)+[G1, G2]), T=2000, chi=0.0); res["ideal_io"].append(dict(T=2000, chi=0.0, F=r0["F"], s=r0["s"], F_h=r0["F_h"]))
f17.P1 = tuple(cur)
rr = f17.run(2000.0, "real", kin_frac=10.0, readout="real", kout_frac=10.0)
res["real_io"] = dict(F=rr["F"], s=rr["s"], F_h=rr["F_h"], pas_F=rr["pas_F"], pas_s=rr["pas_s"], mineig=rr.get("mineig"))
print(f"[S={S}] real I/O T=2000: {rr['F']:.3f} {rr['s']:.3f} {rr['F_h']:.3f} mineig={rr.get('mineig')}", flush=True)
(ROOT/"data"/f"f18tol9_reeval_S{S}.json").write_text(json.dumps(res))
