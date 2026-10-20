"""Eficiencia de la lectura |D1,0> -> |G,1> con el pulso 1 (Om=0.197, d=170, wL=-1.33), modelo completo cerrado, Ncut=4, frente a dxx.
Salida: data/biex_eff_scan.json (para la figura M8). Referencia sin biexciton: dxx=-200."""
import json, numpy as np
from pathlib import Path
exec(open("biex_width.py").read().split("# estado |D1,0>")[0])
g = np.zeros(8); g[0] = 1.0
D1 = sum(vD1[i]*(S[i].T@g) for i in range(3)); psi0 = np.kron(D1, np.eye(NC)[0]); target = np.kron(g, np.eye(NC)[1])
Om, d, wL0 = 0.197, 170, -1.33
def eff(dxx):
    H = hamil(dxx, wL0, Om); e, U = np.linalg.eigh(H); c = U.T@psi0; psi = U@(np.exp(-1j*e*d)*c); return float(abs(target@psi)**2)
xs = [round(x, 3) for x in np.arange(-14.0, -1.5 + 1e-9, 0.05)]
out = dict(xs=xs, eta=[eff(x) for x in xs], ref=eff(-200.0), Om=Om, d=d, wL=wL0, NC=NC)
Path("../data/biex_eff_scan.json").write_text(json.dumps(out)); print("ref", out["ref"], "eta(-3,-6,-9)", [eff(x) for x in (-3, -6, -9)])
