"""Free storage dynamics after writing; run on pcjhon, with no optical drive.

Populations are prepared in each dark eigenstate separately. Coherence is
prepared in (D1+D2)/sqrt(2). This isolates storage from input/output errors.
"""
from pathlib import Path
import numpy as np
import qutip as qt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "memory_figs" / "dark_storage_decay.npz"
N = 3
e = [qt.basis(4, i) for i in range(4)]
v = {1: np.ones(3)/np.sqrt(3), 2: np.array([0,1,-1])/np.sqrt(2),
     3: np.array([2,-1,-1])/np.sqrt(6)}
I = qt.qeye(N)
b = qt.tensor(qt.qeye(4), qt.destroy(N))
P = [qt.tensor(x*x.dag(), I) for x in e]
nd = []
for i in range(3):
    m = sum(v[a][i]*v[c][i]*e[a]*e[c].dag()
            for a in (1,2,3) for c in (1,2,3))
    nd.append(qt.tensor(m, I))

lam = (0.20, 0.14, 0.0)
H = b.dag()*b - 0.35*P[2] - 0.60*P[3]
for x, ni in zip(lam, nd):
    H += x*ni*(b+b.dag())

kappa, gamma, gphi = 0.001, 2e-4, 4e-4
rates = (2.7324407633669805e-5, 3.936245631649005e-4,
         2.5578944410303146e-6)  # F13, 100 GHz: B->D1,B->D2,D1->D2
cbase = [np.sqrt(kappa)*b,
         np.sqrt(2*gamma)*qt.tensor(e[0]*e[1].dag(), I)]
for (a,c), rate in zip(((1,2),(1,3),(2,3)), rates):
    cbase.append(np.sqrt(rate)*qt.tensor(e[c]*e[a].dag(), I))

t = np.unique(np.r_[np.linspace(0,100,101), np.linspace(110,8000,160)])
opts = {"rtol": 1e-8, "atol": 1e-10, "nsteps": 1000000}
ket = lambda j: qt.tensor(e[j],qt.basis(N,0))
rho_d1 = qt.ket2dm(ket(2))
rho_d2 = qt.ket2dm(ket(3))
rho_coh = qt.ket2dm((ket(2)+ket(3)).unit())
cross = qt.tensor(e[3]*e[2].dag(),I)
results = {}
for chi in (0.0, 0.9, 1.0):
    c = list(cbase)
    c += [np.sqrt(gphi*(1-chi))*ni for ni in nd]
    if chi:
        c.append(np.sqrt(gphi*chi)*(P[1]+P[2]+P[3]))
    for label, rho in (("d1",rho_d1),("d2",rho_d2),("coh",rho_coh)):
        sol = qt.mesolve(H,rho,t,c,e_ops=[],options=opts)
        results[f"{label}_chi{chi:g}_pD1"] = np.array(qt.expect(P[2],sol.states),float)
        results[f"{label}_chi{chi:g}_pD2"] = np.array(qt.expect(P[3],sol.states),float)
        if label == "coh":
            results[f"coh_chi{chi:g}"] = 2*np.abs(np.array(qt.expect(cross,sol.states)))
    print(f"chi={chi:g} done",flush=True)
OUT.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(OUT,t=t,**results)
print(OUT,flush=True)
