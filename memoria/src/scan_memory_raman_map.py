"""Remote-only numerical Raman response of the F13 three-dot reduced model.

For each addressed dark slot, start in |G,1> and sweep two-photon detuning
and pulse duration.  This is an isolated write step, not the full I/O memory.
Run on pcjhon; output is a compressed data array for plotting on the Mac.
"""
from pathlib import Path
import numpy as np
import qutip as qt
from f13_reoptimize import rates

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "memory_figs" / "raman_write_map.npz"
OUT.parent.mkdir(exist_ok=True)

nc = 3
q4 = qt.qeye(4)
b = qt.tensor(q4, qt.destroy(nc))
ket = [qt.basis(4, i) for i in range(4)]
P = [k*k.dag() for k in ket]
v = {1:np.ones(3)/np.sqrt(3), 2:np.array([0,1,-1])/np.sqrt(2),
     3:np.array([2,-1,-1])/np.sqrt(6)}
lam = [0.20, 0.14, 0.0]
ndot = [sum(v[a][i]*v[c][i]*ket[a]*ket[c].dag() for a in v for c in v)
        for i in range(3)]
Hphon = b.dag()*b + sum(lam[i]*qt.tensor(ndot[i], qt.qeye(nc))*(b+b.dag()) for i in range(3))
Hd = qt.tensor(ket[0]*ket[1].dag()+ket[1]*ket[0].dag(), qt.qeye(nc))
E = {1:0.0, 2:-0.35, 3:-0.60}
gc1, gc2, gs = rates(100, .35, .60)
kappa, gam, gph = .001, .0002, .0004
c = [np.sqrt(kappa)*b, np.sqrt(2*gam)*qt.tensor(ket[0]*ket[1].dag(),qt.qeye(nc)),
     np.sqrt(gph)*qt.tensor(P[1]+P[2]+P[3], qt.qeye(nc))]
for hi,lo,rate in [(1,2,gc1),(1,3,gc2),(2,3,gs)]:
    c.append(np.sqrt(rate)*qt.tensor(ket[lo]*ket[hi].dag(),qt.qeye(nc)))
rho0 = qt.tensor(ket[0],qt.basis(nc,1)).proj()
deltas = np.linspace(-.11,.07,73)
times = np.linspace(0,250,126)
pop = np.empty((2,len(deltas),len(times)))
cross = np.empty_like(pop)
for j,(slot,omega) in enumerate([(2,.197),(3,.29)]):
    for i,d2 in enumerate(deltas):
        wl = E[slot]-1-d2
        H = Hphon + sum((E[k]-wl)*qt.tensor(P[k],qt.qeye(nc)) for k in (1,2,3)) + omega*Hd
        res = qt.mesolve(H,rho0,times,c,
                         e_ops=[qt.tensor(P[slot],qt.qeye(nc)),
                                qt.tensor(P[5-slot],qt.qeye(nc))],
                         options={"atol":1e-9,"rtol":1e-7,"nsteps":100000})
        pop[j,i,:] = np.real(res.expect[0])
        cross[j,i,:] = np.real(res.expect[1])
    print(f"slot {slot} complete",flush=True)
np.savez_compressed(OUT,deltas=deltas,times=times,pop=pop,cross=cross,
                    omega=np.array([.197,.29]),opt_detuning=np.array([-.02,-.025]),
                    opt_duration=np.array([170.,150.]))
print(OUT,flush=True)
