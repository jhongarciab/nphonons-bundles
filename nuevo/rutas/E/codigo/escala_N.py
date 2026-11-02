# Escala de la fuerza del proceso |N,0> <-> |g,N> en el transmon (JC transversal, sin g_z): V_N por diagonalizacion exacta. Compara con Bin: Omega*lambda^N/sqrt(N!)
import numpy as np
from transmon_f import build, TP
from scipy.optimize import minimize_scalar
def VN(g,al,Nph,N=9):
    nq=Nph+2
    def gap(Dq):
        H,*_=build(g,al,Dq,N,nq); E,V=np.linalg.eigh(H.full()); wt=np.abs(V[Nph*N+0,:])**2+np.abs(V[0*N+Nph,:])**2
        i=np.argsort(wt)[-2:]; return abs(E[i[0]]-E[i[1]])
    c=-al*(Nph-1)/2
    r=minimize_scalar(gap,bounds=(c-0.35*abs(al),c+0.35*abs(al)),method='bounded',options={'xatol':1e-10}); return r.fun/2
for g,al in [(15.2,-138),(5,-250)]:
    print(f"g/2pi={g}, alpha/2pi={al} MHz:", end=' ')
    for Nph in (2,3,4):
        v=VN(g*TP,al*TP,Nph,N=Nph+5)/TP; print(f"V_{Nph}/2pi={v:.4g} MHz",end='; ')
    print()
