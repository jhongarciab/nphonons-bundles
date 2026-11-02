# Prototipo I1: cat mecánico (pérdida de 2 fonones + bombeo): tasas de bit-flip (G1) y parity/phase-flip (G2) vs alpha^2, nth, Kerr
import numpy as np, qutip as qt, scipy.sparse.linalg as sl
def rates(alpha2,k1=0.02,nth=0.0,K=0.0,k2=1.0,N=34):
    a=qt.destroy(N); ad=a.dag(); eps=alpha2*k2/2
    H=eps*(ad*ad+a*a)+K*ad*ad*a*a
    c=[np.sqrt(k2)*a*a, np.sqrt(k1*(1+nth))*a]
    if nth>0: c.append(np.sqrt(k1*nth)*ad)
    L=qt.liouvillian(H,c).data.as_scipy().tocsc()
    ev=sl.eigs(L,k=5,sigma=1e-9,which='LM',return_eigenvectors=False)
    r=np.sort(-ev.real); return r[1],r[2]   # r[0]~0
if __name__=="__main__":
    print("k1/k2=0.02 | alpha2 nth K/k2 | G_bitflip  G_parity | 2*k1*(1+2nth)*alpha2")
    for (al,nth,K) in [(2,0,0),(3,0,0),(4,0,0),(2,0.1,0),(3,0.1,0),(4,0.1,0),(3,0.5,0),(3,0,0.3),(3,0,1.0),(3,0.1,0.3)]:
        g=rates(al,nth=nth,K=K)
        print(f"{al:4.1f} {nth:4.2f} {K:4.2f} | {g[0]:.3e} {g[1]:.3e} | {2*0.02*(1+2*nth)*al:.3e}",flush=True)
