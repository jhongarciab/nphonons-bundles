"""Estadistica EXACTA de clusters (bundles) por ventana tau_c con la cuenta de clicks de la banda omega.
s_m = Tr[Phi^m rho_c], Phi = J * int_0^tau e^{L0 t} dt (m gaps consecutivos < tau tras un click).
n_{>=m}/N = s_{m-1}-s_m  =>  P_k = n_k/sum n = ... ; P2 = (s1-2 s2+s3)/(1-s1); <k>=1/(1-s1)."""
import numpy as np, qutip as qt, scipy.linalg as sl
from modelo import Full
def cluster_stats(H,c,idx,tau,mmax=6,rho=None):
    d=H.shape[0]; L=qt.liouvillian(H,c).full(); C=c[idx]
    Jm=qt.sprepost(C,C.dag()).full(); L0=L-Jm
    if rho is None: rho=qt.steadystate(H,c,method='direct')
    r=rho.full().reshape(-1,order='F'); rc=Jm@r; rate=np.real(rc[::d+1].sum()); rc=rc/rate
    n2=d*d; Z=np.zeros((2*n2,2*n2),complex); Z[:n2,:n2]=L0*tau; Z[:n2,n2:]=np.eye(n2)*tau
    M=sl.expm(Z)[:n2,n2:]               # int_0^tau e^{L0 t} dt
    Phi=Jm@M; s=[1.0]; v=rc
    tr=lambda x: np.real(x[::d+1].sum())
    for m in range(1,mmax+1): v=Phi@v; s.append(tr(v))
    s=np.array(s); nge=s[:-1]-s[1:]            # n_{>=m}/N , m=1..mmax
    nk=np.append(nge[:-1]-nge[1:],nge[-1])     # n_k/N  k=1..mmax (ultimo = >=mmax)
    nk_=nk[:mmax-1]; ncl=nge[0]
    return dict(rate=rate,s=s,Pk=nk/ncl,P2=nk[1]/ncl,Y2=2*nk[1]/1.0,meank=1/ncl)
def eff_model(gx,gz,N=6,Om=0.0,delta=None,kap_m=0,kap_q=0,gphi=0,JP=1.0,dres=0.0,ddr=0.0):
    """Efectivo de 'pocos niveles' (H_ef de James-Jerke + G(s+a^2+hc)), marco rotante a wp=2w+delta, omega=1.
    delta = -8 s/3 (resonancia g0->e0) + ddr ; resonancia e0<->g2: Dq=-4 s + dres.  Purcell: salto sqrt(kap_m*JP)*(gx)*sigma-."""
    s=gx**2; G=2*gx*gz; Dq=-4*s+dres
    if delta is None: delta=Dq+4*s/3+ddr
    a=qt.tensor(qt.qeye(2),qt.destroy(N)); sm=qt.tensor(qt.sigmam(),qt.qeye(N)); sp=sm.dag(); n=a.dag()*a
    pe=sp*sm; pg=sm*sp; sx=sp+sm
    H=-(delta/2)*n+((Dq-delta)/2)*(pe-pg)+s*((4*n/3+1)*pe-(4*n/3+1/3)*pg)-G*(sp*a*a+sm*a.dag()*a.dag())*(-1)+Om*sx
    c=[np.sqrt(kap_m)*a]
    if JP>0: c.append(np.sqrt(kap_m*JP)*gx*sm)
    if kap_q>0: c.append(np.sqrt(kap_q)*sm)
    if gphi>0: c.append(np.sqrt(gphi/2)*(pe-pg))
    return H,c
