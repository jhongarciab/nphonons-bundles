"""Biblioteca propia de verificacion (WP2). Convencion: |g>=0,|x>=1, sigma=destroy(2)."""
import numpy as np, qutip as qt, scipy.sparse as sp, scipy.sparse.linalg as spla, scipy.linalg as sl
from scipy.optimize import minimize_scalar
from math import factorial

def sp_(q): return q.data.as_scipy().tocsc()

class Model:
    """variant: 'lab' (maestra local, marco laboratorio), 'pol_local' (marco polaron, salto b-lam n),
    'pol_dressed' (marco polaron, salto b sin DC). Conteo de fonones = saltos de phon; ciclo = salto rad."""
    def __init__(s, lam, Om, kap, gam, gph, Nb, variant):
        s.lam,s.Om,s.kap,s.gam,s.gph,s.Nb,s.variant=lam,Om,kap,gam,gph,Nb,variant
        sg=qt.tensor(qt.destroy(2),qt.qeye(Nb)); b=qt.tensor(qt.qeye(2),qt.destroy(Nb)); n=sg.dag()*sg
        s.sg,s.b,s.n=sg,b,n
        X=b+b.dag()
        if variant=='lab':
            H0=b.dag()*b+lam*n*X+Om*(sg+sg.dag()); phon=b; rad=sg
        else:
            D=qt.tensor(qt.qeye(2),qt.displace(Nb,-lam))   # T^dag sigma T = sigma D(-lam), T=exp[-lam n (b^dag-b)]
            H0=b.dag()*b-lam**2*n+Om*(sg*D+sg.dag()*D.dag())
            rad=sg*D
            phon=(b-lam*n) if variant=='pol_local' else b
        s.H0,s.Hd=H0,n
        s.cops=[np.sqrt(kap)*phon,np.sqrt(gam)*rad,np.sqrt(gph)*n]
        s.Lphon,s.Lrad=s.cops[0],s.cops[1]
        s.Nph=(s.Lphon.dag()*s.Lphon).full()
        s.L0=qt.liouvillian(H0,s.cops).full(); s.Ld=qt.liouvillian(n,[]).full()
        s.dim=2*Nb
    def L(s,dL): return s.L0+dL*s.Ld
    def steady(s,dL):
        L=s.L(dL).copy(); N=s.dim; L[0,:]=0; L[0,::N+1]=1
        r=np.zeros(N*N,complex); r[0]=1
        v=np.linalg.solve(L,r); return v.reshape((N,N),order='F')
    def flux(s,dL):
        rho=s.steady(dL); return float(np.real(np.sum(rho*s.Nph.T)))
    def Hfull(s,dL): return s.H0+dL*s.Hd

def dmat(lam,N=60):
    D=qt.displace(N,-lam).full(); return D     # D[m,j]=<m|D(-lam)|j>

def eps_n(n,lam,Om,N=60):
    d2=np.abs(dmat(lam,N))**2
    S1=sum(d2[m,0]/(n-m) for m in range(N) if m!=n)
    S2=sum(d2[n,m]/m for m in range(1,N))
    return Om**2*(S1+S2), Om**2*S1, Om**2*S2

def peak(model,guess,half=0.05,npts=41):
    xs=np.linspace(guess-half,guess+half,npts); f=[model.flux(x) for x in xs]
    i=int(np.argmax(f)); lo=xs[max(i-1,0)]; hi=xs[min(i+1,npts-1)]
    r=minimize_scalar(lambda x:-model.flux(x),bounds=(lo,hi),method='bounded',options={'xatol':1e-6})
    return r.x,-r.fun

# ---------- ciclos radiativos: distribucion exacta de conteo entre saltos radiativos consecutivos
def tracevec(N):
    t=np.zeros(N*N); t[::N+1]=1; return t

def sup(A,B):  # rho -> A rho B^dag como sparse
    return sp_(qt.sprepost(A,B.dag()))

def cycle_dist(H,cops,iphon,irad,dimH,M=48,rho0=None):
    """cops[iphon]: salto contado; cops[irad]: salto que cierra ciclo. Devuelve P(N) (N=0..M-1) en estacionario."""
    Ltot=sp_(qt.liouvillian(H,cops)); Jr=sup(cops[irad],cops[irad]); Jc=sup(cops[iphon],cops[iphon])
    Lr=(Ltot-Jr).tocsc(); tr=tracevec(dimH)
    lu=spla.splu(Lr)
    if rho0 is None:
        rho=np.zeros(dimH*dimH,complex); rho[0]=1
    else: rho=rho0
    for it in range(60):
        v=-(Jr@lu.solve(rho)); v=v/(tr@v)
        if np.linalg.norm(v-rho)<1e-13: rho=v;break
        rho=v
    G=np.zeros(M,complex)
    for j in range(M):
        z=np.exp(2j*np.pi*j/M)
        G[j]=-(tr@(Jr@spla.splu((Lr+(z-1)*Jc).tocsc()).solve(rho)))
    P=np.real(np.fft.fft(G)/M)   # P_k = (1/M) sum_j G_j z_j^{-k}
    return P
