"""Nucleo: maestra completa (lab, marco rotante del laser), composicion de ciclos exacta y teoria analitica.
H = dL s+s + b+b + lam s+s (b+b+) + Om (s+s+) ; unidades w_b=1 ; dL = w_sigma - w_L
Resonancia n-fonon: dL = lam^2 - n + eps_n  (eps_n = corrimiento Stark/polaron de 2o orden, ver eps_n())."""
import numpy as np, qutip as qt
from math import factorial, exp, sqrt
from scipy.special import eval_genlaguerre, gammaln
def ops(N):
    b=qt.tensor(qt.qeye(2),qt.destroy(N)); s=qt.tensor(qt.destroy(2),qt.qeye(N)); return b,s
def build(dL,lam,Om,kap,gam,gph,N=12,order=0):
    b,s=ops(N); nx=s.dag()*s
    H=dL*nx+b.dag()*b+lam*nx*(b+b.dag())+Om*(s+s.dag())
    c=[np.sqrt(kap)*b,np.sqrt(gam)*s]+([np.sqrt(gph)*nx] if gph>0 else [])
    L=qt.liouvillian(H,c).full()
    Jb=(kap*qt.sprepost(b,b.dag())).full(); Jg=(gam*qt.sprepost(s,s.dag())).full()
    return H,c,L,Jb,Jg,b,s
def cycles(dL,lam,Om,kap,gam,gph,N=12,K=10):
    """Distribucion exacta P(k) de fonones emitidos entre dos saltos radiativos sucesivos del QD,
    tiempo medio de ciclo y flujo de fonones = <k>/<T>."""
    H,c,L,Jb,Jg,b,s=build(dL,lam,Om,kap,gam,gph,N)
    d=2*N; I=np.eye(d*d,dtype=complex); L0=L-Jb-Jg
    tr=np.eye(d).reshape(-1,order='F')  # vec(I) para traza
    def Gam(z): return Jg@np.linalg.solve(-(L0+z*Jb),I)
    G1=Gam(1.0)
    w,v=np.linalg.eig(G1); i=np.argmax(w.real); rs=v[:,i]; rs=rs/ (tr@rs)
    M=64; th=2*np.pi*np.arange(M)/M
    F=np.array([tr@(Gam(np.exp(1j*t))@rs) for t in th])
    P=np.fft.fft(F)/M   # P_k = (1/M) sum F(z_j) z_j^-k  -> usar conj
    P=np.array([np.mean(F*np.exp(-1j*th*k)) for k in range(K+1)]).real
    T=(tr@np.linalg.solve(-(L0+Jb),rs)).real
    mean_k=(np.arange(K+1)*P).sum()
    return P,T,mean_k/T
# ---------- teoria analitica ----------
def dmat(lam,m,k):
    """<m|D(-lam)|k> exacto (D = exp(-lam(b+ - b))), via Laguerre; real."""
    if m<k: return (-1)**(k-m)*dmat(lam,k,m)*1.0  # simetria D(-a)^+ = D(a)
    a=-lam
    lg=0.5*(gammaln(k+1)-gammaln(m+1))-lam**2/2
    return exp(lg)*a**(m-k)*eval_genlaguerre(k,m-k,lam**2)
def Omn(lam,Om,n): return Om*exp(-lam**2/2)*lam**n/sqrt(factorial(n))
def eps_n(lam,Om,n,M=40):
    """corrimiento de resonancia 2o orden (polaron + Stark dinamico), en la variable delta=dL-lam^2 = -n+eps"""
    Sg=-Om**2*sum(dmat(lam,m,0)**2/(m-n) for m in range(M) if m!=n)
    Sx=-Om**2*sum(dmat(lam,n,k)**2/k for k in range(1,M))
    return Sg-Sx
def dL_res(lam,Om,n): return lam**2-n+eps_n(lam,Om,n)

# ---------- marco de polaron + maestra "vestida" (b sin componente DC) ----------
def build_pol(delta,lam,Om,kap,gam,gph,N=12,dressed=True):
    """H_p = delta s+s + b+b + Om(s+ D + s D+), D=exp(-lam(b+-b)); delta = dL - lam^2.
    dressed=True: saltos sqrt(kap) b, sqrt(gam) s D+, sqrt(gph) s+s (sin el termino DC -lam s+s de b_lab).
    dressed=False: salto de b_lab = b - lam s+s (equivale a la maestra local en el laboratorio)."""
    b,s=ops(N); nx=s.dag()*s
    D=(-lam*(b.dag()-b)).expm()
    H=delta*nx+b.dag()*b+Om*(s.dag()*D+s*D.dag())
    bj=b if dressed else b-lam*nx
    c=[np.sqrt(kap)*bj,np.sqrt(gam)*s*D.dag()]+([np.sqrt(gph)*nx] if gph>0 else [])
    L=qt.liouvillian(H,c).full()
    Jb=(kap*qt.sprepost(bj,bj.dag())).full(); Jg=(gam*qt.sprepost(c[1]/np.sqrt(gam),c[1].dag()/np.sqrt(gam))).full()
    return H,c,L,Jb,Jg
def cycles_pol(delta,lam,Om,kap,gam,gph,N=12,K=10,dressed=True):
    H,c,L,Jb,Jg=build_pol(delta,lam,Om,kap,gam,gph,N,dressed)
    d=2*N; I=np.eye(d*d,dtype=complex); L0=L-Jb-Jg; tr=np.eye(d).reshape(-1,order='F')
    Gam=lambda z: Jg@np.linalg.solve(-(L0+z*Jb),I)
    w,v=np.linalg.eig(Gam(1.0)); rs=v[:,np.argmax(w.real)]; rs=rs/(tr@rs)
    M=64; th=2*np.pi*np.arange(M)/M
    F=np.array([tr@(Gam(np.exp(1j*t))@rs) for t in th])
    P=np.array([np.mean(F*np.exp(-1j*th*k)) for k in range(K+1)]).real
    T=(tr@np.linalg.solve(-(L0+Jb),rs)).real
    return P,T,(np.arange(K+1)*P).sum()/T

# ---------- episodios de cavidad: delimitados por el retorno a vacio (b: 1->0), modelo vestido ----------
def episodes_pol(delta,lam,Om,kap,gam,gph,N=10,K=8,fc=True):
    """Distribucion exacta P_ep(k) del numero de fonones emitidos entre dos retornos sucesivos de la cavidad al vacio
    (agrupa fonones en el tiempo: un 'bundle' = un episodio). Maestra vestida en marco de polaron."""
    b,s=ops(N); nx=s.dag()*s; D=(-lam*(b.dag()-b)).expm()
    H=delta*nx+b.dag()*b+Om*(s.dag()*D+s*D.dag())
    c=[np.sqrt(kap)*b,np.sqrt(gam)*(s*D.dag() if fc else s)]+([np.sqrt(gph)*nx] if gph>0 else [])
    L=qt.liouvillian(H,c).full()
    P1=qt.tensor(qt.qeye(2),qt.basis(N,1)*qt.basis(N,1).dag()); Pge2=qt.tensor(qt.qeye(2),qt.qeye(N))-qt.tensor(qt.qeye(2),qt.basis(N,0)*qt.basis(N,0).dag())-P1
    be=b*P1; b2=b*Pge2
    Je=(kap*qt.sprepost(be,be.dag())).full(); J2=(kap*qt.sprepost(b2,b2.dag())).full()
    d=2*N; I=np.eye(d*d,dtype=complex); L0=L-Je-J2; tr=np.eye(d).reshape(-1,order='F')
    Gam=lambda z: z*(Je@np.linalg.solve(-(L0+z*J2),I))
    w,v=np.linalg.eig(Gam(1.0)); rs=v[:,np.argmax(w.real)]; rs=rs/(tr@rs)
    M=64; th=2*np.pi*np.arange(M)/M
    F=np.array([tr@(Gam(np.exp(1j*t))@rs) for t in th])
    return np.array([np.mean(F*np.exp(-1j*th*k)) for k in range(K+1)]).real
