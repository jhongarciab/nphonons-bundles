"""Mecanismo alternativo SIN g_z: transmon multinivel (anarmonicidad alpha<0) + modo, JC transversal (RWA en g; g/w<<1).
|f,0> <-> |g,2> via |e,1> virtual: V=<g2|Heff|f0> = 4 g^2/alpha (resonancia 2 w_q + alpha = 2 w, w_q = w - alpha/2 + shifts).
Probabilidades EXACTAS de (M fonones emitidos, L=0 saltos del qubit) desde |f,0> con cadena de bloques (sin MC). Unidades: 2pi*MHz -> rad/us."""
import numpy as np, qutip as qt
from scipy.optimize import minimize_scalar
TP=2*np.pi
def build(g,alpha,Dq,N=7,nq=4):
    a=qt.tensor(qt.qeye(nq),qt.destroy(N)); 
    s=qt.tensor(qt.Qobj(np.diag(np.sqrt(np.arange(1,nq)),1)),qt.qeye(N))      # s = sum sqrt(k+1)|k><k+1|
    Ek=np.array([k*Dq+alpha*k*(k-1)/2 for k in range(nq)])
    H=qt.tensor(qt.Qobj(np.diag(Ek)),qt.qeye(N))+g*(s.dag()*a+s*a.dag())
    nqop=qt.tensor(qt.Qobj(np.diag(np.arange(nq)*1.0)),qt.qeye(N))
    return H,a,s,nqop
def resonance(g,alpha,N=7,nq=4):
    def gap(Dq):
        H,*_=build(g,alpha,Dq,N,nq); E,V=np.linalg.eigh(H.full()); wt=np.abs(V[2*N+0,:])**2+np.abs(V[0*N+2,:])**2
        i=np.argsort(wt)[-2:]; return abs(E[i[0]]-E[i[1]])
    r=minimize_scalar(gap,bounds=(-alpha/2-0.6*abs(alpha)*0.5,-alpha/2+0.3*abs(alpha)*0.5),method='bounded',options={'xatol':1e-9})
    return r.x,r.fun
def pares(g,alpha,km,kq,N=7,nq=4,Dq=None,mmax=4,init=(2,0)):
    if Dq is None: Dq,_=resonance(g,alpha,N,nq)
    H,a,s,nqop=build(g,alpha,Dq,N,nq)
    Cm=np.sqrt(km)*a; Cq=np.sqrt(kq)*s          # T1 en escalera: tasa k*kq para el nivel k
    K=qt.liouvillian(H)-0.5*(qt.spre(Cm.dag()*Cm)+qt.spost(Cm.dag()*Cm)+qt.spre(Cq.dag()*Cq)+qt.spost(Cq.dag()*Cq))
    T=qt.sprepost(Cm,Cm.dag()); K=K.full(); T=T.full(); d=H.shape[0]
    rho0=qt.ket2dm(qt.tensor(qt.basis(nq,init[0]),qt.basis(N,init[1]))).full().reshape(-1,order='F')
    src=rho0; reach=[]; Kinv=np.linalg.inv(K-1e-9*np.eye(d*d))   # nucleo |g0><g0| no contribuye (a|0>=0)
    for m in range(mmax):
        Y=-Kinv@src                       # int rho_m dt
        src=T@Y; reach.append(np.real(src[::d+1].sum()))   # prob de alcanzar el salto m+1
    reach=np.array(reach+[0.0]); Pm=reach[:-1]-reach[1:]   # P(M=m+1 saltos de modo y L=0 al final)
    return dict(P1=Pm[0],P2=Pm[1],P3=Pm[2],reach=reach,Dq=Dq)
if __name__=='__main__':
    for g,al in [(15.2*TP,-138*TP),(15.2*TP,-300*TP),(5*TP,-250*TP),(0.35*TP,-250*TP)]:
        Dq,gap=resonance(g,al); V=gap/2
        print(f"g/2pi={g/TP:.2f} MHz alpha/2pi={al/TP:.0f}: Dq/2pi={Dq/TP:.2f} (pred -alpha/2={-al/2/TP:.1f}); V_num/2pi={V/TP:.4f} MHz; 4g^2/|alpha|/2pi={4*g*g/abs(al)/TP:.4f}; beta=2sqrt2 g/|alpha|={2*2**.5*g/abs(al):.3f}")
