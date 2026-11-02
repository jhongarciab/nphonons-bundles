"""Modelo COMPLETO (qubit + modo, g_x y g_z, sin RWA en el acoplo) en la base de autoestados vestidos del H sin bombeo.
Bombeo Omega(sigma+ e^{-i wp t}+hc) con RWA SOLO en el bombeo (secular entre autoestados con dnu=2), marco rotante diagonal K=nu/2.
Disipadores vestidos y seculares por bandas de frecuencia (Lorentz/plano): J(banda) configurable.  omega=1."""
import numpy as np, qutip as qt
from scipy.optimize import minimize_scalar

def bare(N):
    a=qt.tensor(qt.qeye(2),qt.destroy(N)); sz=qt.tensor(qt.sigmaz(),qt.qeye(N))   # e=arriba
    sx=qt.tensor(qt.sigmax(),qt.qeye(N)); sp=qt.tensor(qt.sigmap(),qt.qeye(N))
    return a,sz,sx,sp
def H0f(wq,gx,gz,N):
    a,sz,sx,sp=bare(N); return a.dag()*a+0.5*wq*sz+(a+a.dag())*(gx*sx+gz*sz)
def wq_res(gx,gz,N=12):
    N_=N
    def gap(w):
        E,V=np.linalg.eigh(H0f(w,gx,gz,N).full()); wt=np.abs(V[0,:])**2+np.abs(V[N_+2,:])**2
        i=np.argsort(wt)[-2:]; return abs(E[i[0]]-E[i[1]])
    return minimize_scalar(gap,bounds=(2-40*gx**2-1e-4,2+1e-4),method='bounded',options={'xatol':1e-11}).x
class Full:
    def __init__(s,gx,gz,N=10,wq=None,dwq=0.0):
        s.gx,s.gz,s.N=gx,gz,N
        s.wq=(wq_res(gx,gz,N) if wq is None else wq)+dwq
        H=H0f(s.wq,gx,gz,N); E,V=np.linalg.eigh(H.full()); s.E,s.V=E,V
        a,sz,sx,sp=bare(N); tr=lambda O: V.conj().T@O.full()@V
        s.b,s.sm,s.sz,s.sp=tr(a),tr(sp.dag()),tr(sz),tr(sp)
        dom=np.argmax(np.abs(V)**2,axis=0); sq=dom//N; n=dom%N          # sq=0 -> e
        s.nu=np.where(sq==0,n+1,n-1); s.dom=dom
        s.ig0=int(np.where(dom==N+0)[0][0])
        wt=np.abs(V[0,:])**2+np.abs(V[N+2,:])**2; s.hyb=list(np.argsort(wt)[-2:])
    def band(s,M,k):
        """parte de M con E_j-E_i en banda k (k=0:|dE|<.5, k>0:dE~k, solo bajadas j->i con E_j>E_i)"""
        dE=s.E[None,:]-s.E[:,None]; m=(np.abs(dE-k)<0.5) if k>0 else (np.abs(dE)<0.5)
        return np.where(m,M,0)
    def build(s,Om,delta,kap_m,kap_q,gphi=0.0,Jdc=0.0,J2=1.0,J3=1.0,Jinf=0.0):
        """delta: desintonia del bombeo respecto a 2*omega (wp=2+delta). J2: densidad relativa a 2w (Purcell); Jdc: DC; """
        dim=len(s.E); K=np.diag(s.nu/2.0); wp=2.0+delta
        Hr=np.diag(s.E)-wp*K
        D=np.where((s.nu[:,None]-s.nu[None,:])==2,s.sp,0)       # <i|s+|j> con nu_i-nu_j=2
        Hr=Hr+Om*(D+D.conj().T)
        c=[];lab=[]
        B1=s.band(s.b,1); c.append(np.sqrt(kap_m)*B1); lab.append('b1')
        B2=s.band(s.b,2)
        if J2>0: c.append(np.sqrt(kap_m*J2)*B2); lab.append('b2')
        B3=s.band(s.b,3)
        if J3>0: c.append(np.sqrt(kap_m*J3)*B3); lab.append('b3')
        if Jdc>0: c.append(np.sqrt(kap_m*Jdc)*s.band(s.b,0)); lab.append('b0')
        if kap_q>0: c.append(np.sqrt(kap_q)*s.band(s.sm,2)); lab.append('q')
        if gphi>0: c.append(np.sqrt(gphi/2)*s.band(s.sz,0)); lab.append('phi')
        s.lab=lab; s.B1=B1
        return qt.Qobj(Hr),[qt.Qobj(x) for x in c]
    def ss(s,*a,**k):
        H,c=s.build(*a,**k); rho=qt.steadystate(H,c,method='direct'); return rho,H,c
    def stats(s,rho,c):
        r=rho.full(); out={}
        out['rate']={l:float(np.real(np.trace(x.full().conj().T@x.full()@r))) for l,x in zip(s.lab,c)}
        B=s.B1; ex=lambda O: float(np.real(np.trace(O@r)))
        nb=ex(B.conj().T@B); out['J1']=nb*0+out['rate']['b1']
        out['g2']=ex(B.conj().T@B.conj().T@B@B)/max(ex(B.conj().T@B)**2,1e-300)
        nn=s.V.conj().T@qt.tensor(qt.qeye(2),qt.num(s.N)).full()@s.V
        out['n']=ex(nn)
        # P(n) del modo: proyector sobre el estado bare n en base original
        Pn=[]; r0=s.V@r@s.V.conj().T
        for n in range(s.N): Pn.append(float(np.real(r0[n,n]+r0[s.N+n,s.N+n])))
        out['Pn']=Pn; out['pe']=ex(s.V.conj().T@qt.tensor(qt.sigmap()*qt.sigmam(),qt.qeye(s.N)).full()@s.V)
        return out
