"""Modelo JC completo (transmon 3 niveles + 2 modos de nm niveles). us, rad/us. side='below': w_k=w_q-D_k (spec); 'above': w_k=w_q+D_k."""
import numpy as np, qutip as qt
from math import pi
from scipy.optimize import minimize_scalar
from mydisp import plat_si, logical_states, storage, phonon_cops
NQ=3
def build(nm):
    I=lambda n: qt.qeye(n)
    q=qt.tensor(qt.destroy(NQ),I(nm),I(nm)); a1=qt.tensor(I(NQ),qt.destroy(nm),I(nm)); a2=qt.tensor(I(NQ),I(nm),qt.destroy(nm))
    return q,a1,a2
class JC:
    def __init__(s,D,FSR,g=1.1,alpha=185.,nm=3,side='below',plat=None):
        s.nm=nm; s.q,s.a1,s.a2=build(nm); q,a1,a2=s.q,s.a1,s.a2
        sg=1 if side=='below' else -1
        s.D1=sg*2*pi*D; s.D2=sg*2*pi*(D+FSR)   # w_q-w_k
        s.Hjc=(-pi*alpha*q.dag()*q.dag()*q*q - s.D1*a1.dag()*a1 - s.D2*a2.dag()*a2
               +2*pi*g*(q.dag()*a1+a1.dag()*q+q.dag()*a2+a2.dag()*q))
        s.Hn=q.dag()*q+a1.dag()*a1+a2.dag()*a2   # DL*(q+q+a+a...) ; DL=w_q-w_L ; modes: (w_k-w_L)=DL-D_k
        s.Hdet=q.dag()*q+a1.dag()*a1+a2.dag()*a2
        s.Hd=0.5*(q+q.dag())
        s.nq=q.dag()*q; s.n1=a1.dag()*a1; s.n2=a2.dag()*a2
        # embedding 4->nm^2
        V=np.zeros((nm*nm,4));
        for n1 in range(2):
            for n2 in range(2): V[nm*n1+n2,2*n1+n2]=1
        s.V=V
    def H0(s,DL): return s.Hjc+DL*s.Hdet
    def pulse(s,sigma): 
        O0=pi/(sigma*np.sqrt(2*pi)); tp=4*sigma; t0=tp/2
        return tp,(lambda t,args=None:O0*np.exp(-(t-t0)**2/(2*sigma**2)))
    def cops(s,plat,rho):
        c=phonon_cops(plat,rho,s.a1,s.a2,s.n1,s.n2)
        gq=1/plat['T2q']-1/(2*plat['T1q'])
        return c+[np.sqrt(1/plat['T1q'])*s.q, np.sqrt(2*gq)*s.q.dag()*s.q]
    def pe_pure(s,DL,sigma,ket_ph):
        tp,f=s.pulse(sigma)
        psi0=qt.tensor(qt.basis(NQ,0),qt.Qobj(s.V@ket_ph.full().ravel(),dims=[[s.nm,s.nm],[1,1]]))
        r=qt.sesolve([s.H0(DL),[s.Hd,f]],psi0,[0,tp],options=dict(atol=1e-8,rtol=1e-7,nsteps=2000000)).states[-1]
        return qt.expect(s.nq*0+qt.tensor(qt.fock_dm(NQ,1),qt.qeye(s.nm),qt.qeye(s.nm)),r)
    def best_DL(s,sigma,guess,span=0.4):
        k10,k01=logical_states()[:2]
        fn=lambda DL:-0.5*(s.pe_pure(DL,sigma,k10)+s.pe_pure(DL,sigma,k01))
        r=minimize_scalar(fn,bounds=(guess-span,guess+span),method='bounded',options=dict(xatol=2e-3))
        return r.x,-r.fun
    def run_dm(s,rho1,DL,sigma,plat,rho,eps):
        tp,f=s.pulse(sigma); nm=s.nm
        rb=qt.Qobj(s.V@rho1.full()@s.V.T,dims=[[nm,nm],[nm,nm]])
        r0=qt.tensor(qt.fock_dm(NQ,0),rb)
        r=qt.mesolve([s.H0(DL),[s.Hd,f]],r0,[0,tp],s.cops(plat,rho),options=dict(atol=1e-8,rtol=1e-7,nsteps=2000000)).states[-1]
        R=r.full().reshape(NQ,nm*nm,NQ,nm*nm)
        un=(1-eps)*R[1,:,1,:]+eps*(R[0,:,0,:]+R[2,:,2,:])
        return un,np.trace(un).real
    def metric(s,T,rho,plat,DL,sigma,eps):
        psis=logical_states(); outs=[];pp=[];Ps=[]
        for psi in psis:
            r1=storage(psi,T,rho,plat)
            un,p=s.run_dm(r1,DL,sigma,plat,rho,eps); outs.append(un/p); pp.append(p)
        nm=s.nm
        vs=[s.V@p.full().ravel() for p in psis]
        def F(phi):
            ph=np.ones(nm*nm,complex); ph[1]=np.exp(1j*phi)   # |01> -> idx 1
            return np.mean([ (lambda v:(v.conj()@o@v).real)(ph.conj()*v) for o,v in zip(outs,vs)])
        F0,F1,F2=F(0),F(pi/2),F(pi); a0=(F0+F2)/2; a1=(F0-F2)/2; a2=F1-a0
        return dict(Fc=a0+np.hypot(a1,a2),pp=float(np.mean(pp)),pps=pp)
def chi_eff(D,FSR,side,g=1.1,alpha=185.):
    sg=1 if side=='below' else -1
    c=lambda d: g**2/d*alpha/(alpha-sg*d)*2*pi   # per-mode Koch chi (rad/us), correct sign for side
    return c(D),c(D+FSR)

def transitions(D,FSR,side,g=1.1,alpha=185.,nm=4):
    """frecuencias de transicion e-g (rad/us, relativas a w_q) de los sectores (0,0),(1,0),(0,1) por diagonalizacion"""
    j=JC(D,FSR,g,alpha,nm=nm,side=side); E,V=j.Hjc.eigenstates()
    def en(e,m1,m2):
        k=qt.tensor(qt.basis(NQ,e),qt.basis(nm,m1),qt.basis(nm,m2)); ov=[abs(k.overlap(v))**2 for v in V]; return E[int(np.argmax(ov))]
    return {s:en(1,*s)-en(0,*s) for s in [(0,0),(1,0),(0,1)]}
def best_DL_sym(jc,sigma,D,FSR,side,span=0.5):
    """DL=w_q-w_L que maximiza min(p_e|10, p_e|01) sin ruido, buscando alrededor del punto medio de los sectores"""
    tr=transitions(D,FSR,side); mid=0.5*(tr[(1,0)]+tr[(0,1)]); guess=-mid
    k10,k01=logical_states()[:2]
    fn=lambda DL:-min(jc.pe_pure(DL,sigma,k10),jc.pe_pure(DL,sigma,k01))
    r=minimize_scalar(fn,bounds=(guess-span,guess+span),method='bounded',options=dict(xatol=2e-3))
    return r.x,-r.fun,guess
