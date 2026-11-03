import numpy as np, itertools
from A import H as HA, ops
from scipy.optimize import minimize_scalar
import A
def sp_ops(N):  # qubit(e,g) x osc
    a=np.diag(np.sqrt(np.arange(1,N)),1); sm=np.array([[0,0],[1.,0]]) # e->g : |g><e|, basis (e,g): sm[1,0]=1
    sm=np.array([[0,0],[1.,0]])
    return np.kron(np.eye(2),a),np.kron(sm,np.eye(N))
def vec_ops(H,Ls):
    d=H.shape[0]; I=np.eye(d)
    Hn=H-0.5j*sum(L.conj().T@L for L in Ls)
    S0=-1j*(np.kron(Hn,I)-np.kron(I,Hn.conj()))   # row-major vec: A rho B -> kron(A,B.T)
    J=[np.kron(L,L.conj()) for L in Ls]
    return S0,J
def eta_renew(H,La,Lq,rho0v):
    S0,(Ja,Jq)=vec_ops(H,[La,Lq]); d=H.shape[0]; tr=np.eye(d).reshape(-1)
    R=-np.linalg.inv(S0-1e-9*np.eye(S0.shape[0]))
    r2=Ja@R@Ja@R@rho0v
    surv=tr@r2-tr@((Ja+Jq)@R@r2)
    return surv.real
def eta_formula(kq,km,G): return 1/(1+(kq/(4*G**2))*(km+kq/2+2*G**2/km))
if __name__=="__main__":
    G=1.0; N=8; a,sm=sp_ops(N)
    Hef=G*(sm.T@a@a+sm@a.T@a.T)    # sigma+ a^2 + h.c. ; sm.T = sigma+
    e0=np.zeros(2*N); e0[0]=1; rho0=np.outer(e0,e0).reshape(-1)
    print("kq km | eta_renew(Hef,N=8) | eta_formula_claimed | eta_closed(mine)")
    mx=0
    for kq in (0.05,0.2,0.5,1.4):
        for km in (0.5,1,np.sqrt(2),2,4):
            e1=eta_renew(Hef,np.sqrt(km)*a,np.sqrt(kq)*sm,rho0); e2=eta_formula(kq,km,G)
            y1,y2,V2=kq/2,km,2*G*G; e3=1/(1+y1/y2+y1*(y1+y2)/V2)
            mx=max(mx,abs(e1-e2)); print(f"{kq} {km:.4f} | {e1:.6f} | {e2:.6f} | {e3:.6f}")
    print("max |renew-claimed| =",mx)
    # umbral
    from scipy.optimize import brentq
    f=lambda x: eta_formula(x,np.sqrt(2),1)-0.95; print("kq*(eta=0.95)=",brentq(f,1e-3,1))
    # km optimo
    for kq in (0.05,1.4):
        r=minimize_scalar(lambda k:-eta_formula(kq,k,1),bounds=(0.1,10),method='bounded',options={'xatol':1e-10}); print("km* opt kq=",kq,r.x,"sqrt2=",np.sqrt(2))
    print("1/eta-1 at km*: ", [ (1/eta_formula(kq,np.sqrt(2),1)-1, kq/np.sqrt(2)+kq**2/8) for kq in (0.05,1.4)])
    # MC check (sin bombeo) sobre Hef
    from mc import run
    rng=np.random.default_rng(1)
    for kq,km in [(0.2,np.sqrt(2)),(1.4,1.0)]:
        B=20000; ev=run(Hef,[np.sqrt(km)*a,np.sqrt(kq)*sm],e0.astype(complex),B,40.0,0.01,rng)
        na=np.zeros(B,int); nq=np.zeros(B,int)
        for tr,t,k in ev:
            (na if k==0 else nq)[tr]+=1   # tr unique within each call
        eta=np.mean((na==2)&(nq==0)); print(f"MC kq={kq} km={km:.3f}: eta={eta:.4f}+-{np.sqrt(eta*(1-eta)/B):.4f} formula={eta_formula(kq,km,1):.4f}")
    # Modelo A completo, jumps dressed (parte de frecuencia negativa)
    for gx,gz,kq,km in [(0.02,0.1,0.2,np.sqrt(2)),(0.02,0.1,1.4,np.sqrt(2)),(0.03,0.15,0.5,1.0)]:
        Nf=24
        Gt=2*gx*gz*np.exp(-2*gz**2)
        f=lambda w:A.gap(w,gx,gz,Nf); r=minimize_scalar(f,bounds=(2-0.05*gx-4*gx**2-20*gx*gz,2+20*gx*gz),method='bounded',options={'xatol':1e-13})
        wq=r.x; Gt=f(wq)/(2*np.sqrt(2)); Hf=HA(wq,gx,gz,Nf); E,V=np.linalg.eigh(Hf)
        aa,sz,sx,I=ops(Nf); sm_=np.kron(np.array([[0,0],[1.,0]]),np.eye(Nf))
        def low(O):
            Od=V.T@O@V; mask=(E[:,None]<E[None,:]); return Od*mask   # <i|O|j>, E_i<E_j
        La=low(aa); Lq=low(sm_)
        psi=np.zeros(2*Nf); psi[0]=1; c=V.T@psi; rho=np.outer(c,c).reshape(-1)
        Hd=np.diag(E)
        Gk=Gt*Gt and Gt
        k1,k2=kq*Gt,km*Gt   # tasas en unidades de G_efectivo
        e1=eta_renew(Hd,np.sqrt(k2)*La,np.sqrt(k1)*Lq,rho)
        print(f"Completo gx={gx} gz={gz} G_num~{Gt:.5f} kq/G={kq} km/G={km:.3f}: eta={e1:.5f} formula={eta_formula(kq,km,1):.5f}")
