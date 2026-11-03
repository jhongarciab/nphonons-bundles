import numpy as np
from C import *
from scipy.optimize import minimize_scalar
for gx,gz,kq,km in [(0.006,0.06,0.2,np.sqrt(2)),(0.006,0.06,1.4,np.sqrt(2)),(0.01,0.1,0.2,np.sqrt(2))]:
    Nf=20; f=lambda w:A.gap(w,gx,gz,Nf); r=minimize_scalar(f,bounds=(2-0.05*gx-4*gx**2-20*gx*gz,2+20*gx*gz),method='bounded',options={'xatol':1e-14})
    Gt=f(r.x)/(2*np.sqrt(2)); E,Vv=np.linalg.eigh(HA(r.x,gx,gz,Nf)); aa,sz,sx,I=ops(Nf); sm_=np.kron(np.array([[0,0],[1.,0]]),np.eye(Nf))
    low=lambda O:(Vv.T@O@Vv)*(E[:,None]<E[None,:])
    c=Vv[0]; rho=np.outer(c,c).reshape(-1)
    for tag,s in (("dressed-sec",1),):
        e1=eta_renew(np.diag(E),np.sqrt(km*Gt)*low(aa),np.sqrt(kq*Gt)*low(sm_),rho)
        print(f"gx={gx} gz={gz} kq/G={kq}: eta_full={e1:.5f} formula={eta_formula(kq,km,1):.5f} diff={100*(e1-eta_formula(kq,km,1)):+.2f} pp")
