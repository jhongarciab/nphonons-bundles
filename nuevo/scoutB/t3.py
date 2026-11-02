import numpy as np, qutip as qt
from bl import *
def nb(dL,lam,Om,kap,gam,gph,N):
    H,c,L,Jb,Jg,b,s=build(dL,lam,Om,kap,gam,gph,N)
    rho=qt.steadystate(H,c); return qt.expect(b.dag()*b,rho)
def pi_mun(lam,Om,kap,gam,gph,n,Nf=12,dL=None):
    dL=dL_res(lam,Om,n) if dL is None else dL
    full=nb(dL,lam,Om,kap,gam,gph,Nf); tr=nb(dL,lam,Om,kap,gam,gph,n)
    return 1-tr/full, full
for (lam,kap,gam,Om) in [(.14,.003,2e-4,.1),(.14,.01,2e-4,.1),(.1,.002,2e-4,.2),(.06,.002,2e-4,.2),(.14,.003,2e-3,.1)]:
    print(lam,kap,gam,Om,'n=2 pi_Munoz,<n>full',pi_mun(lam,Om,kap,gam,4e-4,2))
