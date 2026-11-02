import numpy as np,sys
from bl import *
lam,Om,kap,gam,gph=0.14,0.1,0.003,2e-4,4e-4
for n in (2,3):
  r0=dL_res(lam,Om,n); print('n',n,'dL_res',r0,'eps',eps_n(lam,Om,n),'bin(III)',lam**2-np.sqrt(n*n-4*Om**2),'Omn',Omn(lam,Om,n))
  best=None
  for dd in np.linspace(-0.004,0.004,17):
    P,T,F=cycles(r0+dd,lam,Om,kap,gam,gph,N=10,K=6)
    print(f'{dd:+.4f} flux={F*1e5:6.2f} T={T:7.0f} P=',np.round(P,3))
