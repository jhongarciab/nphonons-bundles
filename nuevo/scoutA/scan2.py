import numpy as np,sys
from qdm import *
S=1.0; eps=0.0; J=np.sqrt(S**2-eps**2)/2
print('S=1 eps=0 lam=(0.14,0) kap=0.005 gph=4e-4; mejor wL por (gam,Om) segun min g2 con n>0.02 y max n')
for gam in (2e-4,2e-3,1e-2):
  for Om in (0.05,0.1,0.2,0.4):
    res=[]
    for d in np.linspace(-0.15,0.15,13):
        r,_,_=ss(Nb=20,eps=eps,J=J,wL=S/2+d,gam=gam,Om=Om,kap=0.005)
        res.append((d,r))
    d,r=max(res,key=lambda x:x[1]['n'])
    print('gam=%g Om=%.2f | max n: d=%.3f n=%.3f g2=%.3f Q=%.3f P=%s edge=%.1e'%(gam,Om,d,r['n'],r['g2'],r['Q'],np.round(r['P'][:4],2),r['edge']),flush=True)
