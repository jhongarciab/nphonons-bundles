import numpy as np
from qdm import *
gam=2e-3;kap=0.003
print('## Comparación del mejor g2 (n>0.02) y máx n sobre wL en [-1.3,1.3] (paso 0.02 salvo cerca de resonancia), gam=2e-3 kap=0.003')
cases={'molecula S=1 eps=0 lam=(.14,0)':dict(eps=0,J=0.5,l1=0.14,l2=0.0),
       'molecula lam=(.14,.14) dl=0':dict(eps=0,J=0.5,l1=0.14,l2=0.14),
       '2 indep J=0, lam=(.14,0), eps=0':dict(eps=0,J=0.0,l1=0.14,l2=0.0),
       'emisor unico lam=.14':dict(eps=0,J=0.0,l1=0.14,l2=0.0,Om2=0.0)}
for lab,kw in cases.items():
  for Om in (0.004,0.05):
    best=None;bn=None
    for wL in list(np.linspace(-1.3,1.3,53))+list(np.linspace(0.02,0.1,9))+list(np.linspace(0.45,0.58,14)):
        r,_,_=ss(Nb=18,wL=wL,Om=Om,gam=gam,kap=kap,**kw)
        if r['n']>0.02 and (best is None or r['g2']<best[1]['g2']): best=(wL,r)
        if bn is None or r['n']>bn[1]['n']: bn=(wL,r)
    print(lab,'Om=%g | min g2: wL=%.3f g2=%.3f n=%.3f Q=%.3f | max n: wL=%.3f n=%.3f g2=%.3f'%((Om,best[0],best[1]['g2'],best[1]['n'],best[1]['Q'],bn[0],bn[1]['n'],bn[1]['g2']) if best else (Om,0,0,0,0,bn[0],bn[1]['n'],bn[1]['g2'])),flush=True)
