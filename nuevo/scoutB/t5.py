import numpy as np
from bl import *
base=dict(lam=0.14,Om=0.1,kap=0.003,gam=2e-4,gph=4e-4)
def run(**kw):
    p=dict(base);p.update(kw);n=2
    P,T,F=cycles_pol(-n+eps_n(p['lam'],p['Om'],n),p['lam'],p['Om'],p['kap'],p['gam'],p['gph'],N=9,K=6)
    return P,T,F
for k,vals in dict(Om=[0.03,0.2],kap=[0.001,0.01],gam=[2e-5,2e-3],gph=[0,4e-3],lam=[0.07,0.2]).items():
    for v in vals:
        P,T,F=run(**{k:v}); print(k,v,np.round(P,3),f'T={T:.0f} F={F*1e5:.2f}e-5')
