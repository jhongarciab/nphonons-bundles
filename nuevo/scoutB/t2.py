import numpy as np
from bl import *
base=dict(lam=0.14,Om=0.1,kap=0.003,gam=2e-4,gph=4e-4)
def run(**kw):
    p=dict(base);p.update(kw);n=2
    d=dL_res(p['lam'],p['Om'],n)
    P,T,F=cycles(d,p['lam'],p['Om'],p['kap'],p['gam'],p['gph'],N=9,K=5)
    return P,T,F
print('N-conv',[np.round(cycles(dL_res(.14,.1,2),.14,.1,.003,2e-4,4e-4,N=N,K=5)[0],4) for N in (7,9,12)])
for k,vals in dict(Om=[0.03,0.05,0.2],kap=[0.001,0.01,0.03],gam=[2e-5,2e-3],gph=[0,4e-3],lam=[0.07,0.2,0.28]).items():
    for v in vals:
        P,T,F=run(**{k:v}); print(k,v,np.round(P,3),f'T={T:.0f} F={F*1e5:.2f}e-5')
