import numpy as np
from bl import *
lam,Om,kap,gam,gph=0.14,0.1,0.003,2e-4,4e-4; n=2
dl=-n+eps_n(lam,Om,n)
for dr in (False,True):
    for dd in (-0.001,0,0.001):
        P,T,F=cycles_pol(dl+dd,lam,Om,kap,gam,gph,N=10,K=6,dressed=dr)
        print('dressed' if dr else 'local-pol',dd,np.round(P,3),f'T={T:.0f} F={F*1e5:.2f}e-5')
