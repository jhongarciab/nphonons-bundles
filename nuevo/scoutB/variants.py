import numpy as np
from bl import *; from rate import *
lam,Om,gam,gph=.14,.1,2e-4,4e-4;n=2
print('kap | P2_ep(sigma D^+, fisico) | P2_ep(sigma sin D^+) | closed ; ciclos: P2 vestida / local')
for kap in (.002,.003,.01):
    d=-n+eps_n(lam,Om,n)
    a=episodes_pol(d,lam,Om,kap,gam,gph,N=8,fc=True)[2]; b_=episodes_pol(d,lam,Om,kap,gam,gph,N=8,fc=False)[2]
    cd=cycles_pol(d,lam,Om,kap,gam,gph,N=8,K=5,dressed=True)[0][2]; cl=cycles_pol(d,lam,Om,kap,gam,gph,N=8,K=5,dressed=False)[0][2]
    print(f'{kap} | {a:.4f} | {b_:.4f} | {P_closed(lam,Om,kap,gam,gph,n):.4f} ; {cd:.3f} / {cl:.3f}',flush=True)
