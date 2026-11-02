import numpy as np
from bl import *; from rate import *
for kw in [dict(),dict(kap=.01),dict(kap=.001),dict(gam=2e-5),dict(lam=.2),dict(Om=.2)]:
    p=dict(lam=.14,Om=.1,kap=.003,gam=2e-4,gph=4e-4); p.update(kw);n=2
    pe=episodes_pol(-n+eps_n(p['lam'],p['Om'],n),p['lam'],p['Om'],p['kap'],p['gam'],p['gph'],N=9)
    pr=episodes_rate(p['lam'],p['Om'],p['kap'],p['gam'],p['gph'],n)
    print(kw,'exact',np.round(pe[:6],3),'rate',np.round(pr[:6],3))
