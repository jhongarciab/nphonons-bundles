import numpy as np,time
from bl import *; from rate import *
pts=[dict(),dict(kap=.002),dict(kap=.001),dict(kap=.006),dict(kap=.01),dict(Om=.03),dict(Om=.2),dict(lam=.07),dict(lam=.2),dict(gam=1e-3),dict(kap=.0015,Om=.15)]
print('pt | exact P2 | rate P2 | closed P2 | exact flux | closed flux')
for kw in pts:
    p=dict(lam=.14,Om=.1,kap=.003,gam=2e-4,gph=4e-4); p.update(kw);n=2
    t=time.time()
    d=-n+eps_n(p['lam'],p['Om'],n)
    pe=episodes_pol(d,p['lam'],p['Om'],p['kap'],p['gam'],p['gph'],N=8,K=8)
    pr=episodes_rate(p['lam'],p['Om'],p['kap'],p['gam'],p['gph'],n)
    # flujo exacto de fonones (maestra vestida, estacionario)
    H,c,L,Jb,Jg=build_pol(d,p['lam'],p['Om'],p['kap'],p['gam'],p['gph'],8)
    rho=qt.steadystate(H,c); b,s=ops(8); F=p['kap']*qt.expect(b.dag()*b,rho)
    print(kw,'| %.4f | %.4f | %.4f | %.3e | %.3e | %.1fs'%(pe[2],pr[2],P_closed(p['lam'],p['Om'],p['kap'],p['gam'],p['gph'],n),F,flux_closed(p['lam'],p['Om'],p['kap'],p['gam'],p['gph'],n),time.time()-t),flush=True)
