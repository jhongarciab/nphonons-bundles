import numpy as np
from qdm import *
def row(**kw):
    r,_,_=ss(**kw); return r
print('## A) fino alrededor de d~g, S=1, eps=0, J=0.5, lam=(0.14,0), kap=0.003')
for gam in (2e-4,2e-3,1e-2):
  best=None
  for Om in (0.001,0.002,0.004,0.007):
    for d in np.linspace(0.03,0.09,13):
        r=row(Nb=16,eps=0,J=0.5,wL=0.5+d,gam=gam,Om=Om,kap=0.003)
        if r['n']>0.02 and (best is None or r['g2']<best[2]['g2']): best=(Om,d,r)
  Om,d,r=best; print('gam=%g Om=%g d=%.3f -> g2=%.3f n=%.3f Q=%.3f P=%s'%(gam,Om,d,r['g2'],r['n'],r['Q'],np.round(r['P'][:4],3)),flush=True)
print('## B) convergencia Nb en el optimo gam=2e-3 Om=0.002 d=0.06')
for Nb in (10,16,24): 
    r=row(Nb=Nb,eps=0,J=0.5,wL=0.56,gam=2e-3,Om=0.002,kap=0.003); print(Nb,'g2=%.4f n=%.4f'%(r['g2'],r['n']))
print('## C) limites en el optimo (gam=2e-3,Om=0.002,d=0.06)')
base=dict(Nb=16,eps=0,J=0.5,wL=0.56,gam=2e-3,Om=0.002,kap=0.003)
for lab,mod in (('base',{}),('dl=0 (l1=l2=0.14)',dict(l2=0.14)),('l1=l2=0.07',dict(l1=0.07,l2=0.07)),('J=0.01 (2 emisores indep.)',dict(J=0.01,wL=0.0+0.06)),('emisor unico (l2=0,Om2=0,J=0)',dict(J=0.0,Om2=0.0,wL=0.06))):
    kw=dict(base); kw.update(mod); r=row(**kw); print(lab,'g2=%.3f n=%.4f Q=%.3f'%(r['g2'],r['n'],r['Q']))
