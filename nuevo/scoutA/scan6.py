import numpy as np,sys
from qdm import *
cfgs=[('A l=(.14,0) k=.003 g=2e-4',dict(l1=.14,l2=0,kap=.003,gam=2e-4)),
      ('B l=(.14,0) k=.003 g=1e-2',dict(l1=.14,l2=0,kap=.003,gam=1e-2)),
      ('C l=(.14,0) k=.001 g=1e-2',dict(l1=.14,l2=0,kap=.001,gam=1e-2)),
      ('D l=(.14,-.14) k=.003 g=2e-4',dict(l1=.14,l2=-.14,kap=.003,gam=2e-4)),
      ('E l=(.14,-.14) k=.003 g=1e-2',dict(l1=.14,l2=-.14,kap=.003,gam=1e-2)),
      ('F l=(.14,-.14) k=.001 g=1e-2',dict(l1=.14,l2=-.14,kap=.001,gam=1e-2))]
Oms=[0.002,0.005,0.01,0.02,0.04]
print('Barrido Om in',Oms,'d in [-0.3,0.3] (25 pts); S=1, eps=0, gph=4e-4, Nb=18; criterio n>0.05')
for lab,kw in cfgs:
    R=[]
    for Om in Oms:
        for d in np.linspace(-0.3,0.3,25):
            r,_,_=ss(Nb=18,eps=0,J=0.5,wL=0.5+d,Om=Om,gph=4e-4,**kw); R.append((Om,d,r))
    ok=[x for x in R if x[2]['n']>0.05]
    a=min(ok,key=lambda x:x[2]['g2']); q=min(ok,key=lambda x:x[2]['Q']); f=max(R,key=lambda x:x[2]['P'][1]); f2=max(R,key=lambda x:x[2]['P'][2])
    print(lab,'| min g2: Om=%g d=%.2f g2=%.3f n=%.2f | min Q: Om=%g d=%.2f Q=%.2f n=%.2f P1=%.2f | max P1=%.2f (Om=%g d=%.2f, n=%.2f) | max P2=%.2f'%(a[0],a[1],a[2]['g2'],a[2]['n'],q[0],q[1],q[2]['Q'],q[2]['n'],q[2]['P'][1],f[2]['P'][1],f[0],f[1],f[2]['n'],f2[2]['P'][2]),flush=True)
