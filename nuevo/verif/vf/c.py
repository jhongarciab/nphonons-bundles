import sys; sys.path.insert(0,'/home/user/nphonons-bundles/nuevo/bundles/src')
import numpy as np, bundles_core as bc
for l in (.14,.1):
  ks=np.geomspace(2e-3,2e-2,25);P=[bc.exact_point(2,l,.1,k,2e-4,4e-4)['P'] for k in ks];i=int(np.argmax(P));print(l,ks[i],P[i])
for g in (2e-5,1.8e-4,1.6e-3): print(g,bc.exact_point(2,.14,.1,.003,g,4e-4)['P'])
for x in (1e-6,7e-3): print(x,bc.exact_point(2,.14,.1,.003,2e-4,x)['P'])
print(bc.exact_point(4,.14,.1,.003,2e-4,4e-4)['P'])
