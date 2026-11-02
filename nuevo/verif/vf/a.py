import sys; sys.path.insert(0,'/home/user/nphonons-bundles/nuevo/bundles/src')
import numpy as np, bundles_core as bc
n=2;lam=.14;g=2e-4;gp=4e-4
print('Ommax95',[bc.Om_max(k,.95,lam,g,gp,n) for k in (.002,.003,.005,.01,.02)])
print('Ommax90',[bc.Om_max(k,.90,lam,g,gp,n) for k in (.002,.003)])
print('kstar',[bc.kappa_star(n,lam,O,g,gp) for O in (.05,.1,.2)])
for l in (.14,.1): 
  ks=np.geomspace(1e-3,3e-2,60);P=[bc.exact_point(2,l,.1,k,g,gp)['P'] for k in ks[::3]]
  print('exact max',l,ks[::3][int(np.argmax(P))],max(P))
print('P3 max',max((bc.P_closed(lam,.1,k,g,gp,3),k) for k in np.geomspace(5e-4,.02,200)))
print('P4',bc.P_closed(lam,.1,.003,g,gp,4))
print('gph',[bc.P_closed(lam,.1,.003,g,x,2) for x in (1e-6,7e-3)],[bc.P_closed(lam,.1,.003,x,gp,2) for x in (2e-5,1.8e-4,1.6e-3)])
d=np.load('../../bundles/data/R_requirements.npz');J=d['J'];L=d['lams'];G=d['gams']
print(np.nanmax(J))
for e in (1e-5,):
  m=J>=e; print('lam min',L[m.any(0)].min(),'gam max',G[m.any(1)].max())
  # boundary: for each lam, max gam
  for j in range(0,34,2): 
    ii=np.where(m[:,j])[0]; print(round(L[j],4), G[ii.max()] if len(ii) else None, np.nanmax(J[:,j]) if np.isfinite(J[:,j]).any() else None)
