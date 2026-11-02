import sys;sys.path.insert(0,'../../src')
import numpy as np, tls_model as M
base=dict(nf=0.05,W=80.,glo=0.002,ghi=0.5,glo_rate=1e-3,ghi_rate=1e2,amp=1.0,fsr=3.29,profile='travelling',glob_n=0,glob_amp=0.0)
for prof in ('travelling','standing'):
  p=dict(base,profile=prof);out={}
  for name,grid in (('TG70',np.geomspace(0.5,3000,70)),('fine2000',np.geomspace(0.5,3000,2000)),('fine6000',np.geomspace(0.5,3000,6000))):
    M.TG=grid
    out[name]=np.array([M.phi_curve(np.random.default_rng(s),p,[p['fsr']])[0] for s in range(300)])
  a,b,c=out['TG70'],out['fine2000'],out['fine6000']
  cap=(a[:,1]>=3000)|(b[:,1]>=3000)
  print(prof,'frac t1 capped',np.mean(a[:,1]>=2999),'frac Phi==0 (code)',np.mean(a[:,0]==0))
  for k,nm in ((0,'Phi'),(1,'t1')):
    d=(a[:,k]-c[:,k]); r=(a[:,k]/c[:,k]-1) if k==1 else d
    m=~cap
    print(' ',nm,'TG70-fine6000: mean',d[m].mean().round(4),'rms',np.sqrt((d[m]**2).mean()).round(4),'max',np.abs(d[m]).max().round(3), 'median rel' if k else '', np.median(r[m]).round(4))
    print('   fine2000 vs fine6000 rms',np.sqrt(((b[:,k]-c[:,k])[m]**2).mean()).round(5))
  m=(a[:,1]<3000)&(c[:,1]<3000)
  print(' mean Phi all',a[:,0].mean().round(3),c[:,0].mean().round(3),' mean Phi uncapped-both',a[m,0].mean().round(3),c[m,0].mean().round(3),m.sum())
  np.save(f'gb_{prof}.npy',np.array([a,b,c]))
