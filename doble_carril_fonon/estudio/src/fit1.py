import numpy as np, itertools, time, json
from tls_model import *
data=np.load('../phi_hitchcock_fig4e.npy'); sel=[0,1,3,7,12,17]; det=data[sel,0]; tgt=data[sel,1]; sd=data[sel,2]
base=dict(nf=0.05,W=80.,glo=0.002,ghi=0.5,glo_rate=1e-3,ghi_rate=1e2,amp=1.0,fsr=3.29,profile='travelling',glob_n=0,glob_amp=0.0)
rows=[]; t=time.time()
for nf,ghi,gn,ga in itertools.product((0.02,0.04,0.08,0.16),(0.3,0.5,0.8),(0,3,10),(0.003,0.01)):
    if gn==0 and ga!=0.003: continue
    p=dict(base,nf=nf,ghi=ghi,glob_n=gn,glob_amp=ga)
    rng=np.random.default_rng(3)
    o=np.array([phi_curve(rng,p,det) for _ in range(40)])
    m=o[:,:,0].mean(0); Tphi=np.median(o[:,0,1])
    chi=np.sum(((m-tgt)/0.05)**2)
    rows.append(dict(nf=nf,ghi=ghi,gn=gn,ga=ga,Tphi=float(Tphi),chi2=float(chi),m=m.tolist()))
rows.sort(key=lambda r:r['chi2'])
print("data",np.round(tgt,2))
for r in rows[:10]: print({k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items() if k!='m'}, np.round(r['m'],2))
print("elapsed",time.time()-t)
json.dump(rows,open('fit1_rows.json','w'))
