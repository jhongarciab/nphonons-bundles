import numpy as np, time, json, sys
from tls_model import *
data=np.load('../phi_hitchcock_fig4e.npy'); sel=[0,1,3,7,12,17]; det=data[sel,0]; tgt=data[sel,1]
base=dict(W=80.,glo=0.002,glo_rate=1e-3,fsr=3.29,profile='travelling',glob_n=0,glob_amp=0.0)
rng0=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0); rows=[]; t=time.time()
for it in range(int(sys.argv[2]) if len(sys.argv)>2 else 150):
    p=dict(base,nf=float(np.exp(rng0.uniform(np.log(0.01),np.log(0.4)))),ghi=float(rng0.uniform(0.15,1.2)),amp=float(np.exp(rng0.uniform(np.log(0.2),np.log(5)))),ghi_rate=float(10**rng0.uniform(0,2)))
    rng=np.random.default_rng(5)
    o=np.array([phi_curve(rng,p,det) for _ in range(30)])
    m=o[:,:,0].mean(0); Tphi=float(np.median(o[:,0,1]))
    chi=float(np.sum(((m-tgt)/0.05)**2)+ (np.log(Tphi/107.)/0.3)**2)
    rows.append(dict(p={k:p[k] for k in ('nf','ghi','amp','ghi_rate')},Tphi=Tphi,chi2=chi,m=m.tolist()))
rows.sort(key=lambda r:r['chi2'])
json.dump(rows,open(f'fit2_rows_{sys.argv[1] if len(sys.argv)>1 else 0}.json','w'))
print("done",time.time()-t, rows[0])
