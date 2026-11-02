import numpy as np, time, json, sys
from tls_model import *
data=np.load('../phi_hitchcock_fig4e.npy'); sel=[0,1,3,7,12,17]; det=data[sel,0]; tgt=data[sel,1]; sdd=0.5*(data[sel,2]+data[sel,3])
base=dict(W=80.,glo=0.002,glo_rate=1e-3,fsr=3.29,profile='travelling')
seed=int(sys.argv[1]); N=int(sys.argv[2]); rng0=np.random.default_rng(1000+seed); rows=[]; t=time.time()
for it in range(N):
    gn=int(rng0.choice([0,0,3,10,30]))
    p=dict(base,nf=float(np.exp(rng0.uniform(np.log(0.01),np.log(2.0)))),ghi=float(np.exp(rng0.uniform(np.log(0.05),np.log(1.2)))),
           amp=float(np.exp(rng0.uniform(np.log(0.2),np.log(20)))),ghi_rate=float(10**rng0.uniform(0,2)),
           glob_n=gn,glob_amp=float(np.exp(rng0.uniform(np.log(0.0005),np.log(0.02)))))
    rng=np.random.default_rng(5)
    o=np.array([phi_curve(rng,p,det) for _ in range(30)])
    m=o[:,:,0].mean(0); s=o[:,:,0].std(0); Tphi=float(np.median(o[:,0,1]))
    chi_m=float(np.sum(((m-tgt)/0.05)**2)); chi_s=float(np.sum((np.maximum(s-sdd,0)/0.05)**2)); chi_T=float((np.log(max(Tphi,1.0)/107.)/0.3)**2)
    rows.append(dict(p={k:p[k] for k in ('nf','ghi','amp','ghi_rate','glob_n','glob_amp')},Tphi=Tphi,chi_m=chi_m,chi_s=chi_s,chi_T=chi_T,chi2=chi_m+chi_s+chi_T,m=m.tolist(),s=s.tolist()))
rows.sort(key=lambda r:r['chi2'])
json.dump(rows,open(f'fit3_rows_{seed}.json','w'))
print("done",time.time()-t)
