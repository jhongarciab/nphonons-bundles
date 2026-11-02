import numpy as np, json, glob, time, sys
from tls_model import *
rows=[]
for f in glob.glob('fit2_rows_*.json'): rows+=json.load(open(f))
rows.sort(key=lambda r:r['chi2']); top=rows[:8]
base=dict(W=80.,glo=0.002,glo_rate=1e-3,glob_n=0,glob_amp=0.0)
out=[]; t0=time.time()
idx=int(sys.argv[1]); parts=int(sys.argv[2])
for i,r in enumerate(top):
    if i%parts!=idx: continue
    for prof in ('travelling','standing'):
        for s in (0.25,0.5,1.0,2.0,4.0):
            p=dict(base,profile=prof,fsr=3.29/s,nf=r['p']['nf']*s,ghi=r['p']['ghi'],amp=r['p']['amp']/s,ghi_rate=r['p']['ghi_rate'])
            rng=np.random.default_rng(100+i)
            nd=100
            res=np.array([phi_curve(rng,p,[p['fsr']])[0] for _ in range(nd)])  # (nd,3): Phi,Tphi1,Tphi2
            Phi=res[:,0]; Tphi=np.median(res[:,1])
            out.append(dict(set=i,chi2=r['chi2'],profile=prof,s=s,fsr=3.29/s,Phi_mean=float(Phi.mean()),Phi_std=float(Phi.std()),Phi_med=float(np.median(Phi)),
                            P08=float((Phi>0.8).mean()),P09=float((Phi>0.9).mean()),Tphi_med=float(Tphi),Phi_all=Phi.tolist()))
            print(i,prof,s,f"Phi={Phi.mean():.2f}±{Phi.std():.2f} P>0.8={np.mean(Phi>0.8):.2f} Tphi={Tphi:.0f} [{time.time()-t0:.0f}s]",flush=True)
json.dump(out,open(f'predict1_{idx}.json','w'))
