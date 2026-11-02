import numpy as np, json, glob
from tls_model import *
data=np.load('../phi_hitchcock_fig4e.npy'); sel=[0,1,3,7,12,17]; det=data[sel,0]; tgt=data[sel,1]; sdd=0.5*(data[sel,2]+data[sel,3])
base=dict(W=80.,glo=0.002,glo_rate=1e-3,fsr=3.29,profile='travelling')
sets=[]
r3=sorted(sum([json.load(open(f)) for f in glob.glob('fit3_rows_*.json')],[]),key=lambda r:r['chi2'])[:4]
r2=sorted(sum([json.load(open(f)) for f in glob.glob('fit2_rows_*.json')],[]),key=lambda r:r['chi2'])[:4]
for r in r3+r2: sets.append(r['p'])
print("data mean",np.round(tgt,2),"std",np.round(sdd,2))
for p0 in sets:
    p=dict(base,**{'glob_n':0,'glob_amp':0.0,**p0})
    rng=np.random.default_rng(11); nd=60
    o=np.array([phi_curve(rng,p,det) for _ in range(nd)])      # (nd,ndet,3)
    Phi=o[:,:,0]; t1=o[:,0,1]
    valid=np.isfinite(Phi)
    mean=np.array([np.nanmean(Phi[:,i]) for i in range(len(det))]); std=np.array([np.nanstd(Phi[:,i]) for i in range(len(det))])
    cens=1-valid.mean(0)
    print(f"nf={p['nf']:.2f} ghi={p['ghi']:.2f} amp={p['amp']:.1f} Tphi_med={np.nanmedian(t1):.0f} | mean {np.round(mean,2)} | std {np.round(std,2)} | cens {np.round(cens,2)}")
