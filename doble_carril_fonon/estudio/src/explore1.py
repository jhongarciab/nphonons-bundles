import numpy as np, sys, time
from tls_model import *
data=np.load('../phi_hitchcock_fig4e.npy')  # cols: detuning, mean, +err, -err
det=data[:,0]
def run(p,nd=60,seed=1,sel=(0,1,3,7,12,17)):
    rng=np.random.default_rng(seed); ds=det[list(sel)]
    out=np.array([phi_curve(rng,p,ds) for _ in range(nd)])   # (nd, ndet, 3)
    return ds,out
base=dict(nf=0.02,W=80.,glo=0.002,ghi=0.2,glo_rate=1e-3,ghi_rate=1e2,amp=1.0,fsr=3.29,profile='travelling')
if __name__=="__main__":
    t=time.time()
    for nf in (0.005,0.05,0.5):
        for ghi in (0.05,0.5):
            p=dict(base,nf=nf,ghi=ghi)
            ds,o=run(p,nd=40)
            T1=np.median(o[:,0,1])
            print(f"nf={nf} ghi={ghi}: single-mode T_phi median={T1:7.1f} us | Phi mean vs detuning:", " ".join(f"{d:.0f}:{np.mean(o[:,i,0]):.2f}" for i,d in enumerate(ds)), f"[{time.time()-t:.0f}s]")
    print("DATA:", " ".join(f"{d:.0f}:{m:.2f}" for d,m in zip(det[[0,1,3,7,12,17]],data[[0,1,3,7,12,17],1])))
