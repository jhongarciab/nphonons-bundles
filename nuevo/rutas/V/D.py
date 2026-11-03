import numpy as np, sys, time
from mc import run
from C import sp_ops
def clusters(ev,B,tc,t0,T):
    times=[[] for _ in range(B)]
    for tr,t,k in ev:
        if k==0:
            for i in tr: times[i].append(t[0])
    sizes=[]
    for i in range(B):
        x=np.array(times[i]); x=x[x>t0]
        if x.size<2: continue
        cut=np.nonzero(np.diff(x)>=tc)[0]
        edges=np.concatenate(([-1],cut,[x.size-1]))
        sz=np.diff(edges)
        # descartar primer cluster (puede ser parcial por burn-in) y ultimo (censurado)
        sizes.extend(sz[1:-1] if len(sz)>2 else [])
    return np.array(sizes)
if __name__=="__main__":
    G=1.0; km=np.sqrt(2); N=7; a,sm=sp_ops(N); tc=5/km; x=km*tc
    Hef=G*(sm.T@a@a+sm@a.T@a.T)
    psi0=np.zeros(2*N,complex); psi0[N]=1   # |g,0>
    rng=np.random.default_rng(7)
    for kq in (0.05,1.4):
        for W in (0.05,0.15):
            B=1000; T=700.; t0=30.; t=time.time()
            Ls=[np.sqrt(km)*a,np.sqrt(kq)*sm,np.sqrt(W)*sm.T]
            ev=run(Hef,Ls,psi0,B,T,0.01,rng)
            sz=clusters(ev,B,tc,t0,T)
            n=len(sz); p2=np.mean(sz==2); se=np.sqrt(p2*(1-p2)/n)
            R=n/(B*(T-t0)); 
            fr=lambda k: np.mean(sz==k)
            print(f"kq={kq} W={W}: clusters={n} P2={p2:.4f}+-{se:.4f} P1={fr(1):.4f} P3={fr(3):.4f} P4+={np.mean(sz>=4):.4f} | claim 1-2e^-x={1-2*np.exp(-x):.4f} tanh(x/2)={np.tanh(x/2):.4f} | q_impl(vs1-2e^-x)={1-p2/(1-2*np.exp(-x)):.4f} | cluster rate={R:.4f} R*tc={R*tc:.3f}  ({time.time()-t:.0f}s)",flush=True)
