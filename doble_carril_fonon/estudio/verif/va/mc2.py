import numpy as np
BIG=1e7
def integrals(g,tgrid,n,rng,chunk=1000):
    T=tgrid[-1];K=int(g*T+8*np.sqrt(g*T+1)+12)
    out=np.empty((n,len(tgrid)))
    for a in range(0,n,chunk):
        m=min(chunk,n-a)
        s0=rng.choice([-1.,1.],m)
        tau=np.concatenate([np.zeros((m,1)),np.cumsum(rng.exponential(1/g,(m,K)),1)],1)
        assert (tau[:,-1]>T).all()
        sg=s0[:,None]*(-1.)**np.arange(K)[None,:]
        c=np.concatenate([np.zeros((m,1)),np.cumsum(sg*np.diff(tau,axis=1),1)],1)
        rows=np.arange(m)[:,None]*BIG
        flat=(tau+rows).ravel()
        for i,t in enumerate(tgrid):
            idx=np.searchsorted(flat,np.arange(m)*BIG+t,'right')-1
            r=idx//(K+1);l=idx%(K+1)
            out[a:a+m,i]=c[r,l]+sg[r,np.minimum(l,K-1)]*(t-tau[r,l])
    return out
def mc(A1,A2,gam,tgrid,n,seed):
    rng=np.random.default_rng(seed);p1=0;p2=0
    for a1,a2,g in zip(A1,A2,gam):
        I=integrals(g,tgrid,n,rng);p1=p1+a1*I;p2=p2+a2*I
    f=lambda p:-np.log(np.abs(np.exp(-1j*p).mean(0)))
    return f(p1),f(p2),f(p1-p2)
