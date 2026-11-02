import numpy as np
def cf(d,g,t):
    d=np.asarray(d,float);t=np.asarray(t,float)
    mu=np.sqrt((g*g-d*d).astype(complex))
    with np.errstate(all='ignore'):
        r=np.exp(-g*t)*(np.cosh(mu*t)+np.where(np.abs(mu)<1e-12,g*t,g*np.sinh(mu*t)/mu))
    return r.real
def integrals(g,tgrid,n,rng):
    """exact ∫_0^t s for n stationary telegraph trajectories; jump prob g dt"""
    T=tgrid[-1];K=int(g*T+8*np.sqrt(g*T+1)+12)
    s0=rng.choice([-1.,1.],n)
    tau=np.concatenate([np.zeros((n,1)),np.cumsum(rng.exponential(1/g,(n,K)),1)],1)
    assert (tau[:,-1]>T).all()
    sg=s0[:,None]*(-1.)**np.arange(K)[None,:]
    out=np.empty((n,len(tgrid)))
    for i,t in enumerate(tgrid):
        seg=np.clip(t,tau[:,:-1],tau[:,1:])-tau[:,:-1]
        out[:,i]=(sg*seg).sum(1)
    return out
def mc(A1,A2,gam,tgrid,n,seed):
    rng=np.random.default_rng(seed)
    p1=0;p2=0
    for a1,a2,g in zip(A1,A2,gam):
        I=integrals(g,tgrid,n,rng);p1=p1+a1*I;p2=p2+a2*I
    f=lambda p:-np.log(np.abs(np.exp(-1j*p).mean(0)))
    return f(p1),f(p2),f(p1-p2),p1,p2
def closed(A1,A2,gam,t):
    f1=sum(-np.log(np.abs(cf(a,g,t))) for a,g in zip(A1,gam))
    f2=sum(-np.log(np.abs(cf(a,g,t))) for a,g in zip(A2,gam))
    fx=sum(-np.log(np.abs(cf(a-b,g,t))) for a,b,g in zip(A1,A2,gam))
    return f1,f2,fx
def cross(t,f,level=1.0):
    i=np.where(f>=level)[0]
    if len(i)==0: return np.nan
    i=i[0]
    if i==0: return t[0]
    return t[i-1]+(level-f[i-1])*(t[i]-t[i-1])/(f[i]-f[i-1])
