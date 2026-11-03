"""Reproduce el metodo de bsc (pureza de Munoz, maestra LOCAL) para el emisor unico y la molecula, y compara con el MISMO lambda."""
import numpy as np, qutip as qt, sys
from functools import lru_cache
wb=1.0; gam=2e-4; gph=4e-4; Jf=0.5
def ops(nqd, Nc):
    b=qt.destroy(Nc); Ib=qt.qeye(Nc); sm=qt.sigmam(); I2=qt.qeye(2)
    if nqd==1:
        return dict(b=qt.tensor(I2,b), nb=qt.tensor(I2,b.dag()*b), sm=[qt.tensor(sm,Ib)], nqd=1)
    return dict(b=qt.tensor(I2,I2,b), nb=qt.tensor(I2,I2,b.dag()*b), sm=[qt.tensor(sm,I2,Ib), qt.tensor(I2,sm,Ib)], nqd=2)
cache={}
def get(nqd,Nc):
    if (nqd,Nc) not in cache: cache[(nqd,Nc)]=ops(nqd,Nc)
    return cache[(nqd,Nc)]
def H_lab(o, lam, Om, Delta, J):
    ne=sum(s.dag()*s for s in o['sm']); b=o['b']
    H=o['nb']+Delta*ne+lam*ne*(b+b.dag())+Om*sum(s+s.dag() for s in o['sm'])
    if o['nqd']==2: H=H+J*(o['sm'][0].dag()*o['sm'][1]+o['sm'][1].dag()*o['sm'][0])
    return H
def cops_local(o,kap):
    c=[np.sqrt(kap)*o['b']]+[np.sqrt(gam)*s for s in o['sm']]+[np.sqrt(gph)*s.dag()*s for s in o['sm']]; return c
def nmean(nqd,Nc,lam,Om,Delta,kap,J):
    o=get(nqd,Nc); r=qt.steadystate(H_lab(o,lam,Om,Delta,J),cops_local(o,kap)); return float(np.real(qt.expect(o['nb'],r)))
def pi_n(nqd,n,lam,Om,Delta,kap,J,Nfull=14):
    nf=nmean(nqd,Nfull,lam,Om,Delta,kap,J); nt=nmean(nqd,n,lam,Om,Delta,kap,J)
    return max(nf-nt,0)/nf if nf>1e-30 else 0.0
def center(nqd,n,lam,Om):
    f=2.0 if nqd==1 else 8.0
    arg=n*n-f*Om**2; D=-np.sqrt(arg)+lam**2
    return D-(Jf if nqd==2 else 0.0)
def opt(nqd,n,lam,Om,kap,J=Jf,nD=10,win=0.2):
    c=center(nqd,n,lam,Om); best=(-1,None)
    for D in np.linspace(c-win,c+win,nD):
        p=pi_n(nqd,n,lam,Om,D,kap,J)
        if p>best[0]: best=(p,D)
    return best
if __name__=='__main__':
    n=3; Om=0.2
    print('Reproduccion bsc, n=3, Om=0.2, maestra local, pureza de Munoz (barrido de Delta 10 puntos +-0.2)')
    print('%-6s %-10s | %-14s %-18s %-14s'%('lam','kappa(Q)','single Om=.2','single Om=.2*sqrt2','molecula J=.5'))
    for lam in (0.22,0.14):
        for kap in (0.0011,0.002,0.0077,0.011,0.03):
            s1=opt(1,n,lam,Om,kap)[0]; s2=opt(1,n,lam,Om*np.sqrt(2),kap)[0]; m=opt(2,n,lam,Om,kap)[0]
            print('%-6.2f %-6.4f(Q%-4d) | %-14.3f %-18.3f %-14.3f'%(lam,kap,round(1/kap),s1,s2,m),flush=True)
