import numpy as np
from scipy.optimize import minimize_scalar
def HB(wq,al,g,w,K,M):
    E=np.array([k*wq+0.5*al*k*(k-1) for k in range(K)])
    sig=np.diag(np.sqrt(np.arange(1,K)),1)   # sum sqrt(k+1)|k><k+1|
    a=np.diag(np.sqrt(np.arange(1,M)),1)
    return np.kron(np.diag(E),np.eye(M))+w*np.kron(np.eye(K),a.T@a)+g*(np.kron(sig.T,a)+np.kron(sig,a.T))
def gapN(N,al,g,w,K=6,M=10):
    i=N*M+0; j=0*M+N     # |N,0>,|0,N>
    def gp(wq):
        E,V=np.linalg.eigh(HB(wq,al,g,w,K,M)); ww=V[i]**2+V[j]**2; k=np.sort(np.argsort(ww)[-2:])
        return E[k[1]]-E[k[0]]
    c=w-(N-1)*al/2
    r=minimize_scalar(gp,bounds=(c-0.15*abs(al),c+0.15*abs(al)),method='bounded',options={'xatol':1e-12})
    return gp(r.x)/2, r.x-c
if __name__=="__main__":
    w=2300.
    for g,al in [(15.2,-138.),(15.2,-300.),(5.,-250.),(15.2,-138.)]:
        V2,d=gapN(2,al,g,w,5,10); D=abs(al)/2
        Vex=(np.sqrt(D**2+16*g**2)-D)/4   # exact 3x3: 2V=(sqrt(D^2+16g^2)-D)/2
        V3,d3=gapN(3,al,g,w,5,10); V4,d4=gapN(4,al,g,w,5,10)
        print(f"g={g} al={al} g/|al|={g/abs(al):.3f}: V2={V2:.4f} 4g2/|al|={4*g*g/abs(al):.4f} ratio={V2/(4*g*g/abs(al)):.4f} exact3x3={Vex:.4f} (dwq*={d:.2e}) | V3={V3:.4e} V4={V4:.4e} V3/V2={V3/V2:.4f} V4/V3={V4/V3:.4f}")
