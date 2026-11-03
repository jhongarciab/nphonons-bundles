import numpy as np
from scipy.optimize import minimize_scalar
def ops(N):
    a=np.diag(np.sqrt(np.arange(1,N)),1); I2=np.eye(2); IN=np.eye(N)
    sz=np.diag([1.,-1.]); sx=np.array([[0,1.],[1,0]])   # basis (e,g)
    return np.kron(I2,a),np.kron(sz,IN),np.kron(sx,IN),np.kron(I2,IN)
def H(wq,gx,gz,N):
    a,sz,sx,I=ops(N); X=a+a.T
    return a.T@a+0.5*wq*sz+X@(gx*sx+gz*sz)
def idx(N): # |e,0>,|g,2>
    return 0*N+0, 1*N+2
def gap(wq,gx,gz,N,ret=False):
    E,V=np.linalg.eigh(H(wq,gx,gz,N)); i,j=idx(N)
    w=V[i]**2+V[j]**2; k=np.argsort(w)[-2:]; k.sort()
    return (E[k[1]]-E[k[0]], w[k].sum()) if ret else E[k[1]]-E[k[0]]
def shift_pred(gx,gz,N=40):
    # 2nd order in gx, exact in gz (polaron basis): resonance shift = Dg - De
    H0=H(2.0,0,gz,N); E,V=np.linalg.eigh(H0); i,j=idx(N)
    a,sz,sx,I=ops(N); Vp=gx*sx@(a+a.T)
    def lab(s): return np.argmax(np.abs(V[s])) if False else np.argmax(V[s]**2)
    ke=np.argmax(V[i]**2); kg=np.argmax(V[j]**2)
    Vm=V.T@Vp@V; D=[]
    for k in (ke,kg):
        s=0
        for m in range(len(E)):
            if m in (ke,kg): continue
            s+=Vm[m,k]**2/(E[k]-E[m])
        D.append(s)
    return D[1]-D[0]
if __name__=="__main__":
    print("gx gz | G_num/(2gxgz) [N=30] [N=45] | e^{-2lam^2} | dwq*  num | -4gx^2 | pred2nd | ratio")
    for gx,gz in [(0.006,0.06),(0.01,0.1),(0.02,0.1),(0.03,0.15),(0.05,0.1),(0.03,0.3)]:
        out=[]
        for N in (30,45):
            f=lambda w:gap(w,gx,gz,N)
            r=minimize_scalar(f,bounds=(2-8*gx**2-0.1*gx*gz*0-20*gx*gz,2+4*gx**2+20*gx*gz),method='bounded',options={'xatol':1e-12})
            g,wt=gap(r.x,gx,gz,N,True); out.append((r.x,g,wt))
        wq,g,wt=out[1]; G=g/(2*np.sqrt(2))
        print(f"{gx} {gz} | {out[0][1]/(2*np.sqrt(2))/(2*gx*gz):.6f} {G/(2*gx*gz):.6f} | {np.exp(-2*gz**2):.6f} | {wq-2:.3e} | {-4*gx**2:.3e} | {shift_pred(gx,gz):.3e} | {(wq-2)/(-4*gx**2):.4f}  wt={wt:.4f}")
