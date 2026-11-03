import numpy as np
from scipy.linalg import expm
def run(H,Ls,psi0,B,T,dt,rng,stop_dead=False):
    """Trayectorias (tiempo de espera, vectorizado). Devuelve lista (traj,tiempo,tipo)."""
    d=H.shape[0]; Heff=H-0.5j*sum(L.conj().T@L for L in Ls); U=expm(-1j*Heff*dt)
    psi=np.tile(psi0.reshape(-1,1),(1,B)).astype(complex); r=rng.random(B)
    ev=[]; t=0.0; alive=np.ones(B,bool)
    for s in range(int(T/dt)):
        psi=U@psi; t+=dt
        n2=(psi.real**2+psi.imag**2).sum(0); J=np.nonzero(n2<r)[0]
        if J.size:
            ph=psi[:,J]; w=np.array([(np.abs(L@ph)**2).sum(0) for L in Ls])  # (nL,nJ)
            tot=w.sum(0); u=rng.random(J.size)*tot; cs=np.cumsum(w,0); k=(u[None,:]>cs).sum(0); k=np.minimum(k,len(Ls)-1)
            for kk in range(len(Ls)):
                m=k==kk
                if m.any():
                    nv=Ls[kk]@ph[:,m]; nv/=np.sqrt((np.abs(nv)**2).sum(0)); psi[:,J[m]]=nv
                    ev.append((J[m],np.full(m.sum(),t),kk))
            r[J]=rng.random(J.size)
        if stop_dead and s%200==0 and (n2>0).sum()==0: break
    return ev
