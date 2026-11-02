# Verifica G (acoplo efectivo de dos fonones) por diagonalizacion exacta del H completo sin bombeo. omega=1.
import numpy as np, qutip as qt, sys
from scipy.optimize import minimize_scalar
def H0(wq,gx,gz,N=14):
    a=qt.tensor(qt.qeye(2),qt.destroy(N)); sz=qt.tensor(qt.sigmaz(),qt.qeye(N)); sx=qt.tensor(qt.sigmax(),qt.qeye(N))
    # qutip sigmaz: |0>=up. definimos |e>=basis(2,0)
    return a.dag()*a+0.5*wq*sz+(a+a.dag())*(gx*sx+gz*sz), a
def gap(wq,gx,gz,N=14):
    H,a=H0(wq,gx,gz,N); E,V=np.linalg.eigh(H.full())
    # estados dominados por |e,0> y |g,2>
    N_=N; i_e0=0*N_+0; i_g2=1*N_+2   # indices: qubit e=0 (up), g=1
    w=np.abs(V[i_e0,:])**2+np.abs(V[i_g2,:])**2
    idx=np.argsort(w)[-2:]; return abs(E[idx[0]]-E[idx[1]]),E[idx]
for gx,gz in [(0.006,0.06),(0.01,0.1),(0.02,0.1),(0.03,0.15)]:
    r=minimize_scalar(lambda w:gap(w,gx,gz)[0],bounds=(1.97,2.01),method='bounded',options={'xatol':1e-9})
    g=r.fun; G_pred=2*gx*gz
    print(f"gx={gx} gz={gz} wq*={r.x:.6f} (wq-2={r.x-2:+.3e}; 4gx^2/3? {4*gx**2/3:.3e}) gap={g:.4e} => G_num=gap/(2*sqrt2)={g/(2*2**.5):.4e}  G_pred=2gxgz={G_pred:.4e} ratio={g/(2*2**.5)/G_pred:.4f}")
