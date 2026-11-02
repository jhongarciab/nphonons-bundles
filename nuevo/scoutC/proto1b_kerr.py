# Condición cerrada: |alpha|^2 = 2|eps|/sqrt(k2^2+4K^2) (Kerr rota la fase de alpha^2 pero conserva la variedad exacta)
import numpy as np, qutip as qt, scipy.sparse.linalg as sl
def rates(alpha2,K,nth,k1=0.02,k2=1.0,N=34):
    a=qt.destroy(N); ad=a.dag(); eps=alpha2*np.sqrt(k2**2+4*K**2)/2
    H=eps*(ad*ad+a*a)+K*ad*ad*a*a
    c=[np.sqrt(k2)*a*a,np.sqrt(k1*(1+nth))*a]+([np.sqrt(k1*nth)*ad] if nth>0 else [])
    L=qt.liouvillian(H,c).data.as_scipy().tocsc()
    ev=sl.eigs(L,k=5,sigma=1e-9,which='LM',return_eigenvectors=False); r=np.sort(-ev.real)
    return r[0],r[1],r[2]
print("alpha2=3, k1/k2=0.02: K/k2 nth | G0(~0) G_bitflip G_parity")
for K in [0,0.5,1.0,2.0]:
  for nth in [0,0.1]:
    g=rates(3.0,K,nth); print(f"{K:4.1f} {nth:4.2f} | {g[0]:.1e} {g[1]:.3e} {g[2]:.3e}",flush=True)
