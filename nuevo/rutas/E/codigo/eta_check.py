# Verifica eta = P(par | |e,0>) = 1/(1+ kq/(4G^2)*(km+kq/2+2G^2/km))  (forma cerrada del 2x2 no hermitiano) frente al Liouvilliano
# efectivo (N=6) y al COMPLETO vestido (N=8): E[M]/2 con E[M]=int (tasa de la banda omega) dt desde |e,0>.
import numpy as np, qutip as qt
from modelo import Full; from purity import eff_model
def EM(H,c,idx,rho0,ncl):
    d=H.shape[0]; L=qt.liouvillian(H,c).full(); rss=qt.steadystate(H,c).full().reshape(-1,order='F')
    r0=rho0.full().reshape(-1,order='F'); b=-(r0-rss)
    A=np.vstack([L,np.eye(d).reshape(1,-1,order='F')]); bb=np.append(b,0)    # traza X = 0
    X=np.linalg.lstsq(A,bb,rcond=None)[0].reshape(d,d,order='F')
    C=c[idx].full(); return np.real(np.trace(C.conj().T@C@X))
gx,gz=0.006,0.06; G=2*gx*gz
m=Full(gx,gz,N=8)
print(" km/G  kq/G | eta_cerrada  eta_efectivo  eta_completo")
for km,kq in [(1.0,0.05),(1.414,0.05),(2.83,0.05),(6,0.05),(1.414,0.3),(3,1.4),(1.414,1.4)]:
    ec=1/(1+kq/4*(km+kq/2+2/km))
    He,ce=eff_model(gx,gz,N=6,Om=0,kap_m=km*G,kap_q=kq*G,JP=0)   # sin Purcell para comparar con la forma cerrada
    ee=EM(He,ce,0,qt.ket2dm(qt.tensor(qt.basis(2,0),qt.basis(6,0))),6)/2/G*G
    H,c=m.build(0,0,km*G,kq*G,J2=0,J3=0); Ve=m.V
    rho0=qt.Qobj(Ve.conj().T@qt.ket2dm(qt.tensor(qt.basis(2,0),qt.basis(8,0))).full()@Ve)
    ef=EM(H,c,m.lab.index('b1'),rho0,8)/2
    print(f"{km:5.3f} {kq:5.2f} | {ec:.5f}   {ee:.5f}   {ef:.5f}")
