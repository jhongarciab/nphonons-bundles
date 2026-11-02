# Comprobacion en el marco de LABORATORIO (bombeo sin RWA, H dependiente del tiempo, sin base vestida) vs modelo vestido-secular.
import numpy as np, qutip as qt, time
from modelo import *
gx,gz=0.01,0.1; G=2*gx*gz; N=8
m=Full(gx,gz,N=N); Om=0.3*G; km=2.8*G; kq=0.05*G
E0=m.E[m.ig0]; wc=m.E[m.hyb].mean()-E0; delta=wc-2
rho,H,c=m.ss(Om,delta,km,kq,Jdc=1.0,J2=1.0,J3=1.0); s=m.stats(rho,c)
print('vestido: J1/G',s['J1']/G,'n',s['n'],'pe',s['pe'])
a,sz,sx,sp=bare(N); H0=H0f(m.wq,gx,gz,N); wp=2+delta
Ht=[H0,[Om*sp,lambda t,args: np.exp(-1j*wp*t)],[Om*sp.dag(),lambda t,args: np.exp(1j*wp*t)]]
cl=[np.sqrt(km)*a,np.sqrt(kq)*sp.dag()]
T=4000.; tl=np.linspace(0,T,81)
rho0=qt.ket2dm(qt.tensor(qt.basis(2,1),qt.basis(N,0)))
t=time.time(); r=qt.mesolve(Ht,rho0,tl,cl,e_ops=[a.dag()*a,sp*sp.dag()],options={'atol':1e-9,'rtol':1e-7,'nsteps':10**6,'max_step':0.5})
n=np.array(r.expect[0]); print('lab: n(t) ultimos',n[-30:].mean(),'+-',n[-30:].std(),' pe',np.mean(r.expect[1][-30:]),'  tiempo',time.time()-t)
print('lab: J=km*n =',km*n[-30:].mean()/G,'G (el flujo del vestido b1 es solo banda omega)')
