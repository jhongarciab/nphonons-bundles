import numpy as np, qutip as qt
from bl import *
lam,Om,kap,gam,gph=0.14,0.1,0.003,2e-5,4e-4;n=2;N=9
H,c,L,Jb,Jg=build_pol(-n+eps_n(lam,Om,n),lam,Om,kap,gam,gph,N)
rho=qt.steadystate(H,c)
d=np.real(np.diag(rho.full())).reshape(2,N)  # tensor(qubit,fock): index = q*N+m
print('g',np.round(d[0,:6],5)); print('x',np.round(d[1,:6],5))
