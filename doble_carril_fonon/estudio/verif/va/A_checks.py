import numpy as np
from mc import *
rng=np.random.default_rng(1)
# A1: direct MC and ODE vs closed form, limits
t=np.array([0.5,2.,5.])
for d,g in [(1.,0.3),(1.,3.),(2.,1.),(1.,1.)]:
    I=integrals(g,t,200000,rng); print('A1',d,g,np.exp(-1j*d*I).mean(0).real.round(4),cf(d,g,t).round(4))
print('gauss lim',cf(1.,200.,t),np.exp(-t/400)); print('static',cf(1.,1e-4,t),np.cos(t))
# A4 check on telegraph phases
I=integrals(1.,np.array([3.]),200000,rng)[:,0]*1.5
print('A4',abs(np.exp(-1j*I).mean()),1-0.5*(I**2).mean(),1-0.5*I.var())
# A2/A3 gaussian check: correlated gaussian phases
from numpy.random import default_rng
r=0.6;z=default_rng(2).standard_normal((2,10**6));p1=z[0];p2=r*z[0]+np.sqrt(1-r*r)*z[1]
print('A3 var diff',np.var(p1-p2),2*(1-r))
for n in (1,2,3):
    print('Phi n',n,1-(1-r)**(1/n))
