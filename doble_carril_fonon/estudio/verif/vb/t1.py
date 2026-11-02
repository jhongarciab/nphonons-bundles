import numpy as np, qutip as qt
from mydisp import *
plat = plat_si(); 
# 1) sin ruido, sin transmon: conserva
psis = logical_states()
z = dict(T1=1e18, Tphi=1e18, T1q=1e18, T2q=1e18, kp=0.)
r = storage(psis[2], 100, 0.65, z); print('conserva', abs((psis[2].dag()*r*psis[2])))
# 2) solo perdida: p_pass ~ e^{-T/T1}: aqui sin check: traza del estado
p = dict(plat); 
a1 = qt.tensor(qt.destroy(2), qt.qeye(2)); a2 = qt.tensor(qt.qeye(2), qt.destroy(2))
p2 = dict(z); p2['T1']=534.; r = storage(psis[0], 100, 0.65, p2); 
print('pop |10>', (psis[0].dag()*r*psis[0]).real, np.exp(-100/534))
# 3) coherencia relativa
for rho in [0, .65, .95]:
    p3 = dict(Tphi=107., T1=534., kp=0.); T=100.
    r = storage(psis[2], T, rho, p3).full()
    coh = abs(r[1,2]); gam = 2/107.
    ana = 0.5*np.exp(-T/534)*np.exp(-T*gam*(1-rho))
    print('rho',rho,'|coh|',coh,'analitico',ana)
# 4) selectividad pulso pi: n=0 vs n=1, sin ruido, con chi; p(e) vs sigma
from mydisp import pulse_check
for (D,F) in [(6,3.3),(20,3.3),(20,1.0)]:
    c1,c2 = chis(D,F)
    out=[]
    for s in [1,3,8]:
        pe = []
        for st in [qt.fock_dm(2,0), qt.fock_dm(2,1)]:
            pass
        e00 = pulse_check(qt.tensor(qt.fock_dm(2,0),qt.fock_dm(2,0)), s,c1,c2,0,plat,0.,noise=False)[1]
        e10 = pulse_check(qt.tensor(qt.fock_dm(2,1),qt.fock_dm(2,0)), s,c1,c2,0,plat,0.,noise=False)[1]
        e01 = pulse_check(qt.tensor(qt.fock_dm(2,0),qt.fock_dm(2,1)), s,c1,c2,0,plat,0.,noise=False)[1]
        out.append((s, round(e00,4), round(e10,4), round(e01,4)))
    print(D,F,'chi/2pi',c1/2/np.pi,c2/2/np.pi,out)
