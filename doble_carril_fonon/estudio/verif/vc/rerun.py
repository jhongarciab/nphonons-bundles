import sys,json; sys.path.insert(0,'/home/user/nphonons-bundles/doble_carril_fonon/estudio/src')
import numpy as np, erasure_sim as E
from design_run import chik, plat
g,fsr,T1q,D,sg=3.0,3.3,300.,20,0.5
out={}
for conv,Tphi1 in (('code(107)',107.),('corr(214)',214.)):
  for T in (50.,100.):
    Tph=Tphi1/0.58
    P=plat(534.,Tph,g,D,T1q)
    r=E.run(T,0.0,P,sigma=sg,chis=(chik(g,D),chik(g,D+fsr)))
    b2=E.run(T,0.0,dict(T1=534.,Tphi=Tph,T1q=100.,T2q=150.,kp=0.0),use_check=False)
    b1=E.run(T,0.0,dict(T1=534.,Tphi=Tphi1,T1q=100.,T2q=150.,kp=0.0),use_check=False)
    # single rail Fock qubit, unheralded: analytic
    eta=np.exp(-T/534.); c=np.sqrt(eta)*np.exp(-T/Tphi1)  # coherence rate 1/Tphi (code conv)
    print(conv,T,'Fc %.3f p %.3f B2unc %.3f B2h %.3f B1unc %.3f'%(r['F_c'],r['p_pass'],b2['F_unc'],b2['F_h'],b1['F_unc']),flush=True)
