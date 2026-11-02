import time,numpy as np,warnings; warnings.filterwarnings('ignore')
from myjc import *
t=time.time()
for side in ['below','above']:
    jc=JC(6,3.3,nm=3,side=side); r=best_DL_sym(jc,3.0,6,3.3,side); print(side,r,time.time()-t)
    plat=plat_si(Delta=6); m=jc.metric(100,0.65,plat,r[0],3.0,0.01); print(m['Fc'],m['pp'],time.time()-t)
