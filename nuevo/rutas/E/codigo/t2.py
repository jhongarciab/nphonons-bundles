import numpy as np
from modelo import *; from purity import *
gx,gz=0.006,0.06; G=2*gx*gz
m=Full(gx,gz,N=8); E0=m.E[m.ig0]; wc=m.E[m.hyb].mean()-E0
for km,Om,kq in [(2.8*G,0.1*G,0.05*G),(6*G,0.1*G,0.05*G),(6*G,0.1*G,0.3*G),(1*G,0.1*G,0.05*G)]:
  H,c=m.build(Om,wc-2,km,kq); idx=m.lab.index('b1')
  r=cluster_stats(H,c,idx,5/km)
  He,ce=eff_model(gx,gz,N=6,Om=Om,kap_m=km,kap_q=kq); re=cluster_stats(He,ce,0,5/km)
  print(f"km/G={km/G} Om/G={Om/G} kq/G={kq/G}: FULL rate/G={r['rate']/G:.4f} P2={r['P2']:.4f} <k>={r['meank']:.3f} P_k={np.round(r['Pk'][:4],3)} | EFF rate/G={re['rate']/G:.4f} P2={re['P2']:.4f} <k>={re['meank']:.3f}")
