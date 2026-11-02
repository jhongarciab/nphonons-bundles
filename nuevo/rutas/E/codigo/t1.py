import numpy as np, time
from modelo import *
gx,gz=0.006,0.06; G=2*gx*gz
m=Full(gx,gz,N=10); print('wq*',m.wq,'G',G,'hyb E',m.E[m.hyb]-1, 'ig0',m.ig0)
E0=m.E[m.ig0]; print('E g0',E0, 'nu hyb',m.nu[m.hyb])
Eh=m.E[m.hyb]; wc=(Eh.mean()-E0); print('freq g0->hyb centro',wc,' -> delta=',wc-2)
t=time.time()
for km in [1.0*G,2.8*G,6*G]:
  for Om in [0.1*G,0.3*G]:
    rho,H,c=m.ss(Om,wc-2,km,0.05*G)
    s=m.stats(rho,c); print(f"km/G={km/G:.1f} Om/G={Om/G} J1/G={s['J1']/G:.4f} rates/G={ {k:round(v/G,5) for k,v in s['rate'].items()} } g2={s['g2']:.3f} n={s['n']:.4f} pe={s['pe']:.4f}")
print(time.time()-t)
