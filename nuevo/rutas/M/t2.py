import numpy as np, time, sys, mem
from mem import *
P = ((.197,170,-.02),(.29,150,-.025))
for chi in (1.0, 0.0):
    S3 = Sys(h3(), [.2, .14, 0], slots=[1, 0], chi=chi)
    t = time.time(); r = S3.run(2000., P, tol=1e-8); print('3dot chi', chi, {k: round(v,4) for k,v in r.items()}, round(time.time()-t,1),'s', flush=True)
