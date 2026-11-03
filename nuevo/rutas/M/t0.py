import numpy as np, mem
from mem import *
S3 = Sys(h3(), [.2, .14, 0], slots=[0, 1], chi=1.0)   # autovalores ordenados: D(-0.6)=idx0, D(-0.35)=idx1
print('E', S3.w, 'slots', S3.E)
for k in (0, 1):
    D = S3.D[k]; print(k, 'g=', S3.B@S3.Lam@D, 'diag', D@S3.Lam@D, 'peso', D**2)
