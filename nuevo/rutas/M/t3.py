import numpy as np, json, mem
from mem import *
h = h4(); w, V = np.linalg.eigh(h); print('E', np.round(w,3)); print('pesos', np.round(V**2,3).T)
lam = [.2,0,0,0]
S = Sys(h, lam, slots=[0, 1], chi=1.0)   # D de -0.34 y -0.30 ? se elige abajo
for idx in range(3):
    D = V[:, idx]; print(idx, w[idx], 'g=', round(float(S.B@S.Lam@D),4), 'diag', round(float(D@S.Lam@D),4))
