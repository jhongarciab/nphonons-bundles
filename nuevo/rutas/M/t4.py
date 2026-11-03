import numpy as np, json, mem
from mem import *
h = h4(); lam = [.2,0,0,0]
res = {}
S = Sys(h, lam, slots=[0, 2], chi=1.0)
P = []
for k, g in ((0,(.22,191,0.0)),(1,(.20,182,0.0))):
    p, f = S.optimize(k, g); print('slot', k, np.round(p,4), 'closed transfer', round(f,4), flush=True); P.append(tuple(p))
json.dump(P, open('pulses4.json','w'))
for chi in (1.0, 0.0):
    S = Sys(h, lam, slots=[0, 2], chi=chi)
    r = S.run(2000., tuple(P)); print('4dot chi', chi, {k: round(v,4) for k,v in r.items()}, flush=True)
