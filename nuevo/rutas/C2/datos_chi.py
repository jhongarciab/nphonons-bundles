import numpy as np, canal2
chis = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0); out = []
for c in chis:
    r, sep = canal2.directo('gauss', c, 0.0); out.append((c, r['s'], r['V'])); print('chi=%.2f s=%.3f V0=%.3f'%(c, r['s'], r['V']), flush=True)
np.save('chi_V0.npy', np.array(out))
