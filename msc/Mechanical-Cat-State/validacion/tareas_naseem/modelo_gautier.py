# -*- coding: utf-8 -*-
"""Tarea 44: modelo efectivo con buffer de dos niveles explicito, baño termico.
H = g2[(a^2-alpha^2) s+ + h.c.], kappa(1+nth) D[s-] + kappa nth D[s+] (+ kappa1 D[a] opcional). kappa=1."""
import numpy as np
from qutip import destroy, qeye, tensor, sigmam, liouvillian
import modelo_ladder as ml


def build(g2, alpha2, nth, kappa1, N=None):
    N = int(np.ceil(alpha2 + 6 * np.sqrt(alpha2))) + 4 if N is None else N
    a = tensor(destroy(N), qeye(2)); sm = tensor(qeye(N), sigmam()); sp = sm.dag()
    c = a * a - alpha2 * tensor(qeye(N), qeye(2)); H = g2 * (c * sp + (c * sp).dag())
    cops = [np.sqrt(1 + nth) * sm]
    if nth > 0: cops.append(np.sqrt(nth) * sp)
    if kappa1 > 0: cops.append(np.sqrt(kappa1) * a)
    return liouvillian(H, cops), N
