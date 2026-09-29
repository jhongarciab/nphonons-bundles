# -*- coding: utf-8 -*-
"""Tarea 29-bis (reformulada): compuerta Z de un qubit gato, tres esquemas. g=1, kappa_1=1e-3 g, |alpha|^2=alpha2.
 A: kappa2 D[a^2-alpha^2]
 B: A + TPE g_h (a^2-alpha^2) sigma_+ + h.c. (qubit sin perdida propia), g_h = g
 C: g (a^2-alpha^2) sigma_+ + h.c. + kappa D[sigma_-]   (kappa2_eff = 4 g^2/kappa se usa en A y B)
Drive de compuerta eps (a+a^dag), eps*t_g = pi/(4 alpha)  (rotacion Z de pi en el codigo: 2 alpha eps t = pi/2).
Errores: p_pf = (1+<P>)/2 partiendo de |C+> (ideal: C+ -> C-, <P> -> -1);
         p_bf = (1-<sgn x>)/2 partiendo de |alpha> (ideal: sin cambio).
Baseline sin drive (idle) a igual tiempo: p_pf_idle=(1-<P>)/2.
Uso: worker <kappa/g> <alpha2> <outfile>"""
import sys, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
from qutip import destroy, qeye, tensor, sigmam, basis, coherent, liouvillian, operator_to_vector, Qobj

kg, alpha2, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
KAPPA1 = 1e-3; g = 1.0; gh = 1.0
alpha = np.sqrt(alpha2); Nc = int(np.ceil(alpha2 + 6 * alpha)) + 4
kappa2_eff = 4 * g**2 / kg
TIMES = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

a1 = destroy(Nc); I1 = qeye(Nc)
# operadores logicos en el oscilador
xw, xv = np.linalg.eigh((a1 + a1.dag()).full())
Sgn = Qobj(xv @ np.diag(np.sign(xw)) @ xv.conj().T)
Par = Qobj(np.diag((-1.0) ** np.arange(Nc)))
ket_p = coherent(Nc, alpha); ket_m = coherent(Nc, -alpha)
Cp = (ket_p + ket_m).unit()


def sistema(scheme, eps):
    if scheme == 'A':
        a, ext = a1, (lambda O: O)
        core = a * a - alpha2 * I1
        H = eps * (a + a.dag())
        cops = [np.sqrt(kappa2_eff) * core, np.sqrt(KAPPA1) * a]
        return H, cops, (lambda ket: ket), (lambda O: O)
    a = tensor(a1, qeye(2)); sm = tensor(I1, sigmam()); core = a * a - alpha2 * tensor(I1, qeye(2))
    Hx = core * sm.dag(); Hx = Hx + Hx.dag()
    H = eps * (a + a.dag())
    g_ = gh if scheme == 'B' else g
    H = H + g_ * Hx
    cops = [np.sqrt(KAPPA1) * a]
    cops.append(np.sqrt(kappa2_eff) * core if scheme == 'B' else np.sqrt(kg) * sm)
    gnd = basis(2, 1)   # sigma_- = |1><0| -> |1> es el estado base
    return H, cops, (lambda ket: tensor(ket, gnd)), (lambda O: tensor(O, qeye(2)))


def evolve(scheme, eps, t, ket0, obs):
    H, cops, lift_ket, lift_op = sistema(scheme, eps)
    L = liouvillian(H, cops).data.tocsc()
    k = lift_ket(ket0); rho0 = operator_to_vector(k * k.dag()).full().ravel()
    rho = sla.expm_multiply(L * t, rho0)
    d = int(round(np.sqrt(len(rho)))); R = rho.reshape(d, d, order='F')
    return float(np.real(np.trace(lift_op(obs).full() @ R)))


res = {}
for sch in 'ABC':
    pf, bf, pf0, bf0 = [], [], [], []
    for t in TIMES:
        eps = np.pi / (4 * alpha * t)
        Pv = evolve(sch, eps, t, Cp, Par); Sv = evolve(sch, eps, t, ket_p, Sgn)
        Pi = evolve(sch, 0.0, t, Cp, Par); Si = evolve(sch, 0.0, t, ket_p, Sgn)
        pf.append((1 + Pv) / 2); bf.append((1 - Sv) / 2); pf0.append((1 - Pi) / 2); bf0.append((1 - Si) / 2)
    res[sch] = (pf, bf, pf0, bf0)
    print(f"kappa/g={kg:.4g} scheme {sch}: pf={np.round(pf, 5)} bf={np.round(bf, 6)}")
np.savez(out, kappa_over_g=kg, kappa2_eff=kappa2_eff, alpha2=alpha2, times=TIMES,
         **{f"{k}_{s}": np.array(v[i]) for s, v in res.items() for i, k in enumerate(["pf", "bf", "pf_idle", "bf_idle"])})
