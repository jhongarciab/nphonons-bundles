"""Implementacion independiente (VB) del modelo dispersivo. Unidades: us, rad/us."""
import numpy as np, qutip as qt
from math import pi
SI = dict(T1=534., Tphi=107., T1q=100., T2q=150.)
def plat_si(g=1.1, alpha=185., Delta=6.):
    p = dict(SI); p['kp'] = (g/Delta)**2/p['T1q']; return p
def chis(Delta, FSR, g=1.1, alpha=185.):
    c = lambda D: 2*pi*(g**2/D)*alpha/(D+alpha)
    return c(Delta), c(Delta+FSR)

def logical_states():
    """vectores en base fonon (n1,n2): indices  |00>,|01>,|10>,|11> via tensor(m1,m2)"""
    k10 = qt.tensor(qt.basis(2,1), qt.basis(2,0)); k01 = qt.tensor(qt.basis(2,0), qt.basis(2,1))
    s = 1/np.sqrt(2)
    return [k10, k01, s*(k10+k01), s*(k10-k01), s*(k10+1j*k01), s*(k10-1j*k01)]

def phonon_cops(plat, rho, a1, a2, n1, n2):
    kap = 1/plat['T1'] + plat['kp']; gam = 2/plat['Tphi']
    return [np.sqrt(kap)*a1, np.sqrt(kap)*a2,
            np.sqrt(gam*(1+rho)/2)*(n1+n2), np.sqrt(gam*(1-rho)/2)*(n1-n2)]

def storage(psi, T, rho, plat, loss=True, deph=True):
    a1 = qt.tensor(qt.destroy(2), qt.qeye(2)); a2 = qt.tensor(qt.qeye(2), qt.destroy(2))
    c = phonon_cops(plat, rho, a1, a2, a1.dag()*a1, a2.dag()*a2)
    if not loss: c = c[2:]
    if not deph: c = c[:2]
    H = 0*a1
    if T == 0: return psi*psi.dag()
    return qt.mesolve(H, psi*psi.dag(), [0, T], c, options=dict(atol=1e-10, rtol=1e-9)).states[-1]

def pulse_check(rho_ph, sigma, chi1, chi2, rho, plat, eps, noise=True, loss=True):
    """rho_ph: operador 4x4 (m1 x m2) -> (rho_ph_unnorm_pass, p_pass)"""
    q = qt.tensor(qt.destroy(2), qt.qeye(2), qt.qeye(2))
    a1 = qt.tensor(qt.qeye(2), qt.destroy(2), qt.qeye(2)); a2 = qt.tensor(qt.qeye(2), qt.qeye(2), qt.destroy(2))
    n1, n2 = a1.dag()*a1, a2.dag()*a2
    ee = q.dag()*q; gg = 1-ee
    cbar = (chi1+chi2)/2
    H0 = (cbar - chi1*n1 - chi2*n2)*ee
    Hd = 0.5*(q + q.dag())
    O0 = pi/(sigma*np.sqrt(2*pi)); tp = 4*sigma; t0 = tp/2
    cof = lambda t, args=None: O0*np.exp(-(t-t0)**2/(2*sigma**2))
    c = []
    if noise:
        c = phonon_cops(plat, rho, a1, a2, n1, n2)
        if not loss: c = c[2:]
        sz = 2*ee - 1; gq = 1/plat['T2q'] - 1/(2*plat['T1q'])
        c += [np.sqrt(1/plat['T1q'])*q, np.sqrt(gq/2)*sz]
    r0 = qt.tensor(qt.fock_dm(2,0), rho_ph)
    r = qt.mesolve([H0, [Hd, cof]], r0, [0, tp], c, options=dict(atol=1e-10, rtol=1e-9, nsteps=100000)).states[-1]
    pe = qt.ptrace(r, [1,2]).full()*0  # placeholder
    rg = qt.tensor(qt.basis(2,0).dag(), qt.qeye(4)) if False else None
    R = r.full().reshape(2,4,2,4)
    rgg, ree = R[0,:,0,:], R[1,:,1,:]
    un = (1-eps)*ree + eps*rgg
    return un, np.trace(un).real

def Fc_from(rho_un_list, psis):
    """fidelidad media max sobre fase Z comun; rho_un_list sin normalizar -> normalizadas"""
    # base ordering (n1,n2): idx 0=|00>,1=|01>,2=|10>,3=|11>
    def F(phi):
        Rz = np.diag([1, np.exp(1j*phi), 1, 1])
        f = 0
        for rho, psi in zip(rho_un_list, psis):
            v = Rz.conj().T @ psi.full().ravel()   # <psi|Rz rho Rz^+|psi>
            f += (v.conj() @ rho @ v).real
        return f/len(psis)
    F0, F1, F2 = F(0), F(pi/2), F(pi)
    a0 = (F0+F2)/2; a1 = (F0-F2)/2; a2 = F1-a0
    Fm = a0 + np.hypot(a1, a2)
    # chequeo malla
    g = max(F(p) for p in np.linspace(0, 2*pi, 73))
    assert Fm >= g-1e-9
    return Fm, F0

def run_my(T, rho, plat, sigma, chi1, chi2, eps, use_check=True):
    psis = logical_states()
    outs, pp = [], []
    for psi in psis:
        r1 = storage(psi, T, rho, plat)
        if use_check:
            un, p = pulse_check(r1, sigma, chi1, chi2, rho, plat, eps)
        else:
            un, p = r1.full(), 1.0
        outs.append(un/p); pp.append(p)
    # p_pass media sobre los 6 estados
    Fm, F0 = Fc_from(outs, psis)
    return dict(Fc=Fm, F_noZ=F0, pp=float(np.mean(pp)), pps=pp)
