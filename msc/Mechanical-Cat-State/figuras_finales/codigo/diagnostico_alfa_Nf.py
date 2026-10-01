"""(1) Cambio relativo de α_eff² (T = 0 → térmico) en el completo (al2eff de los .npz) y en el efectivo (⟨(a − d)²⟩ no se guarda: se usa |⟨a²⟩| del estado estacionario del efectivo),
en los 8 puntos de la tabla E_obs; efecto sobre E_obs si γ_pf ∝ α². (2) Efectivo κ_f/ω = 0.05, x = 6.86, κ₂/κ = 0.05, 0.1, 0.25, 0.4: γ_pf y γ_bf con N_f = 2 y 3 (T = 0 y térmico);
efecto en el cociente completo/efectivo de γ_bf. El completo sigue con N_f = 2. No se lanzan corridas del completo; el efectivo no guarda cachés aquí. Salida: data/diagnostico_alfa_Nf.txt"""
import os, sys, numpy as np, qutip as qt, scipy.sparse.linalg as sla
import comun as C
import calc_termico as CT
np.savez = lambda *a, **k: None
K = 0.03; FC = os.path.join(C.DATA, 'filtro_completo')
GX = {0.05: 0.0239579, 0.1: 0.0338816, 0.25: 0.0535714, 0.4: 0.0677631}; XM = {0.05: 9.0, 0.1: 9.0, 0.25: 10.1, 0.4: 9.5}
comp = lambda k, x=None: np.load(os.path.join(FC, f'gx{GX[k]:.6g}_w6_gz0.42_kf0.3_al4_N22_Nf2_gam6e-07' + (f'_x{x:g}' if x else '') + '.npz'))
def estado_efectivo(x, k2k, N=22, Nf=2, gzk=14.0, gam=2e-5, w=200.0, al2=4.0, kfw=0.05):
    nq = 1 / np.expm1(x); nm = 1 / np.expm1(x / 2)
    gx = w * np.sqrt(k2k / (16 * gzk**2)); G = 2 * gx * gzk / w; Om = al2 * G; chi = 8 * gx**2 / (3 * w); kf = kfw * w
    ke = lambda d: kf**2 / (4 * d**2 + kf**2); Gm = gx**2 * ke(w) / w**2; Gp = gx**2 * ke(3 * w) / (9 * w**2)
    I = [qt.qeye(N), qt.qeye(2), qt.qeye(Nf)]; op = lambda k, o: qt.tensor(*[o if i == k else I[i] for i in range(3)])
    a, sm = op(0, qt.destroy(N)), op(1, qt.sigmam()); Pe = sm.dag() * sm; Pg = 1 - Pe; b = op(2, qt.destroy(Nf)); J = np.sqrt(kf) / 2
    H = chi * a.dag() * a * Pe + (gx**2 / w) * (Pe - Pg / 3) - G * (sm.dag() * a * a + sm * a.dag() * a.dag()) + Om * (sm.dag() + sm) + J * (sm.dag() * b + sm * b.dag())
    c = [np.sqrt(kf * (nq + 1)) * b, np.sqrt(kf * nq) * b.dag(), np.sqrt(Gm * (nq + 1) + Gp * nq + gam * (nm + 1)) * a, np.sqrt(Gm * nq + Gp * (nq + 1) + gam * nm) * a.dag()]
    L = qt.liouvillian(H, c).data.as_scipy().tocsc(); lam, R = sla.eigs(L, k=1, sigma=-1e-10, which='LM', tol=1e-14, maxiter=100000)
    D = 2 * N * Nf; M = R[:, 0].reshape(D, D, order='F'); M = M / np.trace(M); M = (M + M.conj().T) / 2
    A = a.full(); return abs(np.trace(A @ A @ M))
out = ['(1) α_eff² T = 0 → térmico. Completo: al2eff (parte real) de los .npz; efectivo: |⟨a²⟩| del estado estacionario. E_obs con γ_pf ∝ α²: se corrige r_x → r_x·(α²_T0/α²_x) (el factor que sale si el aumento térmico de γ_pf del completo viniera solo del cambio de α²); también el efecto inverso.',
       'κ₂/κ   x      α²c(T0)   α²c(x)    Δα²c/α²   α²e(T0)   α²e(x)    Δα²e/α²   E_obs(γ_pf∝α²)  E_obs_orig     E_ansatz'] 
Eoa = {}
import io, contextlib
orig = open(os.path.join(C.DATA, 'diagnostico_E_obs.txt')).read().splitlines()[1:9]
cache_e = {}
for i, (k, x) in enumerate([(kk, xx) for kk in GX for xx in (6.86, XM[kk])]):
    z0, z = comp(k), comp(k, x); a0, ax = float(np.real(z0['al2eff'])), float(np.real(z['al2eff']))
    for xx in (60.0, x):
        if (k, xx) not in cache_e: cache_e[(k, xx)] = estado_efectivo(xx, k)
    e0, ex = cache_e[(k, 60.0)], cache_e[(k, x)]
    f = orig[i].split(); r0, rx, ge, Eo_, Ea_ = float(f[2]), float(f[3]), float(f[4]), float(f[5]), float(f[6])
    # si γ_pf ∝ α²: parte del cociente atribuible al cambio de α² entre T=0 y x: r_x^pred = r_0·(ax/a0)·(e0/ex) → E_pred = (r_pred − r_0)·γ_e; el resto E_obs − E_pred queda sin explicar
    rpred = r0 * (ax / a0) / (ex / e0); Epred = (rpred - r0) * ge
    out.append(f'{k:<6g} {x:<6g} {a0:.5f}  {ax:.5f}  {ax/a0-1:+.2e}  {e0:.5f}  {ex:.5f}  {ex/e0-1:+.2e}  E_pred(α²)={Epred:+.2e}  E_obs-E_pred={Eo_-Epred:+.3e}  E_obs={Eo_:+.3e}  E_ansatz={Ea_:+.3e}')
out.append('\n(2) Efectivo κ_f/ω = 0.05, x = 6.86: N_f = 2 y 3 (γ_pf, γ_bf en κ = 1; T = 0 = x 60)')
out.append('κ₂/κ   Nf  γ_pf(T0)     γ_pf(x)      γ_bf(T0)     γ_bf(x)      | γ_bf(x) completo  c/e (N_f=2)  c/e (efectivo N_f=3)  cambio  fuera de ±5%?')
for k in GX:
    z = comp(k, 6.86); bc = float(z['gbf']) / K; res = {}
    for nf in (2, 3):
        e0 = CT.punto(60.0, k, 14.0, 1, 2e-5, 22, Nf=nf, rerun=True); e1 = CT.punto(6.86, k, 14.0, 1, 2e-5, 22, Nf=nf, rerun=True)
        res[nf] = (float(e0['gpf']), float(e1['gpf']), float(e0['gbf']), float(e1['gbf']))
    for nf in (2, 3):
        r = res[nf]; ce = bc / res[nf][3]
        extra = f'| {bc:.4e}  {bc/res[2][3]:.4f}  {bc/res[3][3]:.4f}  {(res[2][3]/res[3][3]-1):+.2%}  {"sí" if abs(bc/res[3][3]-1) > 0.05 else "no"}' if nf == 3 else ''
        out.append(f'{k:<6g} {nf}   {r[0]:.6e} {r[1]:.6e} {r[2]:.5e} {r[3]:.5e} {extra}')
txt = '\n'.join(out); print(txt); open(os.path.join(C.DATA, 'diagnostico_alfa_Nf.txt'), 'w').write(txt + '\n')
