# -*- coding: utf-8 -*-
"""Tarea 47(b): Fig. 2 de Liu et al. con su modelo efectivo (Ecs. 11 y 12), kappa=0. Unidades 2*pi*MHz=1."""
import numpy as np, qutip as qt
from qutip import destroy, qeye, tensor, sigmap, sigmam, sigmaz, basis, coherent, fock
nu, gam, epsp = 35.4, 16.0, 3.53
gx = gz = np.sqrt(2) * 10.0 / 4; geff = 4 * gx * gz / nu
N = 24; m = tensor(qeye(2), destroy(N)); sp = tensor(sigmap(), qeye(N)); sm = sp.dag(); sz = tensor(sigmaz(), qeye(N)); up = tensor(basis(2, 0) * basis(2, 0).dag(), qeye(N))
def run(eps_eff, gphi_over_gam, init_n, gts):
    H = 8 * gx**2 / (3 * nu) * (up + 2 * m.dag() * m * up) - geff * (m * m * sp + m.dag() * m.dag() * sm) + eps_eff * (sp + sm)
    c = [np.sqrt(gam) * sm]
    if gphi_over_gam > 0: c.append(np.sqrt(gphi_over_gam * gam / 2) * sz)           # (gamma_phi/4) L[sz] = (gamma_phi/2) D[sz]
    psi0 = tensor(basis(2, 1), fock(N, init_n))
    return qt.mesolve(H, psi0, np.asarray(gts) / gam, c, options={'atol': 1e-10, 'rtol': 1e-8, 'nsteps': 200000, 'progress_bar': False}).states
def fid(states, alpha, parity):
    kp, km = coherent(N, alpha).full()[:, 0], coherent(N, -alpha).full()[:, 0]; tgt = kp + parity * km; tgt = tgt / np.linalg.norm(tgt)
    out = []
    for s in states:
        r = s.full().reshape(2, N, 2, N); ro = np.einsum('iaib->ab', r); out.append(np.real(tgt.conj() @ ro @ tgt))
    return np.array(out)
gts = np.linspace(0, 45, 91); a1, a2 = np.sqrt(epsp / geff), np.sqrt((epsp / 2) / geff)
L = ["# Tarea 47(b) — Fig. 2 de Liu et al. con el modelo efectivo (Ecs. 11 y 12), κ=0\n",
     f"ν={nu}, g_x=g_z=√2G/4={gx:.4f}, g_eff=4g_xg_z/ν={geff:.4f} (paper: 1.41), ε_p={epsp}, γ={gam}. Eq. (11): H_eff=(8g_x²/3ν)(|↑⟩⟨↑|+2m†m|↑⟩⟨↑|)−g_eff(m²σ̃₊+h.c.)+ε_p(σ̃₊+σ̃₋); Eq. (12): γD[σ̃₋]+(γ_φ/2)D[σ̃_z].",
     f"Candidatos: α²=ε_p/g_eff={epsp/geff:.3f} (α={a1:.3f}; el paper dice α=1.58 en la Fig. 2) y α²=(ε_p/2)/g_eff={(epsp/2)/geff:.3f} (α={a2:.3f}).\n",
     "## Fidelidad F(γt) con el gato par (arranque |0⟩|↓⟩) y el impar (|1⟩|↓⟩), γ_φ/γ ∈ {0, 0.5, 1}\n",
     "| convención de la deriva en H_eff | objetivo | γ_φ/γ | F(γt=7.5) | F(15) | F(30) | F(45) | F final | P(αcorrecto) |", "|---|---|---|---|---|---|---|---|---|"]
res = {}
for lab, eps_eff in (("ε_p (Eq. 11 tal cual)", epsp), ("ε_p/2", epsp / 2)):
    for name, alpha in (("α=1.58 (ε_p/g_eff)", a1), ("α=1.12 ((ε_p/2)/g_eff)", a2)):
        for gp in (0.0, 0.5, 1.0):
            st = run(eps_eff, gp, 0, gts); F = fid(st, alpha, +1); res[lab, name, gp] = F
            idx = lambda g: int(np.argmin(abs(gts - g)))
            L.append(f"| {lab} | {name} | {gp} | {F[idx(7.5)]:.3f} | {F[idx(15)]:.3f} | {F[idx(30)]:.3f} | {F[idx(45)]:.3f} | {F[-1]:.3f} | — |")
L.append("")
# amplitud autoconsistente del estacionario con H_eff tal cual
st = run(epsp, 0.0, 0, np.array([0, 200.0])); r = st[-1].full().reshape(2, N, 2, N); ro = np.einsum('iaib->ab', r); a2num = np.trace(destroy(N).full() @ destroy(N).full() @ ro)
L.append(f"⟨m²⟩ del estacionario con H_eff tal cual (γt=200, γ_φ=0): {a2num.real:.3f}{a2num.imag:+.3f}i ⇒ |α|²≈{abs(a2num):.3f}. Con ε_p/2: ", )
st = run(epsp / 2, 0.0, 0, np.array([0, 200.0])); r = st[-1].full().reshape(2, N, 2, N); ro = np.einsum('iaib->ab', r); a2b = np.trace(destroy(N).full() @ destroy(N).full() @ ro)
L[-1] += f"{a2b.real:.3f}{a2b.imag:+.3f}i ⇒ |α|²≈{abs(a2b):.3f}."
# impar
st = run(epsp, 0.0, 1, gts); Fo = fid(st, a1, -1)
L += ["", f"Gato impar (|1⟩|↓⟩, H_eff tal cual, α=1.58, γ_φ=0): F(γt=7.5,15,30,45) = {Fo[int(np.argmin(abs(gts-7.5)))]:.3f}, {Fo[int(np.argmin(abs(gts-15)))]:.3f}, {Fo[int(np.argmin(abs(gts-30)))]:.3f}, {Fo[-1]:.3f}."]
open("tarea47b_fig2_resultados.md", "w").write("\n".join(L)); print("\n".join(L))
