"""F11: relajación por el continuo de fonones LA entre los niveles de excitón de un triángulo de 3 puntos (regla de oro, T=0).
Γ(a->b) = 2π J(ω_ab) Σ_ij m_i m_j sinc(q d_ij),  m_i=<b|n_i|a>,  J(ω)=F² ω³/(4π² ħ ρ c⁵), F=D_e e^{-(q a_e)²/4} - D_h e^{-(q a_h)²/4}, q=ω/c.
Parámetros de GaAs de memoria (como golden_rule_continuum.py): incertidumbre ≈×3. Triángulo equilátero de lado d; la escala dipolar fija d desde J=gap_BD1/3.
Base: B=(1,1,1)/√3, D1=(0,1,-1)/√2, D2=(2,-1,-1)/√6. Gaps (E_B-E_D) en unidades de ω_b: B->D1: g1, B->D2: g2, D1->D2: g2-g1."""
import numpy as np
HB, EV = 1.054571817e-34, 1.602176634e-19
DE, DH, RHO, C = 7.0*EV, -3.5*EV, 5370.0, 5110.0
AE = 3e-9; AH = AE/1.15
V = {"B": np.ones(3)/np.sqrt(3), "D1": np.array([0, 1, -1])/np.sqrt(2), "D2": np.array([2, -1, -1])/np.sqrt(6)}
def gamma(a, b, w, d, scale=1.0):
    q = w/C; m = V[a]*V[b]
    F = DE*np.exp(-(q*AE)**2/4) - DH*np.exp(-(q*AH)**2/4)
    Jw = F**2*w**3/(4*np.pi**2*HB*RHO*C**5)
    s = 0.0
    for i in range(3):
        for j in range(3):
            x = q*d*(0 if i == j else 1)
            s += m[i]*m[j]*(1.0 if i == j else np.sin(x)/x)
    return 2*np.pi*Jw*s*scale
print("fb(GHz) g1 g2 | d(nm) | Γ(B->D1)/ωb  Γ(B->D2)/ωb  Γ(D1->D2)/ωb | Γs/Γc(B->D1)   [objetivo Γs≲3e-4; F usó Γc=1.5e-3]")
for fb in (50e9, 100e9, 150e9, 200e9):
    w0 = 2*np.pi*fb
    for g1, g2 in ((0.5, 1.0), (0.3, 0.6), (0.25, 0.5), (1.0, 1.5)):
        J = g1/3*w0*HB/EV*1e3  # meV
        # J(meV)->d: 0.13 meV a 6 nm, ∝ d^-3
        d = 6e-9*(0.13/ (g1/3*HB*w0/EV*1e3))**(1/3)
        G = {k: gamma(a, b, w*w0, d) for k, (a, b, w) in {"BD1": ("B", "D1", g1), "BD2": ("B", "D2", g2), "D1D2": ("D1", "D2", g2-g1)}.items()}
        print(f"{fb/1e9:5.0f} {g1:4.2f} {g2:4.2f} | {d*1e9:5.1f} | {G['BD1']/w0:11.2e} {G['BD2']/w0:12.2e} {G['D1D2']/w0:12.2e} | {G['D1D2']/G['BD1']:7.3f}")
