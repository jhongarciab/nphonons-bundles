"""V3: tasa de decaimiento de la paridad (ω_p = 11.98, N = 22, serie estroboscópica t = nT_p de V2).
Ajuste lineal de ln(paridad) vs t en ventanas con P_c > 0.99. Predicción 2|α|²(Γ₁⁻ + Γ₁⁺)."""
import numpy as np
w, g, th, Gam = 6.0, 0.3, np.pi/4, 0.015
kap = 2*Gam; gx = -g*np.sin(th)
r = np.load('res_v2/wp11.9800_N22.npz')['reg']
Gt, Pc, a2, par = r[:, 0], r[:, 1], r[:, 3], r[:, 6]
t = Gt/Gam
def pred(al2, k4=True):
    c = kap**2/4 if k4 else 0
    return 2*al2*(gx**2*kap/(w**2+c) + gx**2*kap/(9*w**2+c))
print(f"P_c>0.99 desde Γt={Gt[Pc>0.99][0]:.1f}")
print(f"pred |α|²=4 con κ²/4: {pred(4):.6e}  sin κ²/4: {pred(4,False):.6e}  (dif rel {pred(4)/pred(4,False)-1:.1e})")
print(f"pred |α|²=|⟨a²⟩|={a2[-1]:.4f}: {pred(a2[-1]):.6e}")
print("ventana Γt      tasa medida     /pred(κ²/4)  /pred(sin)  /pred(|a2|)  resid.max ln")
for lo, hi in [(26,152),(30,300),(60,300),(26,100),(100,200),(150,300)]:
    m = (Gt>=lo)&(Gt<=hi)&(Pc>0.99)
    p = np.polyfit(t[m], np.log(par[m]), 1); k = -p[0]
    res = np.log(par[m]) - np.polyval(p, t[m])
    print(f"[{lo:3d},{hi:3d}]  {k:.6e}   {k/pred(4):.4f}      {k/pred(4,False):.4f}     {k/pred(a2[-1]):.4f}      {abs(res).max():.1e}")

# ajuste con piso: par = A e^{-k t} + c  (la paridad tardía no decae a 0 puro)
from scipy.optimize import curve_fit
f = lambda t, A, k, c: A*np.exp(-k*t) + c
print("\najuste A e^{-kt}+c:   ventana   k            /pred(κ²/4)  c")
for lo, hi in [(26,152),(26,300),(60,300)]:
    m = (Gt>=lo)&(Gt<=hi)&(Pc>0.99)
    q, cv = curve_fit(f, t[m], par[m], p0=(0.5, 3.3e-4, 0), maxfev=20000)
    print(f"             [{lo:3d},{hi:3d}]  {q[1]:.6e}  {q[1]/pred(4):.4f}      {q[2]:+.2e} ± {np.sqrt(cv[2,2]):.1e}")
print("\nparidad tardía:", [(round(x,0), f"{y:+.2e}") for x, y in zip(Gt[::60], par[::60])][-6:])
