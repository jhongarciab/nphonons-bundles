"""Verificación 2: (a) corrimiento de luz con polarón, (b) presupuesto de escritura, (c) correlación del campo de trampas lejanas."""
import numpy as np
from scipy import integrate
vB, vD1, vD2 = np.ones(3)/np.sqrt(3), np.array([0,1,-1])/np.sqrt(2), np.array([2,-1,-1])/np.sqrt(6)
lam=np.array([0.20,0.14,0.0]); V={"B":vB,"D1":vD1,"D2":vD2}; E={"B":0.0,"D1":-0.35,"D2":-0.60}
L=lambda a,b: V[a]@(lam*V[b])
print("(a) δ2* de primer orden vs con polarón (segundo orden en TODAS las |c,1>)")
for s,Om,sim in (("D1",0.197,-0.020),("D2",0.290,-0.025)):
    gap=-E[s]; dB=1+sim+gap; g=L("B",s)
    first=(g**2-Om**2)/dB
    # corrimiento de |D_s,0>: -Σ_c |Λ_cs|²/(ω_b + E_c - E_s), c∈{B,D1,D2}
    sh=sum(L(c,s)**2/(1+E[c]-E[s]) for c in ("B","D1","D2"))
    full=-Om**2/dB+sh
    print(f"   {s}: primer orden {first:.4f} | términos: B {L('B',s)**2/(1+E['B']-E[s]):.4f}, propio {L(s,s)**2/1.0:.4f}, otra oscura {L('D1' if s=='D2' else 'D2',s)**2/(1+E['D1' if s=='D2' else 'D2']-E[s]):.4f} | completo {full:.4f} | simulación {sim}")
print("(b) escritura bin 1: pD1=0.841 tras escribir; η_L=0.9613 -> η_w'=",0.841/0.9613)
Om,g,dB=0.197,L("B","D1"),1.33; Oe=Om*g/dB; d=170; k=1e-3
Pd=np.sin(Oe*d)**2; loss=k*(d/2+np.sin(2*Oe*d)/(4*Oe)); print(f"   sin²(Ω_eff d)={Pd:.4f} pérdida κ∫cos²={loss:.4f} → η_w≈{Pd*(1-loss):.4f}; admixtura (Ω/Δ)²={(Om/dB)**2:.4f}")
print("(c) K(d) para trampas fuera de una esfera de radio r0 (campo ∥ eje del par):")
def Gz(R): return R[2]/np.linalg.norm(R)**3
def corr(r0,d,n=200000,seed=0):
    # Monte Carlo en R>r0 con muestreo ∝ 1/R^4 (densidad radial): R = r0/u^{1/3}... usar importancia exacta: pdf(R)=3 r0^3/R^4
    rng=np.random.default_rng(seed); u=rng.random(n); R=r0/u**(1/3); c=rng.uniform(-1,1,n); ph=rng.uniform(0,2*np.pi,n); s=np.sqrt(1-c*c)
    pos=np.stack([R*s*np.cos(ph),R*s*np.sin(ph),R*c],1)
    f=lambda p,z: (p[:,2]-z)/np.linalg.norm(p-np.array([0,0,z]),axis=1)**3
    f1,f2=f(pos,+d/2),f(pos,-d/2)
    w=R**6   # volumen: 4πR²/pdf(R) ∝ R^6
    c11=np.mean(w*f1*f1); c12=np.mean(w*f1*f2); return 1-c12/c11
for ratio in (4,8,16):
    print(f"   r0/d={ratio}: 1-K MC={np.mean([corr(ratio*1.0,1.0,seed=s) for s in range(5)]):.5f}  teoría 0.4(d/r0)²={0.4/ratio**2:.5f}")
print("   integrales angulares: ∫(1-3c²)²dΩ=",2*np.pi*integrate.quad(lambda c:(1-3*c*c)**2,-1,1)[0]," 16π/5=",16*np.pi/5)
r,rp=6.98,9.67; print(f"   wᵀKw(triángulo, campo axial)={0.8*(4*r*r-rp*rp)/9:.3f} nm²/r0²")
