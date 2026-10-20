"""Verificación numérica de las fórmulas analíticas de tex/memoria_teoria.tex."""
import numpy as np, itertools
np.set_printoptions(precision=5, suppress=True)
vB, vD1, vD2 = np.ones(3)/np.sqrt(3), np.array([0,1,-1])/np.sqrt(2), np.array([2,-1,-1])/np.sqrt(6)
g1, g2 = 0.35, 0.6
E = {"B":0.0,"D1":-g1,"D2":-g2}
h = sum(E[k]*np.outer(v,v) for k,v in (("B",vB),("D1",vD1),("D2",vD2)))
print("1) h =\n", h)
print("   eps1,eps2,J12,J23 fórmula:", -2*g2/3, -g1/2-g2/6, g2/3, g1/2-g2/6)
print("   autovalores:", np.linalg.eigvalsh(h), " filas suman:", h.sum(1))
print("   (r23/r12)=", (2*g2/(3*g1-g2))**(1/3))
lam = np.array([0.20,0.14,0.0])
print("2) g1,g2,<D1|L|D2>:", vB@(lam*vD1), vB@(lam*vD2), vD1@(lam*vD2), "fórmula:", (lam[1]-lam[2])/np.sqrt(6), (2*lam[0]-lam[1]-lam[2])/np.sqrt(18), -(lam[1]-lam[2])/np.sqrt(12))
print("   diag:", vB@(lam*vB), vD1@(lam*vD1), vD2@(lam*vD2), (lam.sum()/3,(lam[1]+lam[2])/2,(4*lam[0]+lam[1]+lam[2])/6))
# 3) Raman
for name,(Om,gs,gap,d2sim,tsim) in {"1":(0.197,vB@(lam*vD1),0.35,-0.02,170),"2":(0.29,vB@(lam*vD2),0.60,-0.025,150)}.items():
    dB = 1+d2sim+gap; Oeff=Om*gs/dB; tpi=np.pi/(2*Oeff); d2s=(gs**2-Om**2)/dB
    print(f"3) ranura {name}: Δ_I={dB:.3f} Ω_eff={Oeff:.5f} t_π={tpi:.1f} (sim {tsim}) δ2*={d2s:.3f} (sim {d2sim}) η_w≈{1-1e-3*tpi/2:.3f}")
# 4) tasas de ruido
n = [np.outer(v:=np.eye(3)[i], v) for i in range(3)]  # proyectores de sitio
def nab(a,b): return np.array([a[i]*b[i] for i in range(3)])
def W(a,b,K): m=nab(a,b); return m@K@m
for chi in (0.0,0.5,1.0):
    K=(1-chi)*np.eye(3)+chi*np.ones((3,3)); w=vD1**2-vD2**2
    W12,W1B,W2B=W(vD1,vD2,K),W(vD1,vB,K),W(vD2,vB,K)
    G=0.5*(w@K@w)+0.5*((W12+W1B)+(W12+W2B)); print(f"4) χ={chi}: W12={W12:.4f} W1B={W1B:.4f} W2B={W2B:.4f} |w|²={w@w:.4f} Γc/γφ={G:.4f}  fórmula (5/6)(1-χ)={5/6*(1-chi):.4f}")
for a in (0.1,0.2):
    al=np.array([1+a,1,1]); K=np.outer(al,al); w=vD1**2-vD2**2
    G=0.5*(w@K@w)+0.5*((W(vD1,vD2,K)+W(vD1,vB,K))+(W(vD1,vD2,K)+W(vD2,vB,K))); print(f"   α=(1+{a},1,1): Γ/γφ={G:.5f}  a²/3={a*a/3:.5f}")
# 5) S_ab
def S(a,b,q,r,rp):
    m=nab(a,b); d=np.array([[0,r,r],[r,0,rp],[r,rp,0]]); s=0
    for i in range(3):
        for j in range(3): s+=m[i]*m[j]*(1.0 if i==j else np.sinc(q*d[i,j]/np.pi))
    return s
q,r,rp=0.02,1.0,1.387
print("5) S pequeño qd: BD1",S(vB,vD1,q,r,rp),(q*rp)**2/18," D1D2",S(vD1,vD2,q,r,rp),(q*rp)**2/36," BD2",S(vB,vD2,q,r,rp),q*q/54*(4*r*r-rp*rp))
x,xp=0.7,0.97
print("   exactas: BD1",S(vB,vD1,1,x,xp),(1-np.sinc(xp/np.pi))/3," D1D2",S(vD1,vD2,1,x,xp),(1-np.sinc(xp/np.pi))/6," BD2",S(vB,vD2,1,x,xp),(6-8*np.sinc(x/np.pi)+2*np.sinc(xp/np.pi))/18)
# reescala de tasas: r12 y r23
d_eq=8.3; r12=6*(0.13/(0.2*0.4136))**(1/3); r23=r12*(2*g2/(3*g1-g2))**(1/3)
print(f"   r12={r12:.2f} nm r23={r23:.2f} nm; BD1 ×{(r23/d_eq)**2:.2f}, D1D2 ×{(r23/d_eq)**2:.2f}, BD2 ×{(4*r12**2-r23**2)/(3*d_eq**2):.2f}")
# 6) dos excitones
D=np.diag(h); O=h-np.diag(D); H2=np.trace(h)*np.eye(3)-np.diag(D)+O
ev=np.linalg.eigvalsh(H2); print("6) tr h =",np.trace(h)," autovalores H2 (sin Δxx):",ev)
wL2 = E["D2"]-1.0-(-0.025)  # ω_L2 = E_D2 - ω_b - δ2*(sim)
print("   ω_L2 =",wL2," Δxx_res=", [wL2-e+E['D1'] for e in ev], " (con δ2=-0.0254 teoría: ", [E['D2']-1+0.051-e+E['D1'] for e in ev],")")
# 7) eficiencias
eps=0.02; kT=4
etaL=(np.sqrt(1-eps)-np.exp(-kT)/np.sqrt(1-eps))**2/(1-np.exp(-kT)); etaR=(1-np.exp(-kT))**2/(1+eps-np.exp(-kT)); print("7) η_L=",etaL," η_R=",etaR)
# 8) T*
k=1e-3; tL=tE=400; eL1,eL2,eR=0.9613,0.9476,0.9621
C=0.25*(eL2*eR+eL1*eR*np.exp(-k*(tL+tE)))+0.5*eR*np.sqrt(eL1*eL2)*np.exp(-k*(tL+tE)/2)
for F0 in (0.58,0.60,0.607): print(f"8) C={C:.4f}  T*(F0={F0})={np.log(C/F0)/k:.1f}")
# 9) ley V(τ12)
tau=np.array([100,400,1000,2000,4000]); Fh=np.array([0.932,0.898,0.858,0.788,0.692]); V=2*Fh-1
print("9) V/V0:",np.round(V/V[0],3)," pred:",np.round(np.exp(-4e-4*(tau-100)/2),3), "  pendiente ajustada:", np.polyfit(tau,np.log(V),1)[0], "(esperado -2e-4)")
print("   tw=800: ",(2*0.862-1)/(2*0.932-1), np.exp(-4e-4*800/2))
