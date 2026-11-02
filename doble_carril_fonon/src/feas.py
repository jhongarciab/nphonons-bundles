import numpy as np
rng=np.random.default_rng(1)
def ou_pair(N,steps,dt,tau,rho,aA=1.0,aB=1.0):
    """two OU processes unit variance, correlation rho, stationary start; return phase phi=∫(aA xA - aB xB)dt"""
    e=np.exp(-dt/tau); s=np.sqrt(1-e*e)
    c=np.sqrt(rho); d=np.sqrt(1-rho)
    z=rng.standard_normal(N); xA=c*z+d*rng.standard_normal(N); xB=c*z+d*rng.standard_normal(N)
    # common part evolves as one OU, private parts separately
    zc=z; pA=d*rng.standard_normal(N); pB=d*rng.standard_normal(N)
    phi=np.zeros(N)
    for _ in range(steps):
        phi+=(aA*(c*zc+pA)-aB*(c*zc+pB))*dt
        zc=e*zc+s*rng.standard_normal(N); pA=e*pA+s*d*rng.standard_normal(N); pB=e*pB+s*d*rng.standard_normal(N)
    return phi
def var_ana(T,tau,rho,sig,aA=1,aB=1):
    sd2=sig**2*(aA**2+aB**2-2*rho*aA*aB)
    return sd2*2*tau**2*(T/tau-1+np.exp(-T/tau))
print("1) check analytic <phi^2> vs MC (sigma=1, tau=1)")
for T,rho in [(0.5,0.9),(3,0.9),(10,0.95),(3,0.5)]:
    steps=600; dt=T/steps; phi=ou_pair(40000,steps,dt,1.0,rho)
    print(f" T={T:5} rho={rho}: MC var={phi.var():.4f}  ana={var_ana(T,1.0,rho,1.0):.4f}  C_MC={abs(np.exp(1j*phi).mean()):.4f}  C_gauss={np.exp(-var_ana(T,1,rho,1)/2):.4f}")
print("\n2) umbral 1-rho* para C>=0.9 (forma cerrada: 1-rho*=-lnC*/(2 tau^2 g), g=T/tau-1+e^{-T/tau}), sigma=1")
for tau in (0.1,1.0,10.0,1e3):
    row=[]
    for T in (0.1,0.3,1,3,10,30,100):
        g=T/tau-1+np.exp(-T/tau); x=-np.log(0.9)/(2*tau**2*g); row.append(f"T={T}:{x:.2g}")
    print(f" sigma*tau={tau}: 1-rho* "," | ".join(row))
print("\n3) non-Gaussian: n telegraph fluctuators (shared fraction=rho), same 2nd-order stats as Gaussian")
def tele(N,steps,dt,g,nsh,npr,amp):
    """each fluctuator ±amp switching rate g; shared ones act on A and B equally; private on one"""
    def evolve(n):
        s=rng.choice([-1,1],size=(n,N)); return s
    sh=evolve(nsh); pa=evolve(npr); pb=evolve(npr)
    phi=np.zeros(N); p=1-np.exp(-g*dt)  # flip prob per step
    for _ in range(steps):
        phi+=amp*(pa.sum(0)-pb.sum(0))*dt   # shared cancel exactly (equal coupling)
        for arr in (sh,pa,pb):
            if len(arr): arr*=np.where(rng.random(arr.shape)<p,-1,1)
    return phi
for (nsh,npr) in [(2,1),(1,1),(0,1),(8,1)]:
    ntot=nsh+npr; amp=1.0/np.sqrt(ntot)  # each ξ has variance 1
    rho=nsh/ntot
    for T,g in [(3,0.3),(3,3.0)]:
        steps=600;dt=T/steps; phi=tele(40000,steps,dt,g,nsh,npr,amp)
        # Gaussian prediction with same variance
        v=phi.var(); print(f" nsh={nsh} npr={npr} rho={rho:.2f} T={T} flip rate={g}: C_MC={abs(np.exp(1j*phi).mean()):.3f}  C_gauss(same var)={np.exp(-v/2):.3f}  var={v:.3f}")
