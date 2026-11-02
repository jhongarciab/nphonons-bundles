"""Bloqueo de fonón convencional con el átomo JC híbrido (modelo reducido RWA): átomo {|0>,|->,|+>}, repuesto óptico 0<->-, bombeo mecánico F sobre el modo.
Marco: |->:E-wL, |+>: +S-wL-wd... resonancia JC (S=wm=1). Se barre wd."""
import numpy as np, qutip as qt, sys
def run(F,wd,g=0.07,Or=0.02,gam=2e-4,gph=4e-4,kap=0.003,Nb=14,S=1.0,wL=0.0,drive_atom=True):
    k=qt.basis(3,0); m=qt.basis(3,1); p=qt.basis(3,2)
    Pm=m*m.dag(); Pp=p*p.dag(); sig0m=k*m.dag(); tau=p*m.dag()
    I3=qt.qeye(3); Ib=qt.qeye(Nb); b=qt.tensor(I3,qt.destroy(Nb)); E=lambda o:qt.tensor(o,Ib)
    # marco: |-> a wL (laser resonante con 0->- si wL=0), |+> a wL+wd ; modo a wd
    H=E(Pm)*(0-wL)+E(Pp)*(S-wL-(wd-1)*0)  # placeholder, corregido abajo
    dm=(0)-wL   # (E- - wL)
    H=dm*E(Pm)+(dm+S-wd)*E(Pp)+(1-wd)*b.dag()*b+g*(E(tau)*b+E(tau.dag())*b.dag())+Or*(E(sig0m)+E(sig0m.dag()))+F*(b+b.dag())
    c=[np.sqrt(gam)*E(sig0m),np.sqrt(gam)*E(k*p.dag()),np.sqrt(gph)*E(Pm),np.sqrt(gph)*E(Pp),np.sqrt(kap)*b]
    rho=qt.steadystate(H,c,method='direct',solver='spsolve')
    n=qt.expect(b.dag()*b,rho); m2=qt.expect(b.dag()**2*b**2,rho)
    P=np.real(np.diag(rho.ptrace(1).full()))
    return n,m2/n**2 if n>1e-12 else np.nan,P,qt.expect(E(Pm)+E(Pp),rho)
if __name__=='__main__':
    g=0.07
    print('## Bloqueo convencional (atomo JC repuesto): g2(0) del modo vs wd; F=0.002, kap=.003')
    for gam,Or in ((2e-4,0.003),(2e-4,0.02),(2e-3,0.02),(1e-2,0.05)):
        best=None
        for wd in np.linspace(1-1.4*g,1+1.4*g,29):
            n,g2,P,pa=run(0.002,wd,gam=gam,Or=Or)
            if n>1e-4 and (best is None or g2<best[2]): best=(wd,n,g2,pa)
        # referencia: modo sin átomo (g=0) -> g2=1
        print('gam=%g Or=%g : min g2=%.3f en wd-1=%.3f (g=%.3f) n=%.4f pop_atomo_excitado=%.2f'%(gam,Or,best[2],best[0]-1,g,best[1],best[3]),flush=True)
