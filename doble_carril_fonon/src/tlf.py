import numpy as np
rng=np.random.default_rng(7)
def cf(delta,gam,T):
    """exact <exp(i*delta*∫s dt)> for telegraph s=±1, flip rate gam, stationary start"""
    w=np.sqrt((gam**2-delta**2).astype(complex))
    w=np.where(np.abs(w)<1e-12,1e-12,w)
    return 0.5*((1+gam/w)*np.exp((w-gam)*T)+(1-gam/w)*np.exp(-(w+gam)*T))
def var_phase(delta,gam,T):
    return delta**2*2*(T/(2*gam)-(1-np.exp(-2*gam*T))/(4*gam**2))
def device(geom,n,R,h,a,c0,T,gdec=6):
    """one device: Poisson TLFs in plane z=h above dot B (z=0); dot A at z=-geom['dz'], lateral offset geom['dx']"""
    N=rng.poisson(n*np.pi*R**2)
    r=R*np.sqrt(rng.random(N)); ph=2*np.pi*rng.random(N)
    x=r*np.cos(ph); y=r*np.sin(ph); z=np.full(N,h)
    def c(dx,dz):
        rx=x-dx; ry=y; rz=z-dz; rr=np.sqrt(rx**2+ry**2+rz**2); cos2=(rz/rr)**2
        return c0*(3*cos2-1)/(rr/a)**3
    cA=c(geom['dx'],-geom['dz']); cB=c(0.0,0.0)
    gam=10**(rng.uniform(-gdec/2,gdec/2,N))/T
    return cA,cB,gam
def stats(geom,n,R,h,a,c0,T,nd=600):
    CA2=CB2=CAB=0.0; Cex=[];Cg=[]
    for _ in range(nd):
        cA,cB,gam=device(geom,n,R,h,a,c0,T)
        d=cA-cB
        Cex.append(np.prod(cf(d,gam,T)).real); Cg.append(np.exp(-0.5*var_phase(d,gam,T).sum()))
        CA2+=(cA**2).sum();CB2+=(cB**2).sum();CAB+=(cA*cB).sum()
    rho=CAB/np.sqrt(CA2*CB2)
    return rho,np.array(Cex),np.array(Cg)
if __name__=="__main__":
    n=3e-4; R=1500.; h=40.; a=10.; T=1.0
    print("geometry | rho_ens | c0*T | median C exact | median C gauss | yield(C>=0.9) exact | yield gauss | frac devices |ΔlnC|>0.5")
    for name,geom in [("stacked 8nm",dict(dx=0.,dz=8.)),("lateral 30nm",dict(dx=30.,dz=0.)),("lateral 108nm",dict(dx=108.,dz=0.))]:
        for c0T in (30.0,300.0,3000.0):
            rho,Ce,Cg=stats(geom,n,R,h,a,c0T/T,T,nd=400)
            ok=(np.abs(np.log(np.clip(Ce,1e-6,1))-np.log(np.clip(Cg,1e-6,1)))>0.5).mean()
            print(f"{name:14s}| {rho:6.3f} | {c0T:5} | {np.median(Ce):.3f} | {np.median(Cg):.3f} | {(Ce>=0.9).mean():.2f} | {(Cg>=0.9).mean():.2f} | {ok:.2f}")
