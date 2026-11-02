"""Factibilidad: qubit de doble carril fononico (dos modos, un fonon) + transmon, chequeo de borrado por JPNS.
Unidades: microsegundos y rad/us. Espacio: transmon(2) x modo1(2) x modo2(2). Sector de un fonon (|10>,|01>,|00>).
Modelo dispersivo efectivo (no el JC completo). Ruido: perdida k=1/T1+Purcell, desfase de frecuencia correlacionado (coef. rho),
decoherencia del transmon (T1q,T2q). Lectura del transmon con error eps_ro."""
import numpy as np, qutip as qt, warnings
warnings.filterwarnings("ignore")
TP=2*np.pi
I2=qt.qeye(2)
q  = qt.tensor(qt.destroy(2),I2,I2)
a1 = qt.tensor(I2,qt.destroy(2),I2)
a2 = qt.tensor(I2,I2,qt.destroy(2))
n1,n2 = a1.dag()*a1, a2.dag()*a2
Pe = q.dag()*q
g0=qt.basis(2,0); 
def ket(m1,m2,e=0): return qt.tensor(qt.basis(2,e),qt.basis(2,m1),qt.basis(2,m2))
s=1/np.sqrt(2)
LOGICAL=[ket(1,0),ket(0,1),s*(ket(1,0)+ket(0,1)),s*(ket(1,0)-ket(0,1)),s*(ket(1,0)+1j*ket(0,1)),s*(ket(1,0)-1j*ket(0,1))]
def phonon_c_ops(T1,Tphi,rho,kp=0.0):
    k=1/T1+kp; c=[np.sqrt(k)*a1,np.sqrt(k)*a2]
    if Tphi is not None and np.isfinite(Tphi):
        gam=2/Tphi
        c+= [np.sqrt(gam*(1+rho)/2)*(n1+n2)]
        if rho<1: c+=[np.sqrt(gam*(1-rho)/2)*(n1-n2)]
    return c
def storage(psi,T,T1,Tphi,rho,kp):
    if T<=0: return qt.ket2dm(psi)
    return qt.mesolve(0*n1,psi,[0,T],phonon_c_ops(T1,Tphi,rho,kp)).states[-1]
def check(rho0,chi,dchi,sigma,T1,Tphi,rho,kp,T1q,T2q,eps,chis=None,wd_shift=0.0):
    """pulso pi gaussiano selectivo en n_tot=1; devuelve (rho_pass_fonones, p_pass, rho_fail)"""
    chi1,chi2=(chi*(1+dchi/2),chi*(1-dchi/2)) if chis is None else chis; cbar=(chi1+chi2)/2+wd_shift
    H0=(cbar-chi1*n1-chi2*n2)*Pe
    Hd=0.5*(q+q.dag()); tp=4*sigma; t0=tp/2
    Om0=np.pi/(sigma*np.sqrt(2*np.pi))
    f=lambda t,args: Om0*np.exp(-(t-t0)**2/(2*sigma**2))
    c=phonon_c_ops(T1,Tphi,rho,kp)
    gq=max(1/T2q-1/(2*T1q),0)
    c+=[np.sqrt(1/T1q)*q, np.sqrt(gq/2)*(2*Pe-1)]
    tl=np.linspace(0,tp,int(tp*40)+2)
    r=qt.mesolve([H0,[Hd,f]],rho0,tl,c,options={"atol":1e-9,"rtol":1e-7,"max_step":0.05}).states[-1]
    # lectura proyectiva del transmon
    Pg=qt.tensor(qt.basis(2,0)*qt.basis(2,0).dag(),I2,I2); Pee=qt.tensor(qt.basis(2,1)*qt.basis(2,1).dag(),I2,I2)
    re=(Pee*r*Pee).ptrace([1,2]); rg=(Pg*r*Pg).ptrace([1,2])
    rp=(1-eps)*re+eps*rg; pp=float(np.real(rp.tr())); rf=(1-eps)*rg+eps*re
    return rp/pp if pp>1e-12 else rp, pp, rf
PH=lambda m1,m2: qt.tensor(qt.basis(2,m1),qt.basis(2,m2))
LOGPH=[PH(1,0),PH(0,1),s*(PH(1,0)+PH(0,1)),s*(PH(1,0)-PH(0,1)),s*(PH(1,0)+1j*PH(0,1)),s*(PH(1,0)-1j*PH(0,1))]
def fid(rp,psi): return float(np.real((psi.dag()*rp*psi).tr())) if rp.type=='oper' and rp.isherm else float(np.real(qt.expect(rp,psi)))
def zfid(rho,psi_ph,phi):
    U=qt.tensor(qt.qeye(2),qt.qeye(2)); Rz=(1j*phi*n2).expm() if False else None
    return None
def run(T,rho,plat,sigma=4.0,chi=2*np.pi*0.128,dchi=0.0,eps=0.01,use_check=True,chis=None,zcomp=True,wd_shift=0.0):
    T1,Tphi,kp,T1q,T2q=plat["T1"],plat["Tphi"],plat["kp"],plat["T1q"],plat["T2q"]
    F_unc=[];F_h=[];F_c=[];pp=[];RC=[]
    for psi,pph in zip(LOGICAL,LOGPH):
        r=storage(psi,T,T1,Tphi,rho,kp)                  # 8-dim, transmon en g
        rph=r.ptrace([1,2])
        F_unc.append(float(np.real((pph.dag()*rph*pph))))
        # heraldado por conteo al final (proyeccion a n=1)
        P1=PH(1,0)*PH(1,0).dag()+PH(0,1)*PH(0,1).dag()
        w=(P1*rph).tr(); rh=(P1*rph*P1)/w if w>1e-12 else rph
        F_h.append(float(np.real((pph.dag()*rh*pph))))
        if use_check:
            rc,p,_=check(r,chi,dchi,sigma,T1,Tphi,rho,kp,T1q,T2q,eps,chis=chis,wd_shift=wd_shift)
            RC.append(rc); pp.append(p)
    out=dict(F_unc=np.mean(F_unc),F_h=np.mean(F_h))
    if use_check:
        phis=np.linspace(-np.pi,np.pi,129) if zcomp else [0.0]
        best=-1
        for ph in phis:
            Rz=qt.tensor(qt.qeye(2),qt.qeye(2)); D=qt.qdiags([1,1,1,1],0)
            U=qt.tensor(qt.qeye(2),qt.qeye(2))
            # Rz logico: |01> -> e^{i ph}|01>; fases de |00>,|10>,|11> irrelevantes en el sector
            U=qt.tensor(qt.qeye(2),qt.qeye(2))*0+qt.Qobj(np.diag([1,np.exp(1j*ph),1,1]),dims=U.dims)
            Fm=np.mean([float(np.real((pph.dag()*(U*rc*U.dag())*pph))) for rc,pph in zip(RC,LOGPH)])
            best=max(best,Fm)
        out.update(F_c=best,p_pass=np.mean(pp),F_c_nocomp=np.mean([float(np.real((pph.dag()*rc*pph))) for rc,pph in zip(RC,LOGPH)]))
    return out
SI =dict(T1=534.,Tphi=107.,T1q=28.,T2q=15.,kp=(1.1/10)**2/28.)   # g=1.1 MHz, Delta=10 MHz
if __name__=="__main__":
    import time; t=time.time()
    print(run(50.,0.65,SI,sigma=4.0), time.time()-t)
