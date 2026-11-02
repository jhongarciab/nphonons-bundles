"""Modelo mínimo: molécula de 2 QDs (0,X1,X2; sin biexcitón) + modo de fonón. ω_b=1."""
import numpy as np, qutip as qt
def ops(Nb):
    k=qt.basis(3,0); x1=qt.basis(3,1); x2=qt.basis(3,2)
    s1=k*x1.dag(); s2=k*x2.dag(); n1=x1*x1.dag(); n2=x2*x2.dag()
    I3=qt.qeye(3); Ib=qt.qeye(Nb); b=qt.tensor(I3,qt.destroy(Nb))
    E=lambda o:qt.tensor(o,Ib)
    return dict(s1=E(s1),s2=E(s2),n1=E(n1),n2=E(n2),b=b,Ib=Ib,I3=I3,
                s12=E(x1*x2.dag()))
def model(Nb=20,eps=0.0,J=0.5,wL=0.0,l1=0.14,l2=0.0,Om=0.1,gam=2e-4,gph=4e-4,kap=0.005,
          Om2=None,nth=0.0,U=None,pump_on_site=True):
    """eps = ω1-ω2 (detuning entre QDs), J Förster, wL = frecuencia del láser medida desde el punto medio
    (ω1+ω2)/2 en el marco rotante; Om, Om2 bombeo (Ω1=Ω, Ω2=Om2 o Ω)."""
    o=ops(Nb); b=o['b']; bd=b.dag(); Om2=Om if Om2 is None else Om2
    H=(eps/2-wL)*o['n1']+(-eps/2-wL)*o['n2']+J*(o['s12']+o['s12'].dag())+bd*b
    H+=(l1*o['n1']+l2*o['n2'])*(b+bd)
    H+=Om*(o['s1']+o['s1'].dag())+Om2*(o['s2']+o['s2'].dag())
    c=[np.sqrt(gam)*o['s1'],np.sqrt(gam)*o['s2'],np.sqrt(gph)*o['n1'],np.sqrt(gph)*o['n2'],np.sqrt(kap*(1+nth))*b]
    if nth>0: c.append(np.sqrt(kap*nth)*bd)
    return H,c,o
def stats(rho,o,Nb):
    b=o['b']; n=b.dag()*b
    m=qt.expect(n,rho); m2=qt.expect(b.dag()*b.dag()*b*b,rho)
    g2=m2/m**2 if m>1e-12 else np.nan
    Q=(m2-m**2)/m if m>1e-12 else np.nan   # Mandel Q = (<n^2>-<n>^2)/<n> -1
    rb=rho.ptrace(1); P=np.real(np.diag(rb.full()))
    pop=[qt.expect(o[k],rho) for k in ('n1','n2')]
    return dict(n=m,g2=g2,Q=Q,P=P,pop=pop,edge=P[-3:].sum())
def ss(**kw):
    Nb=kw.get('Nb',20); H,c,o=model(**kw)
    rho=qt.steadystate(H,c,method='direct',solver='spsolve')
    return stats(rho,o,Nb),rho,o
if __name__=='__main__':
    import time; t=time.time(); r,_,_=ss(Nb=20,eps=0,J=0.5,wL=0.5)
    print(time.time()-t,{k:(np.round(v,4) if k!='P' else np.round(v[:6],3)) for k,v in r.items()})
