import numpy as np, qutip as qt
def card(d):
    k=[qt.basis(d,i) for i in range(d)]
    return k
def run(T1,Tphi,rho,T):
    g=1/Tphi
    # single rail
    s0,s1=qt.basis(2,0),qt.basis(2,1)
    c=[np.sqrt(1/T1)*s0*s1.dag(), np.sqrt(g)*s1*s1.dag()]
    st=[s0,s1,(s0+s1).unit(),(s0-s1).unit(),(s0+1j*s1).unit(),(s0-1j*s1).unit()]
    Fs=np.mean([qt.expect(qt.ket2dm(s),qt.mesolve(0*qt.qeye(2),qt.ket2dm(s),[0,T],c).states[-1]) for s in st])
    # dual rail: basis |10>,|01>,|00>
    b=[qt.basis(3,i) for i in range(3)]
    P=lambda i,j:b[i]*b[j].dag()
    n1,n2=P(0,0),P(1,1)
    cd=[np.sqrt(1/T1)*P(2,0),np.sqrt(1/T1)*P(2,1),
        np.sqrt(g*(1+rho)/2)*(n1+n2),np.sqrt(g*(1-rho)/2)*(n1-n2)]
    ds=[b[0],b[1],(b[0]+b[1]).unit(),(b[0]-b[1]).unit(),(b[0]+1j*b[1]).unit(),(b[0]-1j*b[1]).unit()]
    Fh=[];ps=[]
    for s in ds:
        r=qt.mesolve(0*qt.qeye(3),qt.ket2dm(s),[0,T],cd).states[-1]
        pp=r[0,0].real+r[1,1].real
        Fh.append((s.dag()*r*s).real/pp); ps.append(pp)
    Fh=np.mean(Fh);p=np.mean(ps)
    return Fs,Fh,p
rows=[]
print("T1 Tphi T rho rho* | Fs_num Fs_an | Fd_num Fd_an | dnum danl | side ok")
for Tphi in (107,10):
  T1=534; rs=0.5*(1+Tphi/T1)
  for T in (25,100,200):
    for rho in (rs-0.1+0.0,rs+0.1) if rs<0.9 else (rs-0.1,0.98):
      rho=min(rho,1.0)
      Fs,Fh,p=run(T1,Tphi,rho,T)
      f1=T/(2*Tphi);pa=np.exp(-T/T1);C=np.exp(-(1-rho)*2*f1)
      Fsa=(3+pa+2*np.sqrt(pa)*np.exp(-f1))/6; Fha=(2+C)/3
      Fd=p*Fh+(1-p)/2; Fda=pa*Fha+(1-pa)/2
      ok=(Fd>Fs)==(rho>rs)
      print(f"{T1} {Tphi} {T:4d} {rho:.3f} {rs:.3f} | {Fs:.5f} {Fsa:.5f} | {Fh:.5f} {Fha:.5f} p={p:.4f}/{pa:.4f} | Fd={Fd:.5f}/{Fda:.5f} dF={Fd-Fs:+.5f} {ok}")
