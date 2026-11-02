import numpy as np, qutip as qt
from math import pi
def ops(nq=3,nm=4):
    q=qt.tensor(qt.destroy(nq),qt.qeye(nm),qt.qeye(nm)); a1=qt.tensor(qt.qeye(nq),qt.destroy(nm),qt.qeye(nm)); a2=qt.tensor(qt.qeye(nq),qt.qeye(nm),qt.destroy(nm))
    return q,a1,a2
def H_jc(D,F,g=1.1,alpha=185.,DL=0.,nq=3,nm=4):
    q,a1,a2=ops(nq,nm)
    return (DL*q.dag()*q + pi*alpha*(-1)*q.dag()*q.dag()*q*q   # (alpha/2)*2pi, alpha<0
            + (DL-2*pi*D)*a1.dag()*a1 + (DL-2*pi*(D+F))*a2.dag()*a2
            + 2*pi*g*(q.dag()*a1+a1.dag()*q+q.dag()*a2+a2.dag()*q))
def spec(D,F,nq=3,nm=4):
    # H en rad/us con frecuencias angulares: D,F en MHz
    q,a1,a2=ops(nq,nm)
    q_=q
    H=(-pi*185*q.dag()*q.dag()*q*q + (-2*pi*D)*a1.dag()*a1 + (-2*pi*(D+F))*a2.dag()*a2
       + 2*pi*1.1*(q.dag()*a1+a1.dag()*q+q.dag()*a2+a2.dag()*q))
    E,V=H.eigenstates()
    def ket(e,m1,m2): return qt.tensor(qt.basis(nq,e),qt.basis(nm,m1),qt.basis(nm,m2))
    def en(e,m1,m2):
        k=ket(e,m1,m2); ov=[abs(k.overlap(v))**2 for v in V]; return E[int(np.argmax(ov))]
    out={}
    for s in [(0,0),(1,0),(0,1)]:
        out[s]=(en(1,*s)-en(0,*s))/(2*pi)   # MHz
    return out
if __name__=="__main__":
  for D,F in [(6,3.3),(20,3.3),(20,1.0)]:
      o=spec(D,F); c=lambda d:(1.1**2/d)*185/(d+185)
      c1,c2=c(D),c(D+F)
      print(D,F,{k:round(v,4) for k,v in o.items()},
            'JC split(10)-(00)=%.4f split(01)-(00)=%.4f | 2chi1=%.4f 2chi2=%.4f chi1=%.4f chi2=%.4f'%(o[(1,0)]-o[(0,0)],o[(0,1)]-o[(0,0)],2*c1,2*c2,c1,c2))
