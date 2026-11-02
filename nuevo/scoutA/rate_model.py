"""Modelo analítico de escalera oscura (CPT) + trampa T=|-,0>: bloques {|0,n>,|+,n>,|-,n+1>}, Om_e=sqrt2*Om, g=dl/2 (eps=0,S=1)."""
import numpy as np
def rate_model(Om,gam,kap,g=0.07,N=40):
    Oe=np.sqrt(2)*Om; w=lambda n:Oe**2/(Oe**2+g**2*(n+1))
    M=np.zeros((N+2,N+2)) # índice 0=T, 1+n = D_n ; M[i,j]= tasa j->i
    M[1,0]+=gam                       # T -> D_0
    for n in range(N):
        M[2+n,1+n]+=gam*w(n)          # D_n -> D_{n+1}
        if n>=1: M[n,1+n]+=kap*(n+w(n))   # D_n -> D_{n-1}
    M[N,1+N]+=kap*(N+w(N))
    M[0,1]+=kap*w(0)                  # D_0 -> T
    A=M-np.diag(M.sum(0)); A[-1,:]=1; b=np.zeros(N+2); b[-1]=1
    p=np.linalg.solve(A,b)
    m1=p[0]*0+sum(p[1+n]*(n+w(n)) for n in range(N))
    m2=sum(p[1+n]*((1-w(n))*n**2+w(n)*(n+1)**2) for n in range(N))
    fact=sum(p[1+n]*((1-w(n))*n*(n-1)+w(n)*(n+1)*n) for n in range(N))
    return m1,fact/m1**2,(fact-m1**2)/m1,p[0]
if __name__=='__main__':
    for (Om,gam,kap) in ((0.07,0.01,0.003),(0.05,0.01,0.005),(0.05,0.002,0.003),(0.1,0.002,0.005),(0.1,2e-4,0.003),(0.03,0.03,0.003)):
        print('Om=%g gam=%g kap=%g -> n=%.3f g2=%.3f Q=%.3f pT=%.2f'%((Om,gam,kap)+rate_model(Om,gam,kap)))
