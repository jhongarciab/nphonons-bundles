from erasure_sim import *
def chi_of(g,D,alpha=185.): return 2*np.pi*(g**2/D)*alpha/(D+alpha)
def standalone(g,D,sigma,eps=0.01,T1q=28.,T2q=15.):
    chi=chi_of(g,D); kp=(g/D)**2/T1q
    # |10> deberia pasar, |00> deberia fallar. Sin ruido de fonones.
    r10=ket(1,0); r00=ket(0,0)
    _,p1,_=check(qt.ket2dm(r10),chi,0.0,sigma,1e9,None,0.0,0.0,T1q,T2q,eps)
    _,p0,_=check(qt.ket2dm(r00),chi,0.0,sigma,1e9,None,0.0,0.0,T1q,T2q,eps)
    # ideal sin decoherencia del transmon
    _,p1i,_=check(qt.ket2dm(r10),chi,0.0,sigma,1e9,None,0.0,0.0,1e9,1e9,0.0)
    _,p0i,_=check(qt.ket2dm(r00),chi,0.0,sigma,1e9,None,0.0,0.0,1e9,1e9,0.0)
    return chi/(2*np.pi)*1e3,kp,p1,p0,p1i,p0i
if __name__=="__main__":
    print("g[MHz] D[MHz] chi[kHz] sigma[us] tp[us] | P(pass|n=1) P(pass|n=0) | ideal: P(pass|1) P(pass|0) | Purcell T1p[ms]")
    for g in (1.1,):
        for D in (4,6,10,20):
            for sg in (1.5,3,5,8):
                chik,kp,p1,p0,p1i,p0i=standalone(g,D,sg)
                print(f"{g:5} {D:6} {chik:7.1f} {sg:8} {4*sg:6} | {p1:.3f} {p0:.3f} | {p1i:.3f} {p0i:.3f} | {1/kp/1000:.1f}")
