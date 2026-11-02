import sys; sys.path.insert(0,'/home/user/nphonons-bundles/nuevo/bundles/src')
import numpy as np, bundles_core as bc, itertools
from multiprocessing import Pool
def f(a):
    n,l,O,k,g,gp=a
    Pe=bc.exact_point(n,l,O,k,g,gp,N=n+6)['P']; Pc=bc.P_closed(l,O,k,g,gp,n)
    xi=bc.omega_eff(n,l,O)/bc.gamma_c(n,0,k,g,gp)
    return a+(Pe,Pc,xi)
if __name__=='__main__':
    rng=np.random.default_rng(1)
    A=[]
    for _ in range(64):
        n=int(rng.choice([2,3]));l=float(rng.choice([.12,.16,.18]));O=float(rng.choice([.04,.07,.13,.17]));k=float(rng.choice([.0012,.0022,.0035,.0075]));g=float(rng.choice([2e-4,5e-4]))
        A.append((n,l,O,k,g,4e-4))
    A=list(dict.fromkeys(A))
    with Pool(2) as p: R=p.map(f,A)
    np.save('R.npy',np.array(R))
    print(len(R))
    print(' n lam Om kap gam | Pex Pcl xi | errP% errImp% valid')
    inr=[];out=[]
    for r in R:
        n,l,O,k,g,gp,Pe,Pc,xi=r
        eP=100*(Pc-Pe)/Pe; eI=100*((1-Pc)-(1-Pe))/(1-Pe); v=(xi<.5 and Pe>=.85)
        print(f'{int(n)} {l} {O} {k} {g} | {Pe:.4f} {Pc:.4f} {xi:.2f} | {eP:+.2f} {eI:+.1f} {v}')
        (inr if v else out).append((abs(eP),abs(eI),Pe))
    inr=np.array(inr);out=np.array(out)
    print('valid',len(inr),'of',len(R),'maxP%',inr[:,0].max(),'medP',np.median(inr[:,0]),'maxImp',inr[:,1].max(),'medImp',np.median(inr[:,1]))
    if len(out): print('out',len(out),'maxP%',out[:,0].max(),'med',np.median(out[:,0]),'maxImp',out[:,1].max(),'medImp',np.median(out[:,1]))
