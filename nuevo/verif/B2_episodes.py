import os; os.environ['OMP_NUM_THREADS']='1'
from B_purity import *
lam=0.14; n=2
for (kap,Om,gam) in [(0.01,0.03,2e-5),(0.003,0.05,2e-5),(0.01,0.03,2e-6)]:
    m=Model(lam,Om,kap,gam,4e-4,12,'pol_dressed'); dL=lam**2-n+eps_n(n,lam,Om)[0]
    print(f"kap={kap} Om={Om} gam={gam}  e^-l2={np.exp(-lam**2):.4f}  1/(2-e^-l2)={1/(2-np.exp(-lam**2)):.4f} 2/(2+l2)={2/(2+lam**2):.4f}")
    for tf in (3,5,8,12,20):
        P=episodes(m,dL,tf/kap); k=np.arange(len(P))
        print(f"  tau={tf}/kap: P1={P[1]:.4f} P2={P[2]:.4f} P3={P[3]:.4f} P4={P[4]:.4f}  fr_fonones(n=2)={2*P[2]/(k*P).sum():.4f} mean={(k*P).sum():.4f}",flush=True)
