from erasure_sim import *
def chi_of(g,D,alpha=185.): return 2*np.pi*(g**2/D)*alpha/(D+alpha)
print("\n== HBAR (ETH): T1=75us, sin desfase puro, g=0.273 MHz; contraste")
for T1q in (28.,300.):
    for T in (20.,40.):
        bestv=None
        for D in (1.0,1.5,2.0,3.0):
            for sg in (1.5,2.5,4.0,6.0):
                P=dict(T1=75.,Tphi=None,T1q=T1q,T2q=1.5*T1q if T1q>30 else 15.,kp=(0.273/D)**2/T1q)
                r=run(T,0.65,P,sigma=sg,chi=chi_of(0.273,D))
                if bestv is None or r["F_c"]>bestv[0]["F_c"]: bestv=(r,D,sg)
        r,D,sg=bestv
        base=run(T,0.0,dict(T1=75.,Tphi=None,T1q=T1q,T2q=15.,kp=0.0),use_check=False)
        print(f" T1q={T1q:4.0f} T={T:4.0f}us D={D} sg={sg} | F_c={r['F_c']:.3f} p_pass={r['p_pass']:.3f} | base: F_unc={base['F_unc']:.3f} F_h={base['F_h']:.3f} | p_ideal(no loss)={np.exp(-T/75):.3f}")
