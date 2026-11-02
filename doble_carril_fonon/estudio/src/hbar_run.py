import sys, json, numpy as np; sys.path.insert(0,'.')
import erasure_sim as E
from design_run import chik
g,fsr,T1m=0.273,12.7,82.
out=[]
for T1q in (28.,300.,1000.):
    for T in (20.,40.):
        best=None
        for D in (1.5,3,6,10,20,40):
            if g/D>0.2: continue
            for sg in (0.5,1,2,4,8,16,30):
                P=dict(T1=T1m,Tphi=None,T1q=T1q,T2q=1.5*T1q if T1q>30 else 15.,kp=(g/D)**2/T1q)
                r=E.run(T,0.0,P,sigma=sg,chis=(chik(g,D),chik(g,D+fsr)),zcomp=True)
                Q=r['p_pass']*(2*r['F_c']-1)
                if best is None or Q>best[0]: best=(Q,D,sg,r)
        Q,D,sg,r=best
        b=E.run(T,0.0,dict(T1=T1m,Tphi=None,T1q=100.,T2q=150.,kp=0.0),use_check=False)
        out.append(dict(T1q=T1q,T=T,D=D,sg=sg,F_c=r['F_c'],p_pass=r['p_pass'],B_Func=b['F_unc'],B_Fh=b['F_h'],p_ideal=float(np.exp(-T/T1m))))
        print(f"T1q={T1q:5.0f} T={T:3.0f}: D={D} sg={sg}: F_c={r['F_c']:.3f} p={r['p_pass']:.3f} | base F_unc={b['F_unc']:.3f} F_h={b['F_h']:.3f} p_ideal={np.exp(-T/T1m):.3f}")
json.dump(out,open('design_eval_hbar.json','w'))
