import numpy as np, json
from transmon_f import *
res={}
cases={'PnC-A (g=15.2, a=-138, T1=1.14us)':(15.2,-138,1.14),'PnC g,a como A, T1=10us':(15.2,-138,10),'PnC g=15.2, a=-300, T1=10us':(15.2,-300,10),'PnC g=15.2, a=-300, T1=50us':(15.2,-300,50),'g=5, a=-250, T1=50us':(5,-250,50),'HBAR g=0.35, a=-250(sup), T1=50us':(0.35,-250,50)}
kmgrid=np.geomspace(0.02,40,30)   # kappa_m/2pi en MHz
print(f"{'caso':38s} V/2pi[MHz]  km_opt/2pi  P2max  P1  | P2 en km=V   | km^(P2>=0.9)")
for name,(g,al,T1) in cases.items():
    g_,al_=g*TP,al*TP; Dq,gap=resonance(g_,al_); V=gap/2/TP
    rows=[]
    for k in kmgrid:
        r=pares(g_,al_,k*TP,1.0/T1,Dq=Dq); rows.append((k,r['P2'],r['P1']))
    rows=np.array(rows); i=np.argmax(rows[:,1])
    r2=pares(g_,al_,V*TP,1.0/T1,Dq=Dq)
    print(f"{name:38s} {V:9.4f}   {rows[i,0]:8.3f}   {rows[i,1]:.3f}  {rows[i,2]:.3f} | {r2['P2']:.3f}")
    res[name]=dict(V=V,rows=rows.tolist())
json.dump(res,open('transmon_res.json','w'))
