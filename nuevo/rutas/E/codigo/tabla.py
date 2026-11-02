import numpy as np, qutip as qt
from modelo import Full; from purity import *
TP=2*np.pi
def best_Omega(gx,gz,km,kq,x=5.0,P2min=0.95,full=None):
    G=2*gx*gz; best=None
    for x in (5.0,8.0,12.0):
      for om in np.geomspace(0.005,3,40)*G:
        H,c=eff_model(gx,gz,N=6,Om=om,kap_m=km,kap_q=kq); r=cluster_stats(H,c,0,x/km,mmax=5)
        if r['P2']>=P2min and (best is None or r['rate']>best[1]): best=(om,r['rate'],r['P2'],r['meank'],x)
    return best
casos=[("Naseem (g_x=0.6, g_z=6, w=100 MHz)",100,0.006,0.06),("g_x,g_z x3.3 (hipot.: g_x=2,g_z=10 MHz)",100,0.02,0.10)]
out=[]
for nom,wMHz,ex,lz in casos:
    G=2*ex*lz; Gm=G*wMHz*0.993   # G/2pi en MHz (factor numerico 0.993 / 0.98)
    for kqk in [100,30,10,3,1]:   # kappa_q/2pi en kHz
        kq=kqk*1e-3/wMHz; km=np.sqrt(2)*G
        eta=1/(1+kq/G/4*(km/G+kq/G/2+2/(km/G)))
        b=best_Omega(ex,lz,km,kq)
        Ge=km-np.sqrt(km**2-8*G**2) if km>2*np.sqrt(2)*G else km      # tasa lenta de |e0> (unidades de w)
        Temit=1/Ge+5/km; pulsed=2*eta/Temit*TP*wMHz*1e6                  # clics/s con bombeo pulsado (pi-pulso rapido, sin fusion), T_ciclo ~ 1/Ge+5/km
        if b: out.append((nom,kqk,Gm*1e3,km*wMHz*1e3,kq/G,eta,b[0]/G,b[1]*TP*wMHz*1e6,b[2],b[3],b[4],pulsed))
        else: out.append((nom,kqk,Gm*1e3,km*wMHz*1e3,kq/G,eta,None,None,None,None,None,pulsed))
print("| caso | kq/2pi [kHz] | G/2pi [kHz] | km/2pi [kHz] | kq/G | eta_par | Omega/G | clics/s (CW) | P2 | <k> | x=km*tau_c | pulsado clics/s (est.) |")
for r in out:
    if r[6] is None: print(f"| {r[0]} | {r[1]} | {r[2]:.1f} | {r[3]:.1f} | {r[4]:.3f} | {r[5]:.3f} | no hay Omega con P2>=0.95 | - | - | - | - | - |")
    else:
        # r[7]: tasa de clics en unidades de G*(2pi*MHz) -> clics/s = rate * G_rad/s ; G_rad = 2pi*Gm MHz
        print(f"| {r[0]} | {r[1]} | {r[2]:.1f} | {r[3]:.1f} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.3f} | {r[7]:.3g} | {r[8]:.3f} | {r[9]:.3f} | {r[10]:.0f} | {r[11]:.3g} |")
# DC / disipador consistente (modelo completo) en el punto Naseem-like kq/G=0.05
gx,gz=0.006,0.06; G=2*gx*gz; m=Full(gx,gz,N=10); E0=m.E[m.ig0]; dl=m.E[m.hyb].mean()-E0-2
print("\nDC / disipador (completo, N=10; km=sqrt2 G, Om=0.1G, kq=0.05G):")
for lab,kw in [("vestido fisico (sin DC, J2=1)",dict(Jdc=0)),("DC plano (como 'b' desnudo, Jdc=1)",dict(Jdc=1.0)),("DC plano + J(2w)=0 (guia pasabanda)",dict(Jdc=1.0,J2=0.0))]:
    H,c=m.build(0.1*G,dl,np.sqrt(2)*G,0.05*G,**kw); r=cluster_stats(H,c,m.lab.index('b1'),5/(np.sqrt(2)*G))
    rho=qt.steadystate(H,c); s=m.stats(rho,c)
    print(f"  {lab}: flujo b1/G={r['rate']/G:.5f} P2={r['P2']:.4f} g2(0)={s['g2']:.2f} <n>={s['n']:.4f} kappa*<n>/G={np.sqrt(2)*s['n']:.4f}  tasas/G={ {k:round(v/G,5) for k,v in s['rate'].items()} }")
