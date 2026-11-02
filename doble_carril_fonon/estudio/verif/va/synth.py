import numpy as np, json
from mc2 import mc
import mc as M
# (A1,A2,gam) rad/us, us^-1 ; tipos: S=compartido mismo signo, P=privado, O=opuesto
dev={
 'D1_compartidos(N=4)':([1.0,0.8,0.5,0.3],[1.0,0.8,0.5,0.3],[0.3,1.0,3.0,0.1]),
 'D2_privados(N=6)':([1.0,0.7,0.5,0,0,0],[0,0,0,1.0,0.7,0.5],[0.2,1.,2.,0.3,1.,.1]),
 'D3_opuestos(N=5)':([1.0,0.8,0.6,0.4,0.9],[-1.0,-0.8,-0.6,-0.4,-0.9],[0.3,1.,0.5,2.,0.1]),
 'D4_mixto(N=8)':([1.2,0.9,0.5,0.7,0.0,0.4,0.3,0.6],[1.1,0.8,-0.5,0.0,0.6,0.4,-0.2,0.3],[0.2,0.5,1.,3.,0.4,0.1,2.,0.8]),
 'D5_cuasiestatico(N=3)':([1.0,0.6,0.5],[0.9,0.6,-0.4],[0.02,0.05,0.03]),
}
tg=np.geomspace(0.05,12,70);out={}
for k,(a1,a2,g) in dev.items():
    a1,a2,g=map(np.array,(a1,a2,g))
    m1,m2,mx=mc(a1,a2,g,tg,40000,5)
    c1,c2,cx=M.closed(a1,a2,g,tg)
    X=lambda f:M.cross(tg,f)
    fine=np.geomspace(0.01,200,200000);F=M.closed(a1,a2,g,fine)
    Xf=lambda f:M.cross(fine,f)
    # primer cruce
    r=dict(maxdiff_f=float(max(np.abs(m-c)[(c<1.5)].max() for m,c in ((m1,c1),(m2,c2),(mx,cx)))),
      t1=(X(m1),X(c1),Xf(F[0])),t2=(X(m2),X(c2),Xf(F[1])),tI=(X(m1+m2),X(c1+c2),Xf(F[0]+F[1])),tx=(X(mx),X(cx),Xf(F[2])))
    r['Phi']=(1-r['tI'][0]/r['tx'][0],1-r['tI'][1]/r['tx'][1],1-r['tI'][2]/r['tx'][2])
    # correlacion de fases y cota A4
    out[k]=r;print(k,json.dumps(r,default=float))
