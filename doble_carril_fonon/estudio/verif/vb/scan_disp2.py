import json,numpy as np,warnings; warnings.filterwarnings('ignore')
from mydisp import *
from myjc import chi_eff
PTS=[(6,3.3,100,.65),(20,3.3,100,.65),(20,1.0,100,.65),(6,3.3,50,.95)]
SIG=[1,1.5,2,3,4,6,8]
res=[]
for D,F,T,rho in PTS:
    pl=plat_si(Delta=D)
    base=run_my(T,rho,pl,3.0,0,0,0.01,use_check=False)
    for lab,cc in [('spec',chis(D,F))]+[('2chi_'+sd,tuple(2*x for x in chi_eff(D,F,sd))) for sd in ['below','above']]:
        for s in SIG:
            r=run_my(T,rho,pl,s,cc[0],cc[1],0.01); res.append(dict(D=D,F=F,T=T,rho=rho,lab=lab,s=s,Fc=r['Fc'],pp=r['pp'],base=base['Fc'],c1=cc[0],c2=cc[1]))
json.dump(res,open('disp2_scan.json','w'))
for D,F,T,rho in PTS:
    for lab in ['spec','2chi_below','2chi_above']:
        rr=[r for r in res if (r['D'],r['F'],r['T'],r['rho'],r['lab'])==(D,F,T,rho,lab)]
        b=max(rr,key=lambda r:r['Fc']); print(D,F,T,rho,lab,'chi/2pi=%.4f,%.4f'%(b['c1']/2/np.pi,b['c2']/2/np.pi),'sig_opt',b['s'],'Fc=%.4f pp=%.4f base(no check)=%.4f'%(b['Fc'],b['pp'],b['base']))
