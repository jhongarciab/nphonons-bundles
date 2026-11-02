import json,sys,numpy as np,warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool
from myjc import *
PTS=[(6,3.3,100,.65),(20,3.3,100,.65),(20,1.0,100,.65),(6,3.3,50,.95)]
SIG=[1.5,2,3,4,6]
def job(a):
  try: return job_(a)
  except Exception as e: return dict(D=a[0],F=a[1],T=a[2],rho=a[3],s=a[4],side=a[5],nm=a[6],err=str(e)[:80])
def job_(a):
    D,F,T,rho,s,side,nm=a
    jc=JC(D,F,nm=nm,side=side); DL,pmin,guess=best_DL_sym(jc,s,D,F,side)
    m=jc.metric(T,rho,plat_si(Delta=D),DL,s,0.01)
    return dict(D=D,F=F,T=T,rho=rho,s=s,side=side,nm=nm,DL=DL,off=DL-guess,pmin=pmin,Fc=m['Fc'],pp=m['pp'])
if __name__=='__main__':
    jobs=[(D,F,T,r,s,side,3) for side in ['below','above'] for D,F,T,r in PTS for s in SIG]
    jobs.append((6,3.3,100,.65,3.0,'below',4))   # chequeo de truncacion nm=4
    res=[]
    with Pool(4) as p:
        for r in p.imap_unordered(job,jobs,chunksize=1):
            res.append(r); open('jc_scan.jsonl','a').write(json.dumps(r)+'\n')
    json.dump(res,open('jc_scan.json','w'))
    for r in res: print({k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()})
