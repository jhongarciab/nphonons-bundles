import numpy as np, json, warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool
from mydisp import *
PTS = [(6,3.3,100,.65),(20,3.3,100,.65),(20,1.0,100,.65),(6,3.3,50,.95)]
SIG = [1,1.5,2,3,4,6,8]
def job(a):
    D,F,T,rho,s = a
    c1,c2 = chis(D,F); r = run_my(T,rho,plat_si(Delta=D),s,c1,c2,0.01)
    return (D,F,T,rho,s,r['Fc'],r['F_noZ'],r['pp'])
if __name__=='__main__':
    jobs=[(D,F,T,r,s) for D,F,T,r in PTS for s in SIG]
    jobs += [(20,3.3,100,.65,s) for s in [3.0] if False]
    with Pool(4) as p: res=p.map(job,jobs)
    json.dump(res,open('my_scan.json','w'))
    for r in res: print(r[:5], 'Fc=%.4f Fnoz=%.4f pp=%.4f'%r[5:])
