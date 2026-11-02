import sys, json, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/user/nphonons-bundles/doble_carril_fonon/estudio/src')
from multiprocessing import Pool
import erasure_sim as E
from mydisp import plat_si, chis
PTS = [(6,3.3,100,.65),(20,3.3,100,.65),(20,1.0,100,.65),(6,3.3,50,.95)]
SIG = [1,1.5,2,3,4,6,8]
def job(a):
    D,F,T,rho,s = a
    r = E.run(T,rho,plat_si(Delta=D),sigma=s,chis=chis(D,F),eps=0.01)
    return (D,F,T,rho,s,r['F_c'],r['F_c_nocomp'],r['p_pass'])
if __name__=='__main__':
    jobs=[(D,F,T,r,s) for D,F,T,r in PTS for s in SIG]
    with Pool(4) as p: res=p.map(job,jobs)
    json.dump(res,open('au_scan.json','w'))
    my=json.load(open('my_scan.json'))
    print('D F T rho s | Fc_my Fc_au dF | pp_my pp_au dpp | Fnoz_my Fnoz_au')
    for m,a in zip(my,res):
        print(m[:5],'%.4f %.4f %+.4f | %.4f %.4f %+.4f | %.4f %.4f'%(m[5],a[5],a[5]-m[5],m[7],a[7],a[7]-m[7],m[6],a[6]))
