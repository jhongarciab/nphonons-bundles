import json
from multiprocessing import Pool
from scan_jc import job, PTS
if __name__=='__main__':
    jobs=[(D,F,T,r,1.0,side,3) for side in ['below','above'] for D,F,T,r in PTS]
    with Pool(4) as p: res=p.map(job,jobs)
    json.dump(res,open('jc_scan_s1.json','w'))
    for x in res: print(x['side'],x['D'],x['F'],x['T'],'s=1 Fc=%.4f pp=%.4f off=%.3f'%(x['Fc'],x['pp'],x['off']))
