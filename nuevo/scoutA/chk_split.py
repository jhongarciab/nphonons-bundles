import numpy as np, qutip as qt
from qdm import *
print('## Desdoblamiento de vacio: H sin bombeo en la variedad 1-exciton (+0 y 1 fonon), resonancia S=1; formula g=dl*J/S')
for eps in (0.0,0.5,0.8):
    S=1.0; J=np.sqrt(S**2-eps**2)/2
    for l1,l2 in ((0.14,0.0),(0.14,0.07)):
        H,c,o=model(Nb=14,eps=eps,J=J,wL=0.0,l1=l1,l2=l2,Om=0.0)
        # proyecta en excitón (n1+n2=1), niveles cerca de energía de |+,0>~S/2 :
        P=o['n1']+o['n2']; H1=(H.full()); Pm=np.real(np.diag(P.full())); idx=np.where(Pm>0.5)[0]
        ev=np.linalg.eigvalsh(H1[np.ix_(idx,idx)])
        # niveles polaron: refiere a energías sin desplazar; busca el doblete más cercano a S/2 - lam_bar^2 shift
        lb=(l1+l2)/2; sh=-lb**2  # desplazamiento polaron común (n_X=1)
        cand=ev[(ev>S/2+sh-0.25)&(ev<S/2+sh+0.25)]
        g_th=(l1-l2)*J/S
        print('eps=%.1f dl=%.2f g_th=%.4f | doblete numerico cerca de |+,0>,|-,1>:'%(eps,l1-l2,g_th),np.round(cand,4),'split/2=%.4f'%((cand[1]-cand[0])/2 if len(cand)>=2 else np.nan))
