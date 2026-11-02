import numpy as np
from bl import *; from rate import *
lam,gam,gph=.14,2e-4,4e-4;n=2
print('Om kap | T_exact CV2_exact | T_formula CV2_formula (renovacion 2 etapas 1/Gamma0, 1/gam)')
for Om,kap in [(.1,.003),(.05,.003),(.2,.003),(.1,.01),(.1,.006)]:
    d=-n+eps_n(lam,Om,n); H,c,L,Jb,Jg=build_pol(d,lam,Om,kap,gam,gph,8)
    S=16; I=np.eye(S*S,dtype=complex); L0=L-Jb-Jg; tr=np.eye(S).reshape(-1,order='F')
    G=Jg@np.linalg.solve(-(L0+Jb),I); w,v=np.linalg.eig(G); rs=v[:,np.argmax(w.real)]; rs/=tr@rs
    X=np.linalg.inv(-(L0+Jb)); T1=(tr@X@rs).real; T2=(2*tr@X@X@rs).real; cv2=T2/T1**2-1
    G0=Gamma_j(lam,Om,kap,gam,gph,n,0); Tf=1/G0+1/gam; cvf=(1/G0**2+1/gam**2)/Tf**2
    print(f'{Om} {kap} | {T1:.0f} {cv2:.3f} | {Tf:.0f} {cvf:.3f}')
