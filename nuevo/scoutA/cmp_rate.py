from qdm import *; from rate_model import rate_model
print('Comparación numérico (Lindblad, Nb=18, d=0) vs modelo analítico de escalera oscura; eps=0,S=1,l=(.14,0),gph=4e-4')
print('Om gam kap | num: n g2 Q | modelo: n g2 Q')
for (Om,gam,kap) in ((0.07,0.01,0.003),(0.05,0.01,0.005),(0.05,0.002,0.003),(0.1,0.002,0.005),(0.1,2e-4,0.003),(0.03,0.03,0.003)):
    r,_,_=ss(Nb=18,eps=0,J=0.5,wL=0.5,Om=Om,gam=gam,kap=kap)
    m=rate_model(Om,gam,kap)
    print('%g %g %g | %.3f %.3f %.3f | %.3f %.3f %.3f'%(Om,gam,kap,r['n'],r['g2'],r['Q'],m[0],m[1],m[2]))
