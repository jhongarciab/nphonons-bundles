import numpy as np
h=6.626e-34;k=1.381e-23
print("n_th(f,T):  f[MHz] | T=10mK  20mK  50mK | T para n_th=0.02")
for f in (4.4,6,100,300,1000,4347):
    r=[1/np.expm1(h*f*1e6/(k*T)) for T in (0.01,0.02,0.05)]
    T02=h*f*1e6/(k*np.log(1+1/0.02))
    print(f"{f:7.1f} | {r[0]:8.3g} {r[1]:8.3g} {r[2]:8.3g} | {T02*1e3:.2f} mK")
