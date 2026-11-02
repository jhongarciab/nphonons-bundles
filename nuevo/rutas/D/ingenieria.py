"""Seccion 4: acoplo longitudinal flujo-mecanico lam = S_Phi * B*l*x_zpf  (S_Phi = d f_q/d Phi en Hz/Phi0, lam en Hz). [CALCULO]
Referencias medidas: Potts 2407.02024: S=26.0 GHz/Phi0, Phi_zpf=461 uPhi0 -> g0=12.0 MHz; gamma_phi=1.114 MHz (de T2*=235 ns, T1=664 ns)."""
import numpy as np
hbar=1.0546e-34; Phi0=2.0678e-15
S_potts=26e9; Phz_potts=461e-6; sig_eff=1.114e6/S_potts      # sigma_Phi efectivo en Phi0 (dephasing total/S)
print(f"Potts: sigma_Phi efectivo = {sig_eff*1e6:.1f} uPhi0 ; lam/gam_phi = {11.9/1.114:.1f}; lam/wb={11.9/4347:.4f}")
print("wb[MHz] m[pg] B[T] l[um] x_zpf[fm] Phi_zpf[uPhi0] | S_req(lam/wb=0.01)[GHz/Phi0] S_req/S_potts | gphi implied[MHz],gphi/wb | S_req(0.03)")
for wb,m,B,l in [(6,0.75,0.06,40),(6,0.75,0.5,40),(100,0.75,0.06,40),(100,1,0.1,40),(100,1,0.5,40),(300,1,0.1,40),(300,0.1,0.5,10)]:
    xz=np.sqrt(hbar/(2*m*1e-15*2*np.pi*wb*1e6)); Pz=B*l*1e-6*xz/Phi0
    out=[]
    for r in (0.01,0.03):
        S=r*wb*1e6/Pz; out.append(S)
    gphi=out[0]*sig_eff
    print(f"{wb:6.0f} {m:5.2f} {B:5.2f} {l:4.0f} {xz*1e15:8.1f} {Pz*1e6:10.1f} | {out[0]/1e9:10.1f} {out[0]/S_potts:6.2f} | {gphi/1e6:7.2f} {gphi/(wb*1e6):7.3f} | {out[1]/1e9:8.1f}")
# escalado a flujo fijo (modo EM, L fija): Phi_zpf ~ sqrt(w) -> lam/w ~ w^-1/2 ; mecanico: x_zpf~w^-1/2 -> lam/w ~ w^-3/2
print("Escalado mecanico lam/wb ~ wb^-3/2 (m,B,l,S fijos): de 6 MHz a 100 MHz cae un factor", (100/6)**1.5)
