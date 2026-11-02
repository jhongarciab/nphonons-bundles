"""Ruta D: sistemas qubit+modo con acoplo longitudinal. Convencion: H = d s+s + b+b + lam s+s (b+b+) + Om(s+s+), omega_b=1.
lam = coeficiente de s+s=(1+sz)/2 ; si la fuente da g_z*sz*(b+b+) => lam=2 g_z.  gam=1/T1, gph = 2/T2 - gam (coherencia decae a (gam+gph)/2).
Todas las frecuencias en Hz/2pi (MHz) -> se normalizan por omega_b/2pi."""
import numpy as np, sys
sys.path.insert(0, '/home/user/nphonons-bundles/nuevo/bundles/src')
import bundles_core as bc
from scipy.optimize import brentq
tpi = 2*np.pi
def rate_MHz(T_s): return 1/(tpi*T_s)/1e6          # 1/T -> (1/T)/2pi en MHz

# Potts 2407.02024: T1=664 ns, T2*=235 ns, kappa_int 28.0 kHz + kappa_ext 88.6 kHz, g0=11.9 MHz, wb=4347 MHz, alpha=-388 MHz
g1 = rate_MHz(664e-9); g2 = 2*rate_MHz(235e-9) - g1
SYS = {
 # nombre: (wb MHz, lam MHz, kappa MHz, gam MHz, gph MHz, alpha MHz o None, nota)
 'A Potts (transmon+SQUID, modo EM) [medido]': (4347, 11.9, 0.1166, g1, g2, 388, 'medido'),
 'B Richer17 diseno (gzx=53 MHz, lam=2gzx, wr=7 GHz; coherencia=Potts)': (7000, 106., 0.1166, g1, g2, 250, 'diseno'),
 'C Naseem 2508.10500 (2gz, wm=100 MHz, kq=100 kHz, gph=0)': (100, 12., 15e-6, 0.1, 0.0, None, 'propuesta'),
 'C2 Naseem con gph=2gam': (100, 12., 15e-6, 0.1, 0.2, None, 'propuesta'),
 'D Nongthombam 2508.17105 (fluxonium+SiN, 2gz=0.36 MHz)': (6, 0.36, 5e-6, 2.0, 1.8, None, 'propuesta'),
 'D2 Nongthombam acoplo + coherencia fluxonium Gerashchenko (T1=22.4us,T2=17.1us) [combinacion mia]': (6, 0.36, 5e-6, rate_MHz(22.4e-6), 2*rate_MHz(17.1e-6)-rate_MHz(22.4e-6), None, 'combinacion'),
 'D3 como D2 pero gph=S*sigma_Phi (sigma_Phi=42.8 uPhi0 efectivo de Potts, S=7.2 GHz/Phi0) [CALCULO]': (6, 0.36, 5e-6, rate_MHz(22.4e-6), 7.2e3*42.8e-6, None, 'combinacion'),
 'D4 como D3 con sigma_Phi=5 uPhi0 (flujo 8x mas silencioso)': (6, 0.36, 5e-6, rate_MHz(22.4e-6), 7.2e3*5e-6, None, 'objetivo'),
 'E QD GHz (baseline grupo)': (1.0, 0.002, 1e-3, 0.09, 0.18, None, 'ref'),
}
def Jbest(lam, gam, gph, n=2, X=0.9, Omcap=0.3, kap=None):
    """Maximiza J (cerrado) sobre kappa (si kap None) y Om <= min(Omcap, Om_max(P>=X))."""
    kaps = np.logspace(-6, -1, 51) if kap is None else [kap]
    best = (0, None, None, None)
    for k in kaps:
        try: Om = bc.Om_max(k, X, lam, gam, gph, n, Omcap=Omcap)
        except ValueError: continue
        if np.isnan(Om): continue
        if np.isinf(Om): Om = Omcap
        J = bc.flux_closed(lam, Om, k, gam, gph, n)
        if J > best[0]: best = (J, k, Om, bc.P_closed(lam, Om, k, gam, gph, n))
    return best
if __name__ == '__main__':
    n = 2
    print(f"{'sistema':70s} lam/wb  kap/wb  gam/wb  gph/wb | real: Om  J/wb  J[Hz]  P | opt(kap,Om) | factor lam p/ J>=1e-5")
    for name, (wb, lam, kap, gam, gph, al, nota) in SYS.items():
        l, k, g, gp = lam/wb, kap/wb, gam/wb, gph/wb
        cap_hw = 0.3 if al is None else min(0.3, al/3/wb)
        # real: kappa real, Om = Om_max(P>=0.9) o, si P no llega, Om=0.05 (P resultante)
        Om = bc.Om_max(k, 0.9, l, g, gp, n, Omcap=cap_hw)
        if np.isnan(Om): Om = min(0.05, cap_hw); flag = '(P<0.9)'
        else:
            flag = ''
            if np.isinf(Om): Om = cap_hw
        Jr = bc.flux_closed(l, Om, k, g, gp, n); Pr = bc.P_closed(l, Om, k, g, gp, n)
        Jo, ko, Oo, Po = Jbest(l, g, gp, n, Omcap=cap_hw)
        # factor en lam (gam,gph,wb fijos, kappa y Om optimos) para J>=1e-5
        fac = np.nan
        for s in np.logspace(0, 3, 61):
            if l*s > 0.3: break
            if Jbest(l*s, g, gp, n, Omcap=cap_hw)[0] >= 1e-5: fac = s; break
        print(f'cap_hw={cap_hw:.3f}', end=' '); print(f"{name[:70]:70s} {l:.2e} {k:.1e} {g:.1e} {gp:.1e} | {Om:.3f} {Jr:.2e} {Jr*wb*1e6:.2e} {Pr:.3f}{flag} | k*={ko} Om={Oo} J={Jo:.2e} ({Jo*wb*1e6:.2e} Hz) P={Po} | x{fac:.1f}")
