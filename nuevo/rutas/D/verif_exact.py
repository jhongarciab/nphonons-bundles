import numpy as np, sys, time
sys.path.insert(0,'/home/user/nphonons-bundles/nuevo/bundles/src'); import bundles_core as bc
import sistemas_D as S
n=2
pts={'A real kap, Om=0.03':(S.SYS[list(S.SYS)[0]],0.03,None),
     'A kap=1e-4, Om=0.03':(S.SYS[list(S.SYS)[0]],0.03,1e-4),
     'B kap=4e-6 Om=0.012':(S.SYS[list(S.SYS)[1]],0.012,4e-6),
     'B kap=1e-4 Om=0.012':(S.SYS[list(S.SYS)[1]],0.012,1e-4),
     'C kap=6.3e-3 Om=0.3':(S.SYS[list(S.SYS)[2]],0.3,6.3e-3),
     'C kap=6.3e-3 Om=0.15':(S.SYS[list(S.SYS)[2]],0.15,6.3e-3)}
for k,((wb,lam,kap,gam,gph,al,_),Om,kk) in pts.items():
    l,ka,g,gp=lam/wb,(kk if kk else kap/wb),gam/wb,gph/wb
    t=time.time(); r=bc.exact_point(n,l,Om,ka,g,gp,N=9)
    print(f"{k:24s} lam={l:.4f} kap={ka:.1e} g={g:.1e} gph={gp:.1e} | exacto P2={r['P']:.3f} J={r['flux']:.3e} ({r['flux']*wb*1e6:.3e} Hz) | cerrada P={bc.P_closed(l,Om,ka,g,gp,n):.3f} J={bc.flux_closed(l,Om,ka,g,gp,n):.3e} [{time.time()-t:.0f}s]",flush=True)
wb,lam,kap,gam,gph,al,_=S.SYS[list(S.SYS)[5]]; l,g,gp=lam/wb,gam/wb,gph/wb
for kk,Om in [(1e-3,0.3),(3e-3,0.3),(6e-3,0.3),(1e-3,0.15)]:
    r=bc.exact_point(n,l,Om,kk,g,gp,N=9); print(f"D2 kap={kk:.0e} Om={Om}: exacto P2={r['P']:.3f} J={r['flux']:.2e} ({r['flux']*wb*1e6:.2e} Hz) cerrada P={bc.P_closed(l,Om,kk,g,gp,n):.3f} J={bc.flux_closed(l,Om,kk,g,gp,n):.2e}")
