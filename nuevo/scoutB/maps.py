import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, time, sys
from scipy.optimize import minimize_scalar
from bl import *; from rate import *
lam0,gam,gph=0.14,2e-4,4e-4
def exactP(n,lam,Om,kap,N):
    d=-n+eps_n(lam,Om,n); return episodes_pol(d,lam,Om,kap,gam,gph,N=N,K=2*n+3)[n]
# ---- mapa (Omega,kappa) ----
for n,N,Oms,kaps in [(2,8,np.geomspace(0.02,0.25,8),np.geomspace(6e-4,2e-2,8)),(3,9,np.geomspace(0.05,0.4,6),np.geomspace(6e-4,2e-2,6))]:
    t=time.time(); E=np.zeros((len(kaps),len(Oms))); 
    for i,k in enumerate(kaps):
        for j,O in enumerate(Oms): E[i,j]=exactP(n,lam0,O,k,N)
    np.save(f'data/P{n}_exact.npy',E)
    Og=np.geomspace(Oms[0],Oms[-1],60); kg=np.geomspace(kaps[0],kaps[-1],60)
    C=np.array([[P_closed(lam0,O,k,gam,gph,n) for O in Og] for k in kg])
    R=np.array([[episodes_rate(lam0,O,k,gam,gph,n,M=n+7,K=2*n+3)[n] for O in Og] for k in kg])
    np.save(f'data/P{n}_closed.npy',C); np.save(f'data/P{n}_rate.npy',R)
    # error numerico
    Ec=np.array([[P_closed(lam0,O,k,gam,gph,n) for O in Oms] for k in kaps]); rel=np.abs(Ec-E)/E
    imp_rel=np.abs((1-Ec)-(1-E))/(1-E)
    print(f'n={n} exact grid {E.shape}: max|dP|/P={rel.max():.3f} mean={rel.mean():.3f}; impureza mediana rel err={np.median(imp_rel):.2f}; t={time.time()-t:.0f}s',flush=True)
    fig,ax=plt.subplots(figsize=(5.2,4.2))
    pc=ax.pcolormesh(Oms,kaps,E,shading='nearest',cmap='viridis',vmin=max(0.5,E.min()),vmax=1)
    cs=ax.contour(Og,kg,C,levels=[0.8,0.9,0.95],colors='w',linestyles='--',linewidths=1.2)
    ax.clabel(cs,fmt=lambda v:f'{v:.2f}',fontsize=7)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel(r'$\Omega/\omega_b$');ax.set_ylabel(r'$\kappa/\omega_b$')
    ax.set_title(rf'$P_{n}$ exacta (color) vs cerrada (rayas), $\lambda$={lam0}'); plt.colorbar(pc,label=rf'$P_{n}$')
    # curva Omega_max(kappa) para X=0.95 analitica
    for X,st in ((0.9,'r-'),):
        om=[Om_max(lam0,k,gam,gph,n,X) for k in kg]; ax.plot(om,kg,st,lw=1.5,label=rf'$\Omega_{{max}}(\kappa)$ para $P_{n}\geq${X}')
    ax.set_xlim(Oms[0],Oms[-1]);ax.set_ylim(kaps[0],kaps[-1]);ax.legend(fontsize=7,loc='lower right')
    fig.tight_layout();fig.savefig(f'fig/mapa_P{n}_Om_kap.png',dpi=150);plt.close()
# ---- resonancia: flujo exacto vs delta; maximo vs curva analitica ----
def flux(delta,lam,Om,kap,n,N=9):
    H,c,L,Jb,Jg=build_pol(delta,lam,Om,kap,gam,gph,N); rho=qt.steadystate(H,c); b,s=ops(N); return kap*qt.expect(b.dag()*b,rho)
rows=[]
for n in (2,3):
  for lam in (0.08,0.14,0.2):
    for Om in (0.05,0.1,0.2,0.3):
        kap=0.003; d0=-n+eps_n(lam,Om,n)
        r=minimize_scalar(lambda d:-flux(d,lam,Om,kap,n),bracket=(d0-2e-3,d0,d0+2e-3),tol=1e-7)
        w=(n*kap+gam+gph)  # anchura ~ gc
        # corte: precision del maximo
        rows.append((n,lam,Om,d0,r.x,r.x-d0,(r.x-d0)/Omn(lam,Om,n),(r.x-d0)/(-n+0*0 - (-n+ (-n+lam**0*0)*0) ) if False else np.nan, -n+np.sqrt(0)))
        # formula de Bin regimen III para comparar
        binIII=-np.sqrt(n*n-4*Om**2); rows[-1]=rows[-1][:7]+(r.x-binIII,)
        print(f'n={n} lam={lam} Om={Om}: analit {d0:+.5f} num {r.x:+.5f} err {r.x-d0:+.1e} ({(r.x-d0)/(gam+gph+n*kap)*2:+.3f} anchuras) | Bin-III err {r.x-binIII:+.1e}',flush=True)
np.save('data/res_rows.npy',np.array(rows))
