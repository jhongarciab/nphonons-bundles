import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from bl import *; from rate import *
lam0,gam,gph=0.14,2e-4,4e-4
def flux(delta,lam,Om,kap,N=9):
    H,c,L,Jb,Jg=build_pol(delta,lam,Om,kap,gam,gph,N); rho=qt.steadystate(H,c); b,s=ops(N); return kap*qt.expect(b.dag()*b,rho)
# 1) resonancia
print('## resonancia: maximo del flujo exacto vs analitico (2o orden) vs Bin-III'); lines=[]
for n in (2,3):
  for lam in (0.08,0.14,0.2):
    for Om in (0.05,0.1,0.2,0.3):
        kap=0.003; d0=-n+eps_n(lam,Om,n)
        r=minimize_scalar(lambda d:-flux(d,lam,Om,kap),bounds=(d0-0.02,d0+0.02),method='bounded',options=dict(xatol=1e-7))
        gc=(n*kap+gam+gph)/2; binIII=-np.sqrt(n*n-4*Om**2)
        print(f'n={n} lam={lam} Om={Om}: num {r.x:+.5f} | analit(2o) {d0:+.5f} err {r.x-d0:+.1e} = {(r.x-d0)/gc:+.2f} gc | Bin-III(sin lam^2) err {r.x-binIII:+.1e}',flush=True)
# 2) mapas con curva analitica
for n,Oms,kaps,N in [(2,np.geomspace(0.02,0.25,8),np.geomspace(6e-4,2e-2,8),8),(3,np.geomspace(0.05,0.4,6),np.geomspace(6e-4,2e-2,6),9)]:
    E=np.load(f'data/P{n}_exact.npy'); Cg=lambda O,k:P_closed(lam0,O,k,gam,gph,n)
    Ec=np.array([[Cg(O,k) for O in Oms] for k in kaps]); rel=abs(Ec-E)/E
    print(f'## n={n}: |P_cerrada-P_exacta|/P: max {rel.max():.3f} media {rel.mean():.3f}; fraccion<5% {np.mean(rel<.05):.2f}, <10% {np.mean(rel<.10):.2f}; impureza (1-P) err rel mediano {np.median(abs((1-Ec)-(1-E))/(1-E)):.2f}')
    Og=np.geomspace(Oms[0],Oms[-1],80); kg=np.geomspace(kaps[0],kaps[-1],80); C=np.array([[Cg(O,k) for O in Og] for k in kg])
    fig,ax=plt.subplots(figsize=(5.4,4.3)); pc=ax.pcolormesh(Oms,kaps,E,shading='nearest',cmap='viridis',vmin=max(0.3,E.min()),vmax=1)
    cs=ax.contour(Og,kg,C,levels=[0.8,0.9,0.95],colors='w',linestyles='--',linewidths=1.2); ax.clabel(cs,fmt='%.2f',fontsize=7)
    om=[Om_max(lam0,k,gam,gph,n,0.9) for k in kg]; ax.plot(om,kg,'r-',lw=1.6,label=rf'$\Omega_{{\max}}(\kappa)$: $P_{n}^{{\rm cerr}}=0.9$')
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(Oms[0],Oms[-1]);ax.set_ylim(kaps[0],kaps[-1]);ax.set_xlabel(r'$\Omega/\omega_b$');ax.set_ylabel(r'$\kappa/\omega_b$')
    ax.set_title(rf'$P_{n}$ (color: exacta; rayas: cerrada), $\lambda={lam0}$',fontsize=9);plt.colorbar(pc,label=rf'$P_{n}$');ax.legend(fontsize=7,loc='lower right');fig.tight_layout();fig.savefig(f'fig/mapa_P{n}_Om_kap.png',dpi=150);plt.close()
# 3) frente de Pareto pureza-flujo (cerrado) + puntos exactos
fig,axs=plt.subplots(1,2,figsize=(9,3.8))
for ax,n,Oex in zip(axs,(2,3),((0.05,0.1,0.2),(0.1,0.2,0.4))):
    Og=np.geomspace(0.01,0.5,60); kg=np.geomspace(4e-4,2e-2,60)
    pts=np.array([(flux_closed(lam0,O,k,gam,gph,n),P_closed(lam0,O,k,gam,gph,n)) for k in kg for O in Og])
    ax.scatter(pts[:,0],pts[:,1],s=2,c='0.8')
    o=np.argsort(-pts[:,0]); f=pts[o]; best=-1; env=[]
    for J,P in f:
        if P>best: env.append((J,P)); best=P
    env=np.array(env); ax.plot(env[:,0],env[:,1],'r-',label='frente (cerrada)')
    # puntos exactos sobre el frente: para flujo objetivo, mejor (Om,kap) cerrado, evaluado exacto
    for Jt in np.geomspace(max(env[-1,0]*1.5,1e-6),env[0,0]*0.9,5):
        i=np.argmin(abs(pts[:,0]-Jt)+1e3*(pts[:,1]<np.interp(Jt,env[::-1,0],env[::-1,1])-0.003)) if True else 0
        k=kg[i//len(Og)];O=Og[i%len(Og)]
        Pe=episodes_pol(-n+eps_n(lam0,O,n),lam0,O,k,gam,gph,N=8 if n==2 else 9,K=2*n+3)[n]
        H,c,L,Jb,Jg=build_pol(-n+eps_n(lam0,O,n),lam0,O,k,gam,gph,8 if n==2 else 9); rho=qt.steadystate(H,c); b,s=ops(8 if n==2 else 9); Fe=k*qt.expect(b.dag()*b,rho)
        ax.plot(Fe,Pe,'ko',ms=4,label='exacto' if Jt==np.geomspace(max(env[-1,0]*1.5,1e-6),env[0,0]*0.9,5)[0] else None)
        print(f'Pareto n={n}: Om={O:.3f} kap={k:.4f}: cerrada (J={pts[i,0]:.2e},P={pts[i,1]:.3f}) exacta (J={Fe:.2e},P={Pe:.3f})',flush=True)
    ax.set_xscale('log');ax.set_xlabel(r'flujo de fonones $J/\omega_b$');ax.set_ylabel(rf'pureza $P_{n}$');ax.set_title(rf'n={n}, $\lambda={lam0}$',fontsize=9);ax.legend(fontsize=7)
fig.tight_layout();fig.savefig('fig/pareto.png',dpi=150)
