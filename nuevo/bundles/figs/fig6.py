import sys; sys.path.insert(0,'../src')
from figstyle import *
import bundles_core as bc, os, time
n=2; fn='../data/R_requirements_fine.npz'
lams=np.geomspace(2e-3,0.3,36); gams=np.geomspace(1e-5,3.0,36); kaps=np.geomspace(1e-4,0.1,50); Oms=np.geomspace(0.01,0.3,30)
if not os.path.exists(fn):
    t=time.time(); Jm=np.full((len(gams),len(lams)),np.nan)
    for i,g in enumerate(gams):
        if g>2e-2: continue                       # fuera de banda lateral resuelta: sin solucion
        for j,l in enumerate(lams):
            bj=np.nan
            for k in kaps:
                for O in Oms:
                    gc=bc.gamma_c(n,0,k,g,2*g)
                    if bc.omega_eff(n,l,O)/gc>0.5: continue
                    if bc.P_closed(l,O,k,g,2*g,n)<0.9: continue
                    J=bc.flux_closed(l,O,k,g,2*g,n)
                    if np.isnan(bj) or J>bj: bj=J
            Jm[i,j]=bj
    np.savez(fn,lams=lams,gams=gams,J=Jm,note='J max (cerrada v2) con P2>=0.9, xi<0.5, Omega<=0.3; gphi=2 gamma; kappa 50 pts [1e-4,0.1], Omega 30 pts [0.01,0.3]; NaN=sin solucion; filas con gamma>0.02 no evaluadas'); print('computed',time.time()-t)
d=np.load(fn,allow_pickle=True); Jm=d['J']; L=d['lams']; G=d['gams']
fig=plt.figure(figsize=(4.9,3.8)); ax=fig.add_subplot(111)
Z=np.log10(np.where(np.isfinite(Jm),Jm,np.nan)); X,Y=np.meshgrid(L,G)
ax.set_facecolor('#cfcfcf')
im=ax.pcolormesh(X,Y,np.ma.masked_invalid(Z),shading='auto',cmap='viridis',vmin=-7.0,vmax=-3.3,rasterized=True)
cs=ax.contour(X,Y,np.nan_to_num(Z,nan=-9),levels=[-6,-5,-4],colors='w',linewidths=1.0,linestyles=['-','--',':'])
for lev,txt,pos in ((-6,r'$10^{-6}$',(0.0036,3e-5)),(-5,r'$10^{-5}$',(0.0105,8e-5)),(-4,r'$10^{-4}$',(0.04,8e-4))): ax.text(pos[0],pos[1],txt,color='w',fontsize=7,rotation=0)
ax.axvline(0.1,color='k',ls='-.',lw=0.9); ax.text(0.108,1.25e-5,r'validated $\rightarrow$',fontsize=6.3,color='k')
ax.text(0.0022,1.25e-5,r'$\leftarrow$ extrapolated ($\lambda<0.1$)',fontsize=6.0,color='w')
ax.axhline(1e-2,color=C['red'],ls=':',lw=0.9); ax.text(0.0022,1.35e-2,r'sideband unresolved ($\gamma\gtrsim0.01\,\omega_b$): no solution',fontsize=6.0,color=C['red'])
ax.plot([0.1],[2e-4],'*',color=C['orange'],ms=10,mec='k',mew=0.6); ax.annotate('Bin et al.\n(proposal)',(0.1,2e-4),xytext=(-44,-16),textcoords='offset points',fontsize=6.5,color='w')
ax.plot([0.0021],[0.092],'o',color=C['red'],ms=6,mec='k',mew=0.6,clip_on=False); ax.annotate('QD, Spinnler et al.\n(measured)',(0.0021,0.092),xytext=(7,-4),textcoords='offset points',fontsize=6.3)
ax.annotate(r'QD wire, Kettler et al.: $\gamma/\omega_b\approx480$',xy=(0.22,2.8),xytext=(0.0022,0.9),fontsize=6.0,arrowprops=dict(arrowstyle='->',lw=0.7))
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'exciton-phonon coupling $\lambda/\omega_b$'); ax.set_ylabel(r'radiative rate $\gamma/\omega_b$'); cbar(fig,im,ax,r'$\log_{10}(J_{\max}/\omega_b)$ s.t. $P_2\geq0.9$')
ax.set_xlim(2e-3,0.3); ax.set_ylim(1e-5,3.0)
finish(fig); fig.savefig('fig6.png'); 
# cifras para el texto
import numpy as _n
for thr in (-6,-5,-4):
    ok=np.isfinite(Z)&(Z>=thr); lmin=L[ok.any(axis=0)].min(); gmax=G[ok.any(axis=1)].max(); print(f'J>=1e{thr}: lambda_min={lmin:.4f}, gamma_max={gmax:.2e}')
print('ok')
