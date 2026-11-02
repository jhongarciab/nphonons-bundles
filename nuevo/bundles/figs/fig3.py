import sys; sys.path.insert(0,'../src')
from figstyle import *
import bundles_core as bc
from matplotlib.colors import LogNorm
gam,gph=2e-4,4e-4
def load(n,lam): return np.load(f'../data/D2_n{n}_lam{lam}.npz',allow_pickle=True)
fig=plt.figure(figsize=(7.0,5.9)); gs=fig.add_gridspec(2,2,hspace=0.40,wspace=0.62)
def pmap(ax,n,lam,lab,xlabel=True,showlab=True):
    d=load(n,lam); Om,kap=d['Om'],d['kap']; X,Y=np.meshgrid(Om,kap)
    im=ax.pcolormesh(X,Y,d['P_exact'],shading='gouraud',cmap='viridis',vmin=0.75,vmax=1.0,rasterized=True)
    Of,kf=d['Om_fine'],d['kap_fine']; cs=ax.contour(Of,kf,d['P_closed_fine'],levels=[0.9,0.95],colors='w',linestyles='--',linewidths=1.1)
    ax.clabel(cs,fmt=lambda v:f'{v:.2f}',fontsize=6.5,inline=True)
    for key,ls_ in (('Ommax_X0p95','-'),):
        om=d[key]; ok=np.isfinite(om)&(om<0.299); ax.plot(om[ok],kf[ok],color=C['red'],lw=1.6,label=r'$\Omega_{\max}(\kappa)$, $P\geq0.95$')
    ax.contour(Om,kap,d['xi'],levels=[0.5],colors='k',linestyles=':',linewidths=1.0)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(Om[0],Om[-1]); ax.set_ylim(kap[0],kap[-1]); ax.set_xlabel(r'$\Omega/\omega_b$'); ax.set_ylabel(r'$\kappa/\omega_b$'); panel(ax,lab)
    cb=cbar(fig,im,ax,rf'$P_{n}$'); return d
ax=fig.add_subplot(gs[0,0]); pmap(ax,2,0.14,'a'); ax.text(0.97,0.04,r'$n=2$',transform=ax.transAxes,ha='right',color='w',fontsize=8)
ax.legend(loc='upper left',bbox_to_anchor=(0.12,0.99),fontsize=6.5,facecolor='k',labelcolor='w',framealpha=0.35,frameon=True,edgecolor='none')
ax=fig.add_subplot(gs[0,1]); pmap(ax,3,0.14,'b'); ax.text(0.97,0.04,r'$n=3$',transform=ax.transAxes,ha='right',color='w',fontsize=8)
# (c) flujo exacto con contornos cerrados
ax=fig.add_subplot(gs[1,0]); d=load(2,0.14); X,Y=np.meshgrid(d['Om'],d['kap'])
im=ax.pcolormesh(X,Y,np.log10(d['J_exact']),shading='gouraud',cmap='magma',rasterized=True,vmin=-5.0,vmax=-3.35)
cs=ax.contour(d['Om_fine'],d['kap_fine'],np.log10(d['J_closed_fine']),levels=[-4.5,-4.0,-3.7],colors='w',linestyles='--',linewidths=1.0); ax.clabel(cs,fmt=lambda v:f'{v:.1f}',fontsize=6.5)
ax.contour(d['Om'],d['kap'],d['xi'],levels=[0.5],colors='c',linestyles=':',linewidths=1.0)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$\Omega/\omega_b$'); ax.set_ylabel(r'$\kappa/\omega_b$'); panel(ax,'c'); cbar(fig,im,ax,r'$\log_{10}(J/\omega_b)$'); ax.text(0.97,0.04,r'$n=2$',transform=ax.transAxes,ha='right',color='w',fontsize=8)
# (d) cortes en kappa a Om=0.1 para tres lambda
ax=fig.add_subplot(gs[1,1])
for lam,c in ((0.1,C['blue']),(0.14,C['mag']),(0.2,C['green'])):
    d=load(2,lam); j=int(np.argmin(abs(d['Om']-0.1))); Om=d['Om'][j]; kap=d['kap']
    ax.plot(kap,d['P_exact'][:,j],'o',mfc='none',color=c,ms=3.6)
    kf=np.geomspace(kap[0],2.2e-2,260); ax.plot(kf,[bc.P_closed(lam,Om,k,gam,gph,2) for k in kf],color=c,lw=1.4,label=rf'$\lambda={lam}$')
    ks,Ps=bc.kappa_star(2,lam,Om,gam,gph); ax.plot([ks],[Ps],'*',color=c,ms=9,mec='k',mew=0.5)
ax.set_xscale('log'); ax.set_xlim(4e-4,2.2e-2); ax.set_xlabel(r'$\kappa/\omega_b$'); ax.set_ylabel(r'$P_2$'); ax.set_ylim(0.68,1.0); panel(ax,'d','bl'); ax.legend(loc='lower right',fontsize=7.5)
ax.text(0.97,0.52,r'$\Omega\approx0.1$; lines: closed form,'+'\n'+r'symbols: master eq., $\star$: $\kappa^\ast$',transform=ax.transAxes,fontsize=7,va='top',ha='right')
finish(fig); fig.savefig('fig3.png'); print('ok')
