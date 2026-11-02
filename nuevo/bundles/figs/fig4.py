import sys; sys.path.insert(0,'../src')
from figstyle import *
lam,gam=0.14,2e-4
d=np.load('../data/D4_local_vs_dressed.npz',allow_pickle=True); kap=d['kap']
fig=plt.figure(figsize=(7.0,5.4)); gs=fig.add_gridspec(2,2,hspace=0.42,wspace=0.38)
# (a) pureza por ciclo vs kappa
ax=fig.add_subplot(gs[0,0]); ax.plot(kap,d['n2_dressed_cycle_P'],color=C['blue'],label='dressed'); ax.plot(kap,d['n2_local_cycle_P'],color=C['red'],label='local')
ax.plot(kap,d['n2_Ploc_pred'],'--',color=C['red'],lw=1.0,label=r'$P_{\rm dressed}\,e^{-\kappa\lambda^2/\gamma}$')
ax.errorbar([0.003],[0.803],yerr=[0.004],fmt='o',color='k',mfc=C['blue'],ms=4.5,capsize=2,label='MC (dressed)'); ax.errorbar([0.003],[0.615],yerr=[0.004],fmt='s',color='k',mfc=C['red'],ms=4.5,capsize=2,label='MC (local)')
ax.axvline(lam**2,color='k',ls=':',lw=0.9); ax.text(lam**2*0.9,0.05,r'$\kappa=\lambda^2$',rotation=90,ha='right',fontsize=7.5)
ax.set_xscale('log'); ax.set_xlabel(r'$\kappa/\omega_b$'); ax.set_ylabel(r'$P_2$ per radiative cycle'); ax.set_ylim(0,1.28); panel(ax,'a','bl'); ax.legend(loc='upper left',fontsize=6.3,ncol=2,bbox_to_anchor=(0.10,1.0),columnspacing=0.8)
# (b) exceso de fonones
ax=fig.add_subplot(gs[0,1]); exc=d['n2_local_kmean']-d['n2_dressed_kmean']
ax.loglog(kap,d['n2_cloud_pred'],color='k',lw=1.0,label=r'$\kappa\lambda^2/\gamma$'); ax.loglog(kap,exc,'o',mfc='none',color=C['red'],ms=4.2,label='local $-$ dressed')
ax.axvline(lam**2,color='k',ls=':',lw=0.9); ax.set_xlabel(r'$\kappa/\omega_b$'); ax.set_ylabel(r'extra phonons per cycle'); panel(ax,'b','tl'); ax.legend(loc='lower right',fontsize=7)
# (c),(d) prueba con bano microscopico (datos de verif/C2.log)
ks=[0.003,0.01,0.03]; xs=np.arange(3)
mean={'micro':[2.0215,2.0160,1.9626],'dressed':[2.0217,2.0137,1.9657],'local':[2.3145,2.9912,4.9049]}
P2={'micro':[0.7975,0.9068,0.8472],'dressed':[0.8050,0.9088,0.8994],'local':[0.6095,0.4518,0.2351]}
sty={'micro':('D',C['green'],'microscopic bath (extrap. $\\kappa_c\\to0$)'),'dressed':('+','k','dressed master eq.'),'local':('x',C['red'],'local master eq.')}
for pos,(ax,dat,yl,lab,lo) in enumerate(((fig.add_subplot(gs[1,0]),mean,'emitted phonons per cycle','c',(1.8,5.2)),(fig.add_subplot(gs[1,1]),P2,r'$P_2$ per radiative cycle','d',(0.1,1.0)))):
    for k,(mk,c,l) in sty.items():
        off={'micro':-0.12,'dressed':0.0,'local':0.12}[k]
        ax.plot(xs+off,dat[k],mk,color=c,ms=7 if mk!='D' else 5.5,mew=1.5 if mk!='D' else 0.6,mec='k' if mk=='D' else c,ls='none',label=l.replace('\\\\','\\') if pos==0 else None)
    ax.set_xticks(xs); ax.set_xticklabels([rf'$\kappa={k}$' for k in ks]); ax.set_ylabel(yl); ax.set_ylim(*lo); panel(ax,lab,'tl')
    ax.axvspan(1.5,2.5,color='0.9',zorder=0); ax.text(2.0,lo[1]-0.07*(lo[1]-lo[0]),r'$\kappa>\lambda^2$',ha='center',fontsize=7)
fig.axes[2].legend(loc='upper center',fontsize=6.3,bbox_to_anchor=(0.5,0.82),ncol=1)
finish(fig); fig.savefig('fig4.png'); print('ok')
