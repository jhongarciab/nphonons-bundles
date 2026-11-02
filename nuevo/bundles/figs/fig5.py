import sys; sys.path.insert(0,'../src')
from figstyle import *
fig=plt.figure(figsize=(7.0,3.0)); gs=fig.add_gridspec(1,3,wspace=0.50)
m=np.load('../data/C_mc.npz',allow_pickle=True)
# (a) pureza operativa vs tau_c
ax=fig.add_subplot(gs[0]); sel=[(0,C['mag'],r'$\kappa/\gamma=15$'),(4,C['blue'],r'$\kappa/\gamma=50$'),(7,C['green'],r'$\kappa/\gamma=500$')]
for i,c,lab in sel:
    kap=m['kap'][i]; tau=m['taus'][i]*kap if np.max(m['taus'][i])>50 else m['taus'][i]
    ax.errorbar(tau,m['P_mc'][i],yerr=m['P_mc_err'][i],fmt='o',color=c,ms=4,capsize=2,label=lab); ax.plot(tau,m['P_pred'][i],'-',color=c,lw=1.0)
    ax.axhline(m['P_ep'][i],color=c,ls='--',lw=0.9)
ax.set_xlabel(r'clustering window $\kappa\tau_c$'); ax.set_ylabel(r'$P_2^{\rm op}(\tau_c)$'); ax.set_ylim(0.45,1.22); ax.set_xticks([2,5,10]); panel(ax,'a','bl'); ax.legend(loc='lower right',fontsize=6.8)
ax.text(0.04,0.985,'dashed: episode purity\nsolid: operational closed form',transform=ax.transAxes,fontsize=6.3,va='top')
# (b) piso de Franck-Condon
f=np.load('../data/D5_floor.npz',allow_pickle=True); lam=f['lam']
ax=fig.add_subplot(gs[1]); ax.plot(lam,f['floor_exp'],'k-',lw=1.2,label=r'$e^{-\lambda^2}$'); ax.plot(lam,f['floor_ep'],'k:',lw=1.0,label=r'$1/(2-e^{-\lambda^2})$')
ax.plot(lam,f['n2_Pex'],'o',color=C['blue'],ms=4,label=r'$n=2$ exact'); ax.plot(lam,f['n2_Pcl'],'-',color=C['blue'],lw=1.0,label=r'closed form'); ax.plot(lam,f['n2_Pcyc'],'s',mfc='none',color=C['red'],ms=4,label='per cycle')
ax.plot(lam,f['n3_Pex'],'^',color=C['green'],ms=4,label=r'$n=3$')
ax.set_xlabel(r'$\lambda$'); ax.set_ylabel(r'$\max_\kappa P_n$'); ax.set_ylim(0.70,1.0); panel(ax,'b','bl'); ax.legend(loc='lower center',fontsize=5.6,ncol=3,columnspacing=0.6,handletextpad=0.3,handlelength=1.3,bbox_to_anchor=(0.5,1.0),borderaxespad=0.2)
# (c) gamma
g=np.load('../data/D7_dephasing.npz',allow_pickle=True)
ax=fig.add_subplot(gs[2]); ax.semilogx(g['gam'],g['P_vs_gam'],'o',color=C['blue'],mfc='none',ms=4,label='$P_2$ exact'); ax.semilogx(g['gam'],g['Pcl_vs_gam'],'-',color=C['blue'],lw=1.2,label='closed form')
ax.set_xlabel(r'$\gamma/\omega_b$'); ax.set_ylabel(r'$P_2$'); panel(ax,'c','bl'); ax2=ax.twinx(); ax2.loglog(g['gam'],g['J_vs_gam'],'s',color=C['orange'],mfc='none',ms=3.6); ax2.loglog(g['gam'],g['Jcl_vs_gam'],'-',color=C['orange'],lw=1.2); ax2.set_ylabel(r'$J/\omega_b$',color=C['orange'])
ax.legend(loc='lower center',fontsize=6.3,ncol=2,bbox_to_anchor=(0.5,1.0),borderaxespad=0.2,handlelength=1.4); ax2.tick_params(axis='y',colors=C['orange'],direction='in')
finish(fig); fig.savefig('fig5.png'); print('ok')
