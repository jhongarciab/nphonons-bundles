import sys; sys.path.insert(0,'../src')
from figstyle import *
import bundles_core as bc
lam,Om0,kap,gam,gph=0.14,0.1,0.003,2e-4,4e-4
d1=np.load('../data/D1_spectrum.npz',allow_pickle=True)
fig=plt.figure(figsize=(7.0,5.6)); gs=fig.add_gridspec(2,2,height_ratios=[1,1],hspace=0.36,wspace=0.30)
# ---- (a) espectro completo ----
ax=fig.add_subplot(gs[0,:]); dL=d1['dL']; J=d1['J']
ax.semilogy(dL,np.maximum(J,1e-9),color='k',lw=1.0)
for n,p in zip(d1['pos_n'],d1['pos_dL']):
    ax.axvline(p,color=COL[n%len(COL)],ls='--',lw=0.9); ax.text(p+0.03,2.5e-3,rf'$n={n}$',color=COL[n%len(COL)],fontsize=8,rotation=0,va='top')
ax.axvline(lam**2,color=C['gray'],ls=':',lw=0.9); ax.text(lam**2+0.03,2.5e-3,'carrier',color=C['gray'],fontsize=7.5,va='top')
ax.set_xlim(-4.65,0.65); ax.set_ylim(1e-8,4e-3); ax.set_xlabel(r'laser detuning $\delta_L/\omega_b$'); ax.set_ylabel(r'phonon flux $J/\omega_b$'); panel(ax,'a','bl')
ax.text(0.99,0.05,r'$\lambda=0.14,\ \Omega=0.1,\ \kappa=0.003,\ \gamma=2\times10^{-4},\ \gamma_\phi=4\times10^{-4}$',transform=ax.transAxes,ha='right',va='bottom',fontsize=7.5)
ax.text(0.13,0.05,r'dashed: $\delta_L=\lambda^2-n+\epsilon_n$',transform=ax.transAxes,fontsize=7.5)
# ---- (b) forma de linea n=2: analitica vs exacta ----
ax=fig.add_subplot(gs[1,0]); n=2
for Om,c in zip((0.05,0.1,0.2),(C['blue'],C['mag'],C['green'])):
    gc=bc.gamma_c(n,0,kap,gam,gph); d0=bc.delta_res(n,lam,Om); x=np.linspace(-5,5,41)*gc
    Jn=np.array([bc.exact_point(n,lam,Om,kap,gam,gph,dL=d0+xx,kind='cycle',flux=True)['flux'] for xx in x])
    xs=np.linspace(-5,5,400)*gc; G=2*Om**2*bc.dmat(lam,n,0)**2*gc/(xs**2+gc**2); Ja=(n+lam**2)/(1/G+1/gam)
    ax.plot(xs/gc,Ja*1e4,color=c,lw=1.4,label=rf'$\Omega={Om}$'); ax.plot(x/gc,Jn*1e4,'o',mfc='none',color=c,ms=3.8)
ax.set_xlabel(r'$(\delta_L-\delta_{\rm res})/\gamma_c$'); ax.set_ylabel(r'$J\ (10^{-4}\,\omega_b)$'); ax.legend(loc='upper right',fontsize=7.5); panel(ax,'b')
ax.text(0.97,0.04,r'lines: closed form, symbols: master eq. ($n=2$)',transform=ax.transAxes,fontsize=7,ha='right',va='bottom')
# ---- (c) error de la posicion de la resonancia vs Omega ----
ax=fig.add_subplot(gs[1,1]); rows=[]
for l in open('../data/resonance.txt').read().splitlines()[1:]:
    p=l.replace('|',' ').replace('(',' ').replace(')',' ').split()
    rows.append([float(x) for x in (p[0],p[1],p[2],p[3],p[4],p[6])])
rows=np.array(rows)
for n,mk,ls in ((2,'o','-'),(3,'s','--')):
    sel=(rows[:,0]==n)&(rows[:,1]==0.14); Om=rows[sel,2]; dnum=rows[sel,3]; gc=bc.gamma_c(n,0,kap,gam,gph)
    e2=np.abs(rows[sel,4]); eB=np.abs(rows[sel,5]); eS=np.abs(dnum-(lam**2-n+2*Om**2/n))/gc
    ax.plot(Om,np.maximum(e2,1e-3),mk+ls,color=C['blue'],ms=4,label=r'$\epsilon_n$ (this work)' if n==2 else None)
    ax.plot(Om,np.maximum(eB,1e-3),mk+ls,color=C['orange'],ms=4,mfc='none',label=r'Bin et al. (III) $+\lambda^2$' if n==2 else None)
    ax.plot(Om,np.maximum(eS,1e-3),mk+ls,color=C['gray'],ms=4,mfc='none',label=r'$2\Omega^2/n$ only' if n==2 else None)
ax.set_yscale('log'); ax.set_xlabel(r'$\Omega/\omega_b$'); ax.set_ylabel(r'$|\delta_{\rm num}-\delta_{\rm th}|/\gamma_c$'); ax.set_ylim(1e-3,4); panel(ax,'c','tl')
ax.text(0.98,0.05,r'solid: $n=2$, dashed: $n=3$',transform=ax.transAxes,ha='right',fontsize=7); ax.legend(loc='upper left',fontsize=7,bbox_to_anchor=(0.14,0.99))
finish(fig); fig.savefig('fig2.png'); print('ok')
