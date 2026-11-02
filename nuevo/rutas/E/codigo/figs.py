import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.rcParams.update({'font.size':9,'figure.dpi':130})
def Ge(km,kq):
    a=kq/2; disc=((km-a)/2)**2-2.0; return 2*((a+km)/2-np.real(np.sqrt(disc+0j)))
def omega_curve(KM,kq,x,qstar):
    out=[]
    for k in KM:
        tau=x/k; G_=Ge(k,kq); fac=tau-(1-np.exp(-G_*tau))/G_
        W=qstar/fac; out.append(np.sqrt(W*(kq+4/k)/4))     # W=4 Om^2/(kq+4G^2/km)
    return np.array(out)
def mapa(ax_f,ax_p,kq,x=5.0,cb=True):
    d=np.load(f'mapa_kq{kq}_x{x}.npz'); KM,OM,out=d['KM'],d['OM'],d['out']
    P2=np.where(out[:,:,0]>1e-6,out[:,:,1],np.nan)
    im=ax_f.pcolormesh(KM,OM,out[:,:,0],norm=LogNorm(1e-3,2),shading='auto',cmap='viridis')
    ax_f.contour(KM,OM,P2,[0.95],colors='w',linewidths=1.5)
    im2=ax_p.pcolormesh(KM,OM,P2,vmin=0.5,vmax=1,shading='auto',cmap='magma')
    cs=ax_p.contour(KM,OM,P2,[0.95],colors='c',linewidths=1.5)
    qs=(0.9865-0.95)/0.9865; oc=omega_curve(KM,kq,x,qs)
    for ax in (ax_f,ax_p):
        ax.set_xscale('log'); ax.set_yscale('log'); ax.axvline(np.sqrt(2),color='r',ls='--',lw=1); ax.axvline(2*np.sqrt(2),color='orange',ls=':',lw=1.2)
        ax.plot(KM,oc,'w--' if ax is ax_f else 'c--',lw=1); ax.set_xlabel(r'$\kappa_m/G$'); ax.set_ylim(OM[0],OM[-1])
    ax_f.set_ylabel(r'$\Omega/G$'); ax_f.set_title(rf'tasa de clics $/G$, $\kappa_q/G={kq}$'); ax_p.set_title(rf'pureza $P_2$ (ventana $\tau_c={x:g}/\kappa_m$)')
    return im,im2
fig,ax=plt.subplots(2,3,figsize=(12.5,7.2))
for r,kq in enumerate([0.05,1.4]):
    im,im2=mapa(ax[r,0],ax[r,1],kq)
    plt.colorbar(im,ax=ax[r,0]); plt.colorbar(im2,ax=ax[r,1])
# panel eta
KM=np.geomspace(0.3,30,200); KQ=np.geomspace(0.003,3,200); A,B=np.meshgrid(KM,KQ)
eta=1/(1+B/4*(A+B/2+2/A)); im=ax[0,2].pcolormesh(KM,KQ,eta,vmin=0.3,vmax=1,shading='auto',cmap='cividis'); plt.colorbar(im,ax=ax[0,2])
for s,c in [(1/0.95-1,'w'),(1/0.9-1,'y')]:
    Bk=KM+2/KM; kqmax=-Bk+np.sqrt(Bk**2+8*s); ax[0,2].plot(KM,kqmax,c+'--',lw=1.3)
ax[0,2].set_xscale('log'); ax[0,2].set_yscale('log'); ax[0,2].axvline(np.sqrt(2),color='r',ls='--',lw=1); ax[0,2].set_xlabel(r'$\kappa_m/G$'); ax[0,2].set_ylabel(r'$\kappa_q/G$')
ax[0,2].set_title(r'$\eta_{\rm par}$ cerrada; --: $\eta=0.95$, $0.90$'); 
# comparacion con la figura del exacto: puntos de eta_check
pts=[(1.0,0.05),(1.414,0.05),(2.83,0.05),(6,0.05),(1.414,0.3),(3,1.4),(1.414,1.4)]
ax[0,2].plot(*zip(*pts),'ko',ms=3)
# panel transmon
R=json.load(open('transmon_res.json')); 
for name,v in R.items():
    if name.startswith('HBAR'): continue
    rr=np.array(v['rows']); ax[1,2].plot(rr[:,0],rr[:,1],label=name.replace('PnC',''))
ax[1,2].set_xscale('log'); ax[1,2].set_xlabel(r'$\kappa_m/2\pi$ [MHz]'); ax[1,2].set_ylabel(r'$P(M=2,\,L=0\,|\,|f,0\rangle)$'); ax[1,2].set_ylim(0.5,1.0); ax[1,2].legend(fontsize=6); ax[1,2].set_title('transmon $|f,0\\rangle\\to|g,2\\rangle$ (sin $g_z$)')
fig.tight_layout(); fig.savefig('../mapa_E.png'); print('ok')
