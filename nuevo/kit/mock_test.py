from figstyle import *
fig=plt.figure(figsize=(7.0,5.4)); gs=fig.add_gridspec(2,3,width_ratios=[1.25,0.95,1.1],hspace=0.30,wspace=0.55)
# (a) esquema
ax=fig.add_subplot(gs[0,0:2]); ax.set_xlim(0,10); ax.set_ylim(0.2,4.4); clean(ax); panel(ax,'a','tl')
bracket_cavity(ax,3.2,1.0,w=3.2,h=1.5,color=C['blue']); ax.text(4.8,1.0,r'$\omega_b,\ \kappa$',ha='center',va='center',color=C['blue'],fontsize=11)
sphere(ax,3.9,3.1,0.5); sphere(ax,5.7,3.1,0.5); ax.text(3.9,3.1,'QD$_1$',ha='center',va='center',color='w',fontsize=7); ax.text(5.7,3.1,'QD$_2$',ha='center',va='center',color='w',fontsize=7)
wavy(ax,(4.4,3.1),(5.2,3.1),color=C['orange']); ax.text(4.8,3.45,r'$J$',color=C['orange'],ha='center',fontsize=11)
straight(ax,(3.9,2.6),(3.9,1.75),color=C['red']); straight(ax,(5.7,2.6),(5.7,1.75),color=C['red']); ax.text(3.55,2.15,r'$\lambda_1$',color=C['red'],ha='right',fontsize=10); ax.text(6.05,2.15,r'$\lambda_2$',color=C['red'],fontsize=10)
arrow(ax,(0.6,3.6),(3.2,3.25),color=C['purp'],lw=2.0,ms=10); ax.text(0.9,3.85,r'$\Omega,\ \omega_L$',color=C['purp'],fontsize=10)
wavy(ax,(6.4,1.0),(8.6,1.0),color=C['green'],n=7); ax.text(7.5,1.3,'phonon out',color=C['green'],ha='center',fontsize=8)
# (b) niveles
ax=fig.add_subplot(gs[0,2]); ax.set_xlim(-0.8,2.4); ax.set_ylim(-0.4,4.2); clean(ax); panel(ax,'b')
ys=[0,1.5,2.0,3.4]; levels(ax,ys[:1],x0=0.0,w=0.9,labels=[r'$|g,0\rangle$']); levels(ax,[1.5],x0=0.0,w=0.9,labels=[r'$|-,0\rangle$'],colors=[C['blue']]); levels(ax,[2.0],x0=1.2,w=0.9,labels=None,colors=[C['mag']]); ax.text(2.15,2.0,r'$|+,0\rangle$',fontsize=8,va='center')
levels(ax,[3.4],x0=0.0,w=0.9,labels=[r'$|+,1\rangle$'],colors=[C['mag']])
arrow(ax,(0.3,0.05),(0.3,1.45),color=C['purp'],lw=1.5); arrow(ax,(0.65,1.55),(0.65,3.35),color=C['green'],both=True,lw=1.3); ax.text(0.8,2.4,r'$g$',color=C['green'],fontsize=10)
# (c) mapa 2D
ax=fig.add_subplot(gs[1,0:2]); x=np.linspace(-3,3,300); y=np.linspace(0.02,0.3,200); X,Y=np.meshgrid(x,y)
Z=np.log10(0.02+((X-0.2*np.sin(8*Y))**2)*(1+8*Y)); im=ax.pcolormesh(X,Y,Z,cmap=CM_G2,shading='auto',rasterized=True)
ax.plot(0.2*np.sin(8*y),y,'k--',lw=1.2); ax.set_xlabel(r'$\Delta/\kappa$'); ax.set_ylabel(r'$\Omega/\omega_b$'); panel(ax,'c'); cbar(fig,im,ax,r'$\log_{10} g^{(2)}(0)$')
# (d) cortes
ax=fig.add_subplot(gs[1,2]); th=np.linspace(-2,2,400)
for k,(c,lab) in enumerate(zip([C['mag'],C['blue'],C['green']],[r'$\lambda=0.10$',r'$0.14$',r'$0.20$'])):
    f=-1.2-3*np.exp(-((th-0.1*k)/0.06)**2)+0.3*np.cos(2*np.pi*th)*0.4; ax.plot(th,f,color=c,label=lab)
    ax.plot(th[::22],f[::22]+0.02*np.random.default_rng(k).standard_normal(len(th[::22])),'o',mfc='none',color=c,ms=3.4)
vline(ax,0.1,r'$\theta_{\rm opt}=0.10\pi$',ypos=0.30); ax.set_xlabel(r'$\theta/\pi$'); ax.set_ylabel(r'$\log_{10}g^{(2)}(0)$'); panel(ax,'d'); ax.legend(loc='upper right',fontsize=7,ncol=1,bbox_to_anchor=(1.0,0.78))
fig.savefig('mock_test.png'); print('ok')
