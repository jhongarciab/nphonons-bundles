from figstyle import *
d=np.load('../data/D6_trajectories.npz',allow_pickle=True); meta=eval(str(d['meta']).replace('null','None'))
fig=plt.figure(figsize=(7.0,5.0)); gs=fig.add_gridspec(2,2,height_ratios=[1.0,0.9],hspace=0.28,wspace=0.12,width_ratios=[1.25,1.0])
# ---------- (a) esquema ----------
ax=fig.add_subplot(gs[0,0]); ax.set_xlim(0,10); ax.set_ylim(0,5); clean(ax); panel(ax,'a')
bracket_cavity(ax,2.4,2.0,w=4.4,h=2.4,color=C['blue'])
ax.text(4.6,0.45,r'phononic cavity: $\omega_b,\ \kappa$',ha='center',color=C['blue'],fontsize=9)
sphere(ax,4.6,2.55,0.5); ax.text(4.6,2.55,'QD',ha='center',va='center',color='w',fontsize=7)
ax.text(4.6,1.45,r'$\lambda\,\sigma^\dagger\sigma\,(b+b^\dagger)$',ha='center',color=C['red'],fontsize=9)
arrow(ax,(1.5,4.35),(4.05,2.95),color=C['purp'],lw=2.0,ms=10); ax.text(1.3,4.6,r'laser $\Omega,\ \omega_L$',color=C['purp'],fontsize=9)
for dx in (0.0,0.5):
    wavy(ax,(7.0+dx,2.4),(7.45+dx,2.4),amp=0.08,n=2,color=C['green'],lw=1.6)
arrow(ax,(8.05,2.4),(9.7,2.4),color=C['green'],lw=1.8,ms=9); ax.text(8.6,2.75,r'$n$-phonon',color=C['green'],ha='center',fontsize=8.5); ax.text(8.6,1.95,'bundle',color=C['green'],ha='center',fontsize=8.5)
wavy(ax,(4.75,3.0),(5.9,4.45),amp=0.07,n=5,color=C['red'],lw=1.5); ax.text(6.1,4.5,r'photon ($\gamma$)',color=C['red'],fontsize=8.5)
# ---------- (b) diagrama de niveles (polaron) ----------
ax=fig.add_subplot(gs[0,1]); ax.set_xlim(-0.2,4.3); ax.set_ylim(-0.7,5.3); clean(ax); panel(ax,'b')
xg,xx=0.3,2.9; w=0.95
for m in range(5):
    ax.plot([xg,xg+w],[m,m],color='k',lw=1.6); ax.plot([xx,xx+w],[m+0.35,m+0.35],color=C['blue'],lw=1.6)
    ax.text(xg-0.06,m,rf'$|g,{m}\rangle$',ha='right',va='center',fontsize=7.5); ax.text(xx+w+0.06,m+0.35,rf'$|\tilde x,{m}\rangle$',ha='left',va='center',fontsize=7.5)
arrow(ax,(xg+w*0.5,0.03),(xx+w*0.5,2.32),color=C['mag'],lw=2.0,ms=10); ax.text(1.12,1.62,r'$\Omega_n$',color=C['mag'],fontsize=10,rotation=0)
arrow(ax,(xg+w*0.85,0.02),(xx+w*0.15,1.33),color=C['gray'],lw=1.0,ls='--',ms=7); arrow(ax,(xg+w*0.85,0.02),(xx+w*0.15,3.33),color=C['gray'],lw=1.0,ls='--',ms=7)
ax.text(1.35,3.75,r'off-resonant: $n\pm1$',color=C['gray'],fontsize=7.5)
arrow(ax,(xx+w*0.5,2.32),(xx+w*0.5,0.4),color=C['green'],lw=1.2,ms=8,ls=':'); ax.text(xx+w*0.58,0.78,r'$n\kappa$',color=C['green'],fontsize=8)
arrow(ax,(xx+w*0.25,0.4),(xg+w*0.85,-0.02),color=C['red'],lw=1.6,ms=8); ax.text(1.95,-0.5,r'$\gamma$ (Franck-Condon)',color=C['red'],fontsize=7.5,ha='center')
ax.text(1.0,4.95,r'$\delta_L=\lambda^2-n+\epsilon_n$',fontsize=9)
# ---------- (c) registro de detecciones ----------
ax=fig.add_subplot(gs[1,:]); ax.set_xlim(0,20000); ax.set_ylim(-0.4,3.6)
ax.set_yticks([0.5,1.5,2.5]); ax.set_yticklabels([r'$n=3$',r'$n=2$',r'$n=1$']); ax.set_xlabel(r'time $\omega_b t$'); panel(ax,'c','tr')
for row,(n,Y) in enumerate(((1,2.5),(2,1.5),(3,0.5))):
    ph=np.array(d[f'n{n}_phonon_t0']); po=np.array(d[f'n{n}_photon_t0']); kap=meta['cases'][str(n)]['kap']
    ax.plot([0,20000],[Y,Y],color='0.8',lw=0.6,zorder=0)
    for t in ph: ax.plot([t,t],[Y,Y+0.42],color=C['green'],lw=1.5)
    for t in po: ax.plot([t,t],[Y-0.42,Y],color=C['red'],lw=1.5)
    # agrupar bundles (hueco < 5/kappa)
    tau=5/kap; grp=[]; 
    for t in ph:
        if grp and t-grp[-1][-1]<tau: grp[-1].append(t)
        else: grp.append([t])
    for g in grp:
        if len(g)>=2: ax.plot([g[0]-60,g[-1]+60],[Y+0.52]*2,color=C['orange'],lw=2.2,solid_capstyle='butt'); ax.text(np.mean(g),Y+0.62,f'{len(g)}',ha='center',fontsize=7,color=C['orange'])
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],color=C['green'],lw=2,label='phonon detection'),Line2D([0],[0],color=C['red'],lw=2,label='photon (radiative decay)'),Line2D([0],[0],color=C['orange'],lw=2.4,label=r'cluster, gap $<5/\kappa$ (number = phonons)')],loc='upper left',ncol=3,fontsize=7,bbox_to_anchor=(0.0,1.14))
finish(fig); fig.savefig('fig1.png'); print('ok')
