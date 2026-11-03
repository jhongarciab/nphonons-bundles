import numpy as np, sys
from figstyle import *
from scipy.linalg import expm, eigh
P=np.load('../P/datos/figuras_P.npz',allow_pickle=True)
# ================= Fig P1: esquema, niveles, registro =================
def fig1():
    fig=plt.figure(figsize=(7.0,5.2)); gs=fig.add_gridspec(2,2,height_ratios=[0.72,1.0],hspace=0.30,wspace=0.12,width_ratios=[1.2,1.0])
    ax=fig.add_subplot(gs[0,0]); ax.set_xlim(0,10); ax.set_ylim(0,5); clean(ax); panel(ax,'a')
    # transmon (circulo con cruz) acoplado a resonador de cristal fononico, guia y detector
    sphere(ax,1.5,2.6,0.55,c0=(1.0,0.8,0.5),c1=(0.85,0.4,0.05)); ax.text(1.5,2.6,'transmon',ha='center',va='center',fontsize=6.5,color='w')
    bracket_cavity(ax,3.4,2.6,w=2.8,h=1.7,color=C['blue']); ax.text(4.8,2.6,'phononic\ncrystal\nmode $\\omega_m$',ha='center',va='center',color=C['blue'],fontsize=7.5)
    wavy(ax,(2.05,2.6),(3.4,2.6),amp=0.08,n=5,color=C['orange']); ax.text(2.7,3.0,r'$g$',color=C['orange'],fontsize=10,ha='center')
    wavy(ax,(6.2,2.6),(8.0,2.6),amp=0.09,n=6,color=C['green']); ax.text(7.1,2.0,r'waveguide, $\kappa_m$',color=C['green'],fontsize=7.5,ha='center')
    ax.add_patch(FancyBboxPatch((8.1,2.2),1.5,0.8,boxstyle='round,pad=0.05',fc='#e6f4ea',ec=C['green'])); ax.text(8.85,2.6,'detector',ha='center',va='center',fontsize=7.5,color=C['green'])
    arrow(ax,(0.5,4.4),(1.35,3.2),color=C['purp'],lw=2.0,ms=10); ax.text(0.2,4.6,r'two $\pi$ pulses  $g\to e\to f$',color=C['purp'],fontsize=7.5)
    ax.text(5.0,1.1,r'pair of phonons from one $|f\rangle$ excitation: $|f,0\rangle\to|g,2\rangle$',ha='center',fontsize=7.8)
    # niveles
    ax=fig.add_subplot(gs[0,1]); ax.set_xlim(0,4.2); ax.set_ylim(-0.4,4.0); clean(ax); panel(ax,'b')
    ax.plot([0.3,1.5],[3.0,3.0],color=C['mag'],lw=2); ax.text(0.28,3.0,r'$|f,0\rangle$',ha='right',va='center',fontsize=8)
    ax.plot([2.5,3.7],[3.0,3.0],color=C['blue'],lw=2); ax.text(3.75,3.0,r'$|g,2\rangle$',ha='left',va='center',fontsize=8)
    ax.plot([1.4,2.6],[1.7,1.7],color='0.55',lw=1.4,ls='--'); ax.text(2.0,1.35,r'$|e,1\rangle$ (virtual)',ha='center',fontsize=7.5,color='0.4')
    arrow(ax,(1.5,2.95),(2.5,2.95),color=C['red'],lw=2.0,both=True,ms=10); ax.text(2.0,3.25,r'$V\simeq 4g^2/|\alpha|$ (exact: Fig. 2a)',ha='center',fontsize=7.2,color=C['red'])
    arrow(ax,(1.1,2.9),(1.9,1.8),color='0.5',lw=1.0,ls=':'); arrow(ax,(2.1,1.8),(2.9,2.9),color='0.5',lw=1.0,ls=':'); ax.text(1.95,0.75,r'$\omega_q=\omega-\alpha/2$',ha='center',fontsize=7.5)
    arrow(ax,(3.1,2.9),(3.1,0.2),color=C['green'],lw=1.2,ms=8,ls='-'); ax.text(3.35,1.2,r'$2\kappa_m,\ \kappa_m$',color=C['green'],fontsize=7.5)
    # registro temporal
    ax=fig.add_subplot(gs[1,:]); t=P['ts_t']; ens=P['ts_ens']
    for k,(c,lab) in enumerate(((C['k'],r'$P_g$'),(C['blue'],r'$P_e$'),(C['mag'],r'$P_f$'))):
        ax.plot(t*1e3,ens[k],color=c if k else '0.2',lw=1.3,label=lab)
    ax.axvspan(0,2*P['ts_tp']*1e3,color=C['purp'],alpha=0.18,lw=0); ax.text(2*P['ts_tp']*1e3/2,0.5,'pulses',rotation=90,ha='center',va='center',fontsize=7,color=C['purp'])
    ax.set_xlabel('time (ns)'); ax.set_ylabel('transmon population'); ax.set_xlim(0,420); ax.set_ylim(0,1.05)
    for tc in P['ts_traj_click_t']: ax.axvline(tc*1e3,color=C['green'],lw=1.2,ls='-',alpha=0.9)
    ax2=ax.twinx(); ax2.plot(t*1e3,ens[4],color=C['green'],lw=1.3,ls='--',label=r'$\langle n\rangle$ (mode)'); ax2.set_ylabel(r'phonons in the mode',color=C['green']); ax2.tick_params(axis='y',colors=C['green'],direction='in'); ax2.set_ylim(0,1.3)
    ax.legend(loc='center right',fontsize=7,ncol=1,bbox_to_anchor=(0.99,0.62)); panel(ax,'c','tr')
    ax.text(0.5,1.02,'ensemble average (published parameters + $\\kappa_m=10.8$ MHz); green lines: clicks of one trajectory',transform=ax.transAxes,ha='center',fontsize=7)
    finish(fig); fig.savefig('pair_fig1.png'); print('f1')
# ================= Fig P2: acoplo efectivo y escala =================
def Vnum(g,al,N=2,mdim=None):
    ndim=N+3; mdim=mdim or N+5
    # transmon operadores
    sig=np.diag(np.sqrt(np.arange(1,ndim)),1); a=np.diag(np.sqrt(np.arange(1,mdim)),1)
    Ek=lambda wq: np.array([k*wq+al*k*(k-1)/2 for k in range(ndim)])
    I_t=np.eye(ndim); I_m=np.eye(mdim)
    def H(wq):
        return (np.kron(np.diag(Ek(wq)),I_m)+np.kron(I_t,np.diag(np.arange(mdim)*1.0))*1.0+g*(np.kron(sig.T,a)+np.kron(sig,a.T)))
    # frecuencias en MHz con omega=0 en el marco: usamos omega=wm; ponemos wm=2300 MHz
    return H
def Vsplit(g,al,N=2,wm=2300.):
    ndim=N+3; mdim=N+5
    sig=np.diag(np.sqrt(np.arange(1,ndim)),1); a=np.diag(np.sqrt(np.arange(1,mdim)),1)
    I_t=np.eye(ndim); I_m=np.eye(mdim)
    def H(wq):
        Ek=np.array([k*wq+al*k*(k-1)/2 for k in range(ndim)])
        return np.kron(np.diag(Ek),I_m)+wm*np.kron(I_t,np.diag(np.arange(mdim)*1.0))+g*(np.kron(sig.T,a)+np.kron(sig,a.T))
    i1=lambda k,m: k*mdim+m
    wres=wm-al*(N-1)/2; best=1e9
    for wq in np.linspace(wres-8,wres+8,321):
        Hq=H(wq); idx=[i1(N,0),i1(0,N)]
        # espacio de misma excitacion: autovalores de bloque con N excitaciones
        exc=[(k,m) for k in range(ndim) for m in range(mdim) if k+m==N]; ii=[i1(k,m) for k,m in exc]
        ev=np.linalg.eigvalsh(Hq[np.ix_(ii,ii)])
        # dos autovalores mas cercanos a E(N,0) y E(0,N)
        e0=0.5*(Hq[idx[0],idx[0]]+Hq[idx[1],idx[1]]); o=np.argsort(abs(ev-e0)); gap=abs(ev[o[0]]-ev[o[1]]); best=min(best,gap)
    return best/2
def fig2():
    fig=plt.figure(figsize=(7.0,3.1)); gs=fig.add_gridspec(1,3,wspace=0.55,width_ratios=[1.05,1.0,1.0])
    ax=fig.add_subplot(gs[0]); al=-138.; gs_=np.linspace(2,25,24); Vn=np.array([Vsplit(g,al) for g in gs_])
    ax.plot(gs_,0.25*(np.sqrt(al**2/4+16*gs_**2)-abs(al)/2),color=C['blue'],label=r'exact $3\times3$ block'); ax.plot(gs_,Vn,'o',mfc='none',color=C['blue'],ms=4,label='full diagonalization')
    ax.plot(gs_,4*gs_**2/abs(al),'--',color=C['gray'],label=r'$4g^2/|\alpha|$')
    ax.axvline(15.2,color='k',ls=':',lw=0.9); ax.text(15.0,0.3,'PnC\n(15.2 MHz)',ha='right',fontsize=6.5)
    ax.set_xlabel(r'$g/2\pi$ (MHz)'); ax.set_ylabel(r'$V/2\pi$ (MHz)'); panel(ax,'a','tl'); ax.legend(loc='upper left',fontsize=6.2,bbox_to_anchor=(0.12,0.98))
    ax=fig.add_subplot(gs[1]); Ns=[2,3,4]; V=[Vsplit(15.2,-138.,N) for N in Ns]
    ax.semilogy(Ns,V,'o-',color=C['blue'],ms=5,label=r'transmon ladder, $V_N$'); 
    lam=np.array([0.002,0.006]); 
    for l,c in zip(lam,(C['red'],C['orange'])):
        bin_=[2300*0.3*np.exp(-l**2/2)*l**n/np.sqrt(np.math.factorial(n)) if hasattr(np,'math') else 2300*0.3*np.exp(-l**2/2)*l**n/np.sqrt(float(np.prod(np.arange(1,n+1)))) for n in Ns]
        ax.semilogy(Ns,bin_,'s--',color=c,mfc='none',ms=4.5,label=rf'Bin, $\lambda/\omega={l}$, $\Omega/\omega=0.3$')
    ax.set_xticks(Ns); ax.set_xlabel('bundle size $N$'); ax.set_ylabel('effective rate (MHz)'); panel(ax,'b','bl'); ax.legend(loc='upper right',fontsize=5.8,bbox_to_anchor=(1.0,1.0))
    ax=fig.add_subplot(gs[2]); ratio=[V[1]/V[0],V[2]/V[1]]; ax.bar([0,1],ratio,color=C['blue'],width=0.5); ax.set_xticks([0,1]); ax.set_xticklabels([r'$V_3/V_2$',r'$V_4/V_3$']); ax.set_ylabel('ratio per extra phonon'); ax.set_ylim(0,0.3)
    for i,r in enumerate(ratio): ax.text(i,r+0.008,f'{r:.2f}',ha='center',fontsize=7)
    ax.axhline(15.2/138.,color=C['gray'],ls='--',lw=0.9); ax.text(1.45,15.2/138.+0.008,r'$g/|\alpha|$',ha='right',fontsize=7,color=C['gray']); panel(ax,'c','tr')
    finish(fig); fig.savefig('pair_fig2.png'); print('f2',V,ratio)
# ================= Fig P3: eficiencia de par =================
def eta_formula(kq,km,G): y1=kq/2; y2=km; V2=2*G**2; return 1/(1+y1/y2+y1*(y1+y2)/V2)
def eta_num(kq,km,G):
    V=np.sqrt(2)*G; M=np.array([[-1j*kq/2,V],[V,-1j*km]],dtype=complex)   # amplitudes: |e,0>, |g,2> (perdida de |g,2> a 2 km -> amplitud km)
    # eta = 2 km * int |b|^2 dt con a(0)=1
    w,Vv=np.linalg.eig(-1j*M); c=np.linalg.solve(Vv,np.array([1,0],dtype=complex))
    tot=0
    for i in range(2):
        for j in range(2):
            tot+=c[i]*np.conj(c[j])*Vv[1,i]*np.conj(Vv[1,j])*(-1.0/(w[i]+np.conj(w[j])))
    return (2*km*tot).real
def fig3():
    fig=plt.figure(figsize=(7.0,3.2)); gs=fig.add_gridspec(1,2,wspace=0.45,width_ratios=[1.15,1])
    ax=fig.add_subplot(gs[0]); kq=np.geomspace(0.01,5,120); km=np.geomspace(0.1,20,120); X,Y=np.meshgrid(km,kq)
    Z=eta_formula(Y,X,1.0); im=ax.pcolormesh(X,Y,Z,shading='gouraud',cmap='viridis',vmin=0,vmax=1,rasterized=True)
    cs=ax.contour(X,Y,Z,levels=[0.5,0.9,0.95],colors='w',linestyles='--',linewidths=1.0); ax.clabel(cs,fmt=lambda v:f'{v:.2f}',fontsize=6.5)
    ax.axvline(np.sqrt(2),color=C['red'],lw=1.4,label=r'$\kappa_m=\sqrt{2}\,G$')
    kqm=np.array([0.05,0.2,0.5,1.4]); kmm=np.array([0.5,1,np.sqrt(2),2,4]); 
    for a_ in kqm:
        for b_ in kmm: ax.plot([b_],[a_],'o',mfc='none',mec='w',ms=3.5,mew=0.8)
    ax.plot([1.43],[1.39],'*',color=C['orange'],ms=10,mec='k',mew=0.6); ax.annotate('Naseem\n(cat regime)',(1.43,1.39),xytext=(3.5,0.35),fontsize=6.5,color='w',arrowprops=dict(arrowstyle='->',color='w',lw=0.7))
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$\kappa_m/G$'); ax.set_ylabel(r'$\kappa_q/G$'); panel(ax,'a'); cbar(fig,im,ax,r'pair efficiency $\eta$'); ax.legend(loc='upper left',fontsize=6.5,facecolor='k',labelcolor='w',framealpha=0.4,edgecolor='none',bbox_to_anchor=(0.0,0.90))
    ax=fig.add_subplot(gs[1])
    kmv=np.geomspace(0.15,15,160)
    for kqv,c in ((0.05,C['blue']),(0.2,C['green']),(0.5,C['orange']),(1.4,C['mag'])):
        ax.plot(kmv,eta_formula(kqv,kmv,1.0),color=c,lw=1.3,label=rf'$\kappa_q/G={kqv}$'); kk=np.array([0.5,1,np.sqrt(2),2,4,8]); ax.plot(kk,[eta_num(kqv,k_,1.0) for k_ in kk],'o',mfc='none',color=c,ms=4.2)
        ax.plot([np.sqrt(2)],[eta_formula(kqv,np.sqrt(2),1.0)],'*',color=c,ms=9,mec='k',mew=0.5)
    ax.set_xscale('log'); ax.set_xlabel(r'$\kappa_m/G$'); ax.set_ylabel(r'$\eta$'); ax.set_ylim(0,1.02); panel(ax,'b','bl'); ax.legend(loc='lower center',fontsize=6.0,ncol=4,bbox_to_anchor=(0.5,1.0),columnspacing=0.7,handlelength=1.2,borderaxespad=0.2)
    ax.text(0.97,0.04,'lines: closed form; symbols: exact\nintegration; $\\star$: $\\kappa_m^\\ast$',transform=ax.transAxes,fontsize=6.2,ha='right',va='bottom')
    finish(fig); fig.savefig('pair_fig3.png'); print('f3',max(abs(eta_formula(0.5,2.0,1.0)-eta_num(0.5,2.0,1.0)),0))
# ================= Fig P4: simulacion realista =================
def fig4():
    fig=plt.figure(figsize=(7.0,5.6)); gs=fig.add_gridspec(2,2,hspace=0.45,wspace=0.40)
    KM=P['KM']; ax=fig.add_subplot(gs[0,0]); i5=5
    ax.semilogx(KM,P['mapa_T1_1p14_P_Tw0p5'][:,12,2],'k:',lw=1.1,label=r'ideal $|f,0\rangle$, $T_1$ only')
    for key,c,ls,lab in (('P_km_pub_B',C['blue'],'-','published, B'),('P_km_pub_A',C['blue'],'--','published, A'),('P_km_imp_B',C['green'],'-','improved, B'),('P_km_imp_A',C['green'],'--','improved, A')):
        ax.semilogx(KM,P[key][:,i5,2],ls,color=c,lw=1.4,label=lab)
    ax.axvline(0.37,color=C['red'],ls=':',lw=1.0); ax.text(0.40,0.12,'measured mode\n($\\kappa/2\\pi=0.37$ MHz,\nno waveguide)',color=C['red'],fontsize=6.3)
    ax.set_xlabel(r'$\kappa_m/2\pi$ (MHz)'); ax.set_ylabel(r'$P(2)$ per cycle'); ax.set_ylim(0,1.0); panel(ax,'a','tl'); ax.legend(loc='lower right',fontsize=6.2,ncol=1,bbox_to_anchor=(1.0,0.0))
    # presupuesto
    ax=fig.add_subplot(gs[0,1])
    steps=['ideal\n(window)','+ $T_1$\nladder','+ $T_\\phi$','+ thermal\n($n_{th}$, $p_e$)','+ intrinsic\nloss']; pub=[0.9591,0.8609,0.8321,0.7906,0.7180]; imp=[0.9591,0.9471,0.9444,0.9348,0.9255]
    x=np.arange(5); ax.plot(x,pub,'o-',color=C['blue'],label='published'); ax.plot(x,imp,'s-',color=C['green'],label='improved')
    ax.set_xticks(x); ax.set_xticklabels(steps,fontsize=6.0); ax.set_ylabel(r'$P(2)$'); ax.set_ylim(0.65,1.0); panel(ax,'b','bl'); ax.legend(loc='upper right',fontsize=7)
    for xi,p in zip(x,pub): ax.text(xi,p-0.022,f'{p:.2f}',ha='center',fontsize=6.3,color=C['blue'])
    # T1 y n_th
    ax=fig.add_subplot(gs[1,0]); T1=P['T1']
    ax.semilogx(T1,P['P2_vs_T1_only'],'o-',color=C['gray'],mfc='none',label=r'$T_1$ only'); ax.semilogx(T1,P['P2_vs_T1_full'],'o-',color=C['blue'],label='published rest'); ax.semilogx(T1,P['P2_vs_T1_IMPrest'],'s-',color=C['green'],label='improved rest')
    ax.set_xlabel(r'qubit $T_1$ ($\mu$s)'); ax.set_ylabel(r'$P(2)$'); ax.set_ylim(0.6,1.0); panel(ax,'c','bl'); ax.legend(loc='lower right',fontsize=6.5)
    ax=fig.add_subplot(gs[1,1]); lab=['published\nactive reset','published\npassive reset','improved\nactive reset','improved\npassive reset']; flux=[1.11,0.115,1.43,0.018]
    ax.bar(range(4),flux,color=[C['blue'],C['blue'],C['green'],C['green']],alpha=[1,0.5,1,0.5][0]); 
    for i,b in enumerate(ax.patches): b.set_alpha(1 if i%2==0 else 0.45)
    ax.set_yscale('log'); ax.set_xticks(range(4)); ax.set_xticklabels(lab,fontsize=6.0); ax.set_ylabel('pair rate (MHz)'); ax.set_ylim(0.01,4); panel(ax,'d','tr')
    for i,f in enumerate(flux): ax.text(i,f*1.25,f'{f:g}',ha='center',fontsize=7)
    ax.text(0.03,0.95,'ideal detection ($\\eta_{det}=1$)',transform=ax.transAxes,fontsize=6.3,va='top')
    finish(fig); fig.savefig('pair_fig4.png'); print('f4')
if __name__=='__main__':
    which=sys.argv[1:] or ['1','2','3','4']
    for w in which: {'1':fig1,'2':fig2,'3':fig3,'4':fig4}[w]()
