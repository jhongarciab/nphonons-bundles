import json,glob,numpy as np, matplotlib
from matplotlib.ticker import ScalarFormatter
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.25})
O='../figs/'
d=np.load('../phi_hitchcock_fig4e.npy')
# Fig 1: Phi(detuning) digitalizado y mapeo a rho
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
ax[0].errorbar(d[:,0],d[:,1],yerr=[d[:,3],d[:,2]],fmt='o',color='C3',capsize=3,label='media ± desv. (digitalizado, Fig. 4e)')
ax[0].axhline(0,color='k',lw=.5); ax[0].set_xlabel('desintonía Δω/2π (MHz)'); ax[0].set_ylabel('Φ'); ax[0].legend(); ax[0].set_title('Hitchcock et al., Φ(Δω)')
Phi=np.linspace(0,0.95,100)
ax[1].plot(Phi,Phi,label='ρ = Φ (ruido blanco, f∝t)'); ax[1].plot(Phi,1-(1-Phi)**2,label='ρ = 1−(1−Φ)² (cuasi-estático, f∝t²)')
ax[1].axvspan(0.42-0.13,0.42+0.13,color='C3',alpha=.15,label='Φ medido (adyacentes)'); ax[1].axhline(0.5,color='gray',ls='--',lw=.8)
ax[1].text(0.02,0.52,'punto de equilibrio ρ=1/2',fontsize=8); ax[1].set_xlabel('Φ'); ax[1].set_ylabel('ρ (Pearson)'); ax[1].legend(fontsize=8); ax[1].set_title('Φ → ρ según el régimen de ruido')
plt.tight_layout(); plt.savefig(O+'fig1_phi_medido.png',dpi=160); plt.close()
# Fig 2: ganancia
rho=np.linspace(0,0.98,200)
fig,ax=plt.subplots(figsize=(4.8,3.6))
ax.plot(rho,1/(2*(1-rho)),label='ruido blanco: 1/[2(1−ρ)]'); ax.plot(rho,(2*(1-rho))**-.5,label='cuasi-estático: [2(1−ρ)]$^{-1/2}$')
ax.axhline(1,color='k',lw=.6); ax.set_ylim(0,6); ax.set_xlabel('ρ'); ax.set_ylabel('ganancia en tiempo de desfase relativo'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(O+'fig2_ganancia.png',dpi=160); plt.close()
# Fig 3: F_c vs g (T=100us) con baselines
res=json.load(open('design_eval_si.json'))
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
for fsr,a in ((3.3,ax[0]),(1.0,ax[1])):
    for T1q,c in ((28,'C0'),(100,'C1'),(300,'C2'),(1000,'C3')):
        pts=sorted([(r['g'],r['F_c']) for r in res if r['T']==100. and r['sc']=='S1_Phi0.42_white' and r['fsr']==fsr and r['T1q']==T1q])
        a.plot([p[0] for p in pts],[p[1] for p in pts],'o-',color=c,label=f'chequeo, T1q={T1q} µs')
    r0=[r for r in res if r['T']==100. and r['sc']=='S1_Phi0.42_white'][0]
    a.axhline(r0['B1_Func'],color='gray',ls=':',label='B1 indep., sin heraldar'); a.axhline(r0['B2_Func'],color='k',ls='--',label='B2 mismo resonador, sin heraldar'); a.axhline(r0['B2_Fh'],color='k',ls='-.',label='B2 heraldado ideal')
    a.set_xscale('log'); a.set_xticks([1.1,3,10]); a.xaxis.set_major_formatter(ScalarFormatter()); a.minorticks_off(); a.set_xlabel('g/2π (MHz)'); a.set_ylabel('fidelidad condicionada F_c'); a.set_title(f'FSR={fsr} MHz, Φ=0.42, T=100 µs'); a.set_ylim(0.55,0.85)
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(O+'fig3_diseno_g.png',dpi=160); plt.close()
# Fig 4: F_c(T) en tres escenarios, mejor diseno g=10, FSR=3.3, T1q=300
fig,ax=plt.subplots(1,3,figsize=(11,3.6),sharey=True)
for a,sc,t in zip(ax,('S1_Phi0.42_white','S2_Phi0.42_slow','S3_Phi0.90_white'),('Φ=0.42 (ruido blanco)','Φ=0.42 (cuasi-estático)','Φ=0.90 (hipotético)')):
    rr=sorted([r for r in res if r['sc']==sc and r['g']==10.0 and r['fsr']==3.3 and r['T1q']==300.],key=lambda r:r['T'])
    T=[r['T'] for r in rr]
    a.plot(T,[r['F_c'] for r in rr],'o-',color='C2',label='chequeo F_c'); a.plot(T,[r['p_pass'] for r in rr],'s--',color='C2',alpha=.6,label='p_pass')
    a.plot(T,[r['B1_Func'] for r in rr],':',color='gray',label='B1 sin heraldar'); a.plot(T,[r['B2_Func'] for r in rr],'--',color='k',label='B2 sin heraldar'); a.plot(T,[r['B2_Fh'] for r in rr],'-.',color='k',label='B2 heraldado ideal')
    a.set_xscale('log'); a.set_xticks([25,50,100,200,500]); a.xaxis.set_major_formatter(ScalarFormatter()); a.minorticks_off(); a.set_xlabel('espera T (µs)'); a.set_title(t)
ax[0].set_ylabel('fidelidad / probabilidad'); ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(O+'fig4_vs_T.png',dpi=160); plt.close()
print('figs ok')
