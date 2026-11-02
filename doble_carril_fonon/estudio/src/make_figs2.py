import json,numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.25}); O='../figs/'
d=np.load('../phi_hitchcock_fig4e.npy'); res=json.load(open('design2_eval_si.json')); hb=json.load(open('design2_eval_hbar.json'))
# fig1
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
ax[0].errorbar(d[:,0],d[:,1],yerr=[d[:,3],d[:,2]],fmt='o',color='C3',capsize=3,label='media ± desv. est. de todas las medidas (digitalizado, Fig. 4e)')
ax[0].axhline(0,color='k',lw=.5); ax[0].set_xlabel('desintonía Δω/2π (MHz)'); ax[0].set_ylabel('Φ'); ax[0].legend(fontsize=7); ax[0].set_title('Hitchcock et al., Φ(Δω)')
Phi=np.linspace(0,0.95,100)
ax[1].plot(Phi,Phi,label='ρ = Φ (blanco, f∝t)'); ax[1].plot(Phi,1-(1-Phi)**2,label='ρ = 1−(1−Φ)² (cuasi-estático, f∝t²)')
ax[1].axvspan(0.42-0.13,0.42+0.13,color='C3',alpha=.15,label='Φ medido (adyacentes)'); ax[1].set_xlabel('Φ'); ax[1].set_ylabel('ρ (Pearson)'); ax[1].legend(fontsize=8); ax[1].set_title('Φ → ρ según el régimen')
plt.tight_layout(); plt.savefig(O+'fig1_phi_medido.png',dpi=160); plt.close()
# fig3: vs g
fig,ax=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
for a,T1q in zip(ax,(100.,300.)):
    sel=[r for r in res if r['T']==100. and r['Tphi']==107. and r['Phi']==0.42 and r['n']==1 and r['fsr']==3.3 and r['T1q']==T1q]; sel.sort(key=lambda r:r['g'])
    a.plot([r['g'] for r in sel],[r['F_c'] for r in sel],'o-',color='C2',label='chequeo F_c (condicionada)'); a.plot([r['g'] for r in sel],[r['F_ef'] for r in sel],'s-',color='C1',label='chequeo F_ef (fallo→adivinar)')
    r0=sel[0]; a.axhline(r0['B0'],color='C0',ls='--',label='B0: un carril'); a.axhline(r0['B2'],color='k',ls=':',label='B2: doble carril sin chequeo'); a.axhline(r0['F_ef_ideal'],color='gray',ls='-.',label='F_ef con heraldado ideal')
    a.set_xscale('log'); a.set_xticks([1.1,3,10]); a.xaxis.set_major_formatter(ScalarFormatter()); a.minorticks_off(); a.set_xlabel('g/2π (MHz)'); a.set_title(f'T=100 µs, Φ=0.42 (blanco), T1q={T1q:.0f} µs')
ax[0].set_ylabel('fidelidad'); ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(O+'fig3_diseno_g.png',dpi=160); plt.close()
# fig4: vs T
fig,ax=plt.subplots(1,3,figsize=(11.5,3.7),sharey=True)
for a,(Phi_,n,t) in zip(ax,((0.42,1,'Φ=0.42 (ruido blanco, ρ=0.42)'),(0.42,2,'Φ=0.42 (cuasi-estático, ρ=0.66)'),(0.90,1,'Φ=0.90 (hipotético)'))):
    rr=sorted([r for r in res if r['g']==3.0 and r['fsr']==3.3 and r['T1q']==300. and r['Tphi']==107. and r['Phi']==Phi_ and r['n']==n],key=lambda r:r['T']); T=[r['T'] for r in rr]
    a.plot(T,[r['F_c'] for r in rr],'o-',color='C2',label='chequeo F_c'); a.plot(T,[r['F_ef'] for r in rr],'s-',color='C1',label='chequeo F_ef')
    a.plot(T,[r['B0'] for r in rr],'--',color='C0',label='B0 un carril'); a.plot(T,[r['B2'] for r in rr],':',color='k',label='B2 sin chequeo'); a.plot(T,[r['p_pass'] for r in rr],'^--',color='C2',alpha=.5,label='p_pass')
    a.set_xscale('log'); a.set_xticks([25,50,100,200,500]); a.xaxis.set_major_formatter(ScalarFormatter()); a.minorticks_off(); a.set_xlabel('espera T (µs)'); a.set_title(t,fontsize=9)
ax[0].set_ylabel('fidelidad / probabilidad'); ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(O+'fig4_vs_T.png',dpi=160); plt.close()
# fig5: umbral analitico
x=np.linspace(0,1,200)
fig,ax=plt.subplots(figsize=(5.2,3.8)); ax.plot(x,0.5*(1+x),'k',label='ρ* = ½(1 + Tφ/T₁)  (ruido blanco, chequeo ideal)')
ax.fill_between(x,0.5*(1+x),1,color='C2',alpha=.15,label='ventaja sobre un carril (F_ef)')
ax.axvline(107/534,color='C3',ls=':',lw=1); ax.text(107/534+0.01,0.52,'Tφ/T₁ = 0.20\n(Si, un modo)',fontsize=8,color='C3')
ax.axvline(10/534,color='C1',ls=':',lw=1); ax.text(10/534+0.01,0.95,'Tφ≈10 µs\n(extrap.)',fontsize=8,color='C1')
ax.axhspan(0.42-0.13,0.42+0.13,xmin=0,xmax=1,color='C3',alpha=.12); ax.text(0.55,0.36,'ρ medido (blanco): 0.42 ± 0.13',fontsize=8,color='C3')
ax.set_xlabel('Tφ/T₁'); ax.set_ylabel('ρ requerido'); ax.set_ylim(0.3,1.0); ax.legend(fontsize=7,loc='lower right'); plt.tight_layout(); plt.savefig(O+'fig5_umbral_rho.png',dpi=160); plt.close()
# tablas
L=["## Tabla R-A. Silicio (T1=534 µs, FSR=3.3 MHz, g=3 MHz, T1q=300 µs; diseño optimizado con objetivo F_ef, g/Δ≤0.2). Convención Hitchcock corregida (VC).\n",
"| Tφ (µs) | Escenario | T (µs) | F_c | p_pass | F_ef | B0 un carril | B1 | B2 | F_h ideal | F_ef ideal |\n|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for Tphi in (107.,10.):
    for (Phi_,n,name) in ((0.42,1,'Φ=0.42 blanco'),(0.42,2,'Φ=0.42 cuasi-est.'),(0.90,1,'Φ=0.90 blanco')):
        for T in (25.,50.,100.,200.,500.):
            m=[r for r in res if r['g']==3.0 and r['fsr']==3.3 and r['T1q']==300. and r['T']==T and r['Tphi']==Tphi and r['Phi']==Phi_ and r['n']==n]
            if not m: continue
            r=m[0]; L.append(f"| {Tphi:.0f} | {name} | {T:.0f} | {r['F_c']:.3f} | {r['p_pass']:.3f} | {r['F_ef']:.3f} | {r['B0']:.3f} | {r['B1']:.3f} | {r['B2']:.3f} | {r['Dh']:.3f} | {r['F_ef_ideal']:.3f} |")
L+=["\n## Tabla R-B. Sensibilidad a g, FSR y T1q (T=100 µs, Tφ=107, Φ=0.42 blanco)\n","| g (MHz) | FSR | T1q | Δ_tm* | σ* (µs) | F_c | p_pass | F_ef | B0 |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in sorted([r for r in res if r['T']==100. and r['Tphi']==107. and r['Phi']==0.42 and r['n']==1],key=lambda r:(r['g'],r['fsr'],r['T1q'])):
    L.append(f"| {r['g']} | {r['fsr']} | {r['T1q']:.0f} | {r['D']} | {r['sg']} | {r['F_c']:.3f} | {r['p_pass']:.3f} | {r['F_ef']:.3f} | {r['B0']:.3f} |")
L+=["\n## Tabla R-C. HBAR (FSR=12.7 MHz, g=0.273 MHz, T1 modo=82 µs, sin desfase puro)\n","| T1q | T | F_c | p_pass | F_ef | B0 un carril | sin chequeo | F_h ideal | F_ef ideal |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in hb: L.append(f"| {r['T1q']:.0f} | {r['T']:.0f} | {r['F_c']:.3f} | {r['p_pass']:.3f} | {r['F_ef']:.3f} | {r['B0']:.3f} | {r['B2']:.3f} | {r['Dh']:.3f} | {r['F_ef_ideal']:.3f} |")
open('../results_tables.md','w').write('\n'.join(L)+'\n'); print('ok')
