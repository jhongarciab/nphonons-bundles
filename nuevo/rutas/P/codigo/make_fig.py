import numpy as np, pickle, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D='../datos/'; b=np.load(D+'barrido_completo.npz'); m2=np.load(D+'mapa_estatico_N2.npz'); ex=pickle.load(open(D+'extras.pkl','rb')); pr=pickle.load(open(D+'presupuesto.pkl','rb')); ts=pickle.load(open(D+'serie_temporal.pkl','rb')); n3=np.load(D+'mapa_estatico_N3.npz')
tl=ts['tl']; tp=ts['tp']; sel=None
for ct,cw,ev in ts['traj']:
    w=(ct>2*tp)&(ct<=2*tp+0.5)&(cw==0)
    if w.sum()==2 and ev[2][-1]<0.2: sel=(ct,cw,ev); break
# promedio cuasi-estatico gaussiano sobre off (MHz) para PUB B / IMP B
OFF=ex['OFF']; qs={}
for sn in ('pub','imp'):
    y=ex['off_'+sn][:,2]
    for s in (0.5,1,2,3):
        w=np.exp(-0.5*((OFF+2.0)/s)**2); qs[f'{sn}_s{s}']=float((w*y).sum()/w.sum()) if False else float(np.trapezoid(y*np.exp(-0.5*(OFF/s)**2),OFF)/np.trapezoid(np.exp(-0.5*(OFF/s)**2),OFF))
print('cuasi-estatico <P2>:',qs)
T1=[1.14,5,10,50]
np.savez(D+'figuras_P.npz',KM=b['KM'],OFF=b['OFF'],P_km_pub_B=b['pub_B_km'],P_km_pub_A=b['pub_A_km'],P_km_imp_B=b['imp_B_km'],P_km_imp_A=b['imp_A_km'],
 P_off_pub_B=b['pub_B_off'],P_off_imp_B=b['imp_B_off'],P_off_pub_A=b['pub_A_off'],P_off_imp_A=b['imp_A_off'],
 mapa_KM=m2['KM'],mapa_OFF=m2['OFF'],mapa_T1_1p14_P_Tw0p5=m2['T1_1.14_P_Tw0.5'],mapa_T1_10_P_Tw0p5=m2['T1_10_P_Tw0.5'],mapa_T1_1p14_P_Tw3=m2['T1_1.14_P_Tw3'],mapa_T1_10_P_Tw3=m2['T1_10_P_Tw3'],
 T1=np.array(T1),P2_vs_T1_full=np.array([r[3] for r in pr['T1_full']]),P2_vs_T1_IMPrest=np.array([r[3] for r in pr['T1_IMPrest']]),P2_vs_T1_only=np.array([r[3] for r in pr['T1_only']]),
 nth=np.array([r[0] for r in pr['nth_cold']]),P2_vs_nth_cold=np.array([r[3] for r in pr['nth_cold']]),P2_vs_nth_hot=np.array([r[3] for r in pr['nth_hot']]),
 ts_t=tl,ts_ens=np.array(ts['ens']),ts_clicks_cum=ts['clicks_cum'],ts_traj_click_t=sel[0],ts_traj_which=sel[1],ts_traj_expect=np.array(sel[2]),ts_tp=tp,
 N3_KM=n3['KM3'],N3_OFF=n3['OFF3'],N3_T1_1p14=n3['T1_1.14'],N3_T1_10=n3['T1_10'],N3_T1_inf=n3['T1_inf'])
fig,ax=plt.subplots(2,2,figsize=(10,7)); col={'pub_B':'#1f77b4','pub_A':'#ff7f0e','imp_B':'#2ca02c','imp_A':'#d62728'}
for k,c in col.items(): ax[0,0].semilogx(b['KM'],b[k+'_km'][:,5,2],'o-',c=c,label=k.replace('_',' ')); 
ax[0,0].set_xlabel('kappa_m/2pi (MHz)'); ax[0,0].set_ylabel('P(2) , T_w=0.5 us'); ax[0,0].legend(fontsize=8); ax[0,0].set_title('P(2) vs kappa_m (pulsos reales)')
for k,c in col.items(): ax[0,1].plot(b['OFF'],b[k+'_off'][:,5,2],'o-',c=c,label=k)
ax[0,1].set_xlabel('desintonia extra de w_q (MHz)'); ax[0,1].set_title('P(2) vs desintonia (en kappa* de cada caso)')
ax[1,0].semilogx(T1,[r[3] for r in pr['T1_full']],'o-',label='resto PUB'); ax[1,0].semilogx(T1,[r[3] for r in pr['T1_IMPrest']],'s-',label='resto IMP'); ax[1,0].semilogx(T1,[r[3] for r in pr['T1_only']],'^-',label='solo T1 (resto ideal)'); ax[1,0].set_xlabel('T1 (us)'); ax[1,0].set_ylabel('P(2), T_w=0.5 us'); ax[1,0].legend(fontsize=8); ax[1,0].set_title('P(2) vs T1 (caso B, km=10.8 MHz)')
e=np.array(ts['ens']); 
for k in range(4): ax[1,1].plot(tl*1e3,e[k],label=f'P(k={k})')
ax[1,1].plot(tl*1e3,e[4],'k--',label='<n_fonon>'); ax[1,1].plot(sel[0]*1e3,np.zeros_like(sel[0])+0.02,'v',c='r',label='clics (trayectoria)'); ax[1,1].set_xlabel('t (ns)'); ax[1,1].legend(fontsize=7); ax[1,1].set_title('Serie temporal (PUB, B): ensemble'); 
plt.tight_layout(); plt.savefig(D+'figuras_P.png',dpi=110)
print('sel clicks',sel[0],sel[1]); print(e.shape)
