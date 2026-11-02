import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch, Circle
fig,ax=plt.subplots(1,2,figsize=(11,3.9),gridspec_kw={'width_ratios':[1.15,1]})
a=ax[0]; a.set_xlim(0,10); a.set_ylim(0,5); a.axis('off')
a.add_patch(FancyBboxPatch((0.4,1.4),7.2,2.2,boxstyle="round,pad=0.05",fc='#e8eef7',ec='C0',lw=1.5))
x=np.linspace(0.7,7.3,300)
a.plot(x,2.5+0.55*np.sin(2*np.pi*(x-0.7)/(6.6/4)*1),color='C3',lw=1.6); a.text(0.7,3.75,'modo 1 (ω₁)',color='C3',fontsize=9)
a.plot(x,2.5+0.55*np.sin(2*np.pi*(x-0.7)/(6.6/5)*1),color='C2',lw=1.6,alpha=.8); a.text(4.5,3.75,'modo 2 (ω₂ = ω₁ + FSR)',color='C2',fontsize=9)
a.text(4.0,1.55,'resonador piezoacústico multimodo (Si / LN)',ha='center',fontsize=9,color='C0')
a.add_patch(Circle((8.9,2.5),0.75,fc='#fdf0d5',ec='C1',lw=1.8)); a.text(8.9,2.5,'transmon\n(T₁q, χ)',ha='center',va='center',fontsize=8.5)
a.annotate('',xy=(8.15,2.5),xytext=(7.6,2.5),arrowprops=dict(arrowstyle='<->',color='k')); a.text(7.85,2.8,'g',ha='center')
a.text(5,0.55,'qubit: un fonón compartido |10⟩, |01⟩   |   Δ_tm ≫ FSR  ⇒  χ₁ ≈ χ₂',ha='center',fontsize=9)
a.set_title('Arquitectura')
b=ax[1]; b.set_xlim(0,10); b.set_ylim(0,5); b.axis('off')
steps=[('carga\n(iSWAP parciales)',0.2,2.0,'#d9ead3'),('espera T\n(ruido, pérdida)',2.3,2.4,'#fce5cd'),('chequeo JPNS\n(π gaussiano, σ)',4.8,2.3,'#cfe2f3'),('lectura\ntransmon',7.2,1.6,'#ead1dc'),('descarga',8.9,1.0,'#d9ead3')]
for t,x0,w,c in steps:
    b.add_patch(FancyBboxPatch((x0,2.0),w-0.1,1.4,boxstyle="round,pad=0.03",fc=c,ec='k')); b.text(x0+w/2-0.05,2.7,t,ha='center',va='center',fontsize=8)
b.text(5,1.25,'e (excitado) → pasa: n_total = 1   |   g → descarta (pérdida detectada)',ha='center',fontsize=8.5)
b.text(5,0.6,'F_c = fidelidad condicionada al pase,  p_pass = probabilidad de pase',ha='center',fontsize=8.5)
b.set_title('Protocolo')
plt.tight_layout(); plt.savefig('../figs/fig0_arquitectura.png',dpi=160)
print('ok')
