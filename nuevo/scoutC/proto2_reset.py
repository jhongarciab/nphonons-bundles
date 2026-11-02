# Prototipo I5: reset de qubit con baño acústico multimodo: tasa de reset = max Re de H_nh (1 excitación, T=0) ; EP en 1 modo
import numpy as np
def rate(g,N,delta,kappa=1.0,gq=0.01):
    d=(np.arange(N)-(N-1)/2)*delta
    H=np.zeros((N+1,N+1),complex); H[0,0]=-0.5j*gq
    for m in range(N):
        H[m+1,m+1]=d[m]-0.5j*kappa; H[0,m+1]=H[m+1,0]=g
    ev=np.linalg.eigvals(H); return -2*np.max(ev.imag)  # tasa de decaimiento de población (la más lenta)
print("1 modo: g_EP=|kappa-gq|/4 =",abs(1-0.01)/4)
gs=np.linspace(0.01,3,600)
for N,delta in [(1,0),(3,1.0),(3,3.0),(7,1.0),(7,3.0),(15,1.0)]:
    r=np.array([rate(g,N,delta) for g in gs]); i=np.argmax(r)
    print(f"N={N:2d} delta/kappa={delta:3.1f} | max reset rate={r[i]:.3f} kappa  en g_opt={gs[i]:.2f} kappa | g_opt*sqrt(N)... FGR(2pi g^2/delta*)={'-' if N==1 else round(2*np.pi*gs[i]**2/delta,3)}")
# mapa 2D para figura (1 modo vs N=7): guardar
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
G=np.linspace(0.02,3,120); D=np.linspace(0.2,6,120)
M=np.array([[rate(g,7,d) for g in G] for d in D])
plt.figure(figsize=(5,4)); plt.pcolormesh(G,D,M,shading='auto',cmap='viridis'); plt.colorbar(label='tasa de reset (kappa)')
plt.xlabel('g/kappa'); plt.ylabel('FSR/kappa'); plt.title('N=7 modos, qubit centrado'); plt.tight_layout(); plt.savefig('proto2_map.png',dpi=110)
