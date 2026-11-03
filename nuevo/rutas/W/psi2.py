"""Funcion de onda conjunta del par emitido a la guia, psi(t1,t2), y su numero de Schmidt temporal.
Modelo efectivo: A=|f,0> (decae a rate kq), B=|g,2> (salta a la guia a 2 km), acoplo A-B = V = sqrt2*G (G=1).
H_nh: A: -i kq/2 ; B: -i km ; A-B: V. beta(t) = amplitud en B (alfa(0)=1). Salto en t1 (solo desde B): -> |g,1> con amplitud sqrt(2 km) beta(t1); luego sale el 2o fonon con km.
psi(t1<t2) = sqrt(2) km beta(t1) exp(-km (t2-t1)/2). Estado de dos bosones simetrico: Psi_s(t1,t2) = psi(orden). Norma = eficiencia eta."""
import numpy as np, scipy.linalg as sl
G = 1.0; V = np.sqrt(2)*G
def beta_of_t(t, kq, km):
    M = np.array([[-0.5j*kq, V], [V, -1j*km]]); out = np.empty(len(t), complex)
    w, U = np.linalg.eig(-1j*M); Ui = np.linalg.inv(U); c = Ui@np.array([1, 0])
    for i, ti in enumerate(t): out[i] = (U@(c*np.exp(w*ti)))[1]
    return out
def schmidt(kq, km, N=1100, q=0.999):
    # horizonte: cuando el marginal de t1 y el decaimiento km ya cubren q
    tt = np.linspace(0, 400/max(min(km, 1), 0.05), 40000); b = beta_of_t(tt, kq, km); pdf = 2*km*np.abs(b)**2
    cdf = np.cumsum(pdf)*(tt[1]-tt[0]); eta = cdf[-1]
    T1 = tt[np.searchsorted(cdf, q*eta)] + 20/km
    t = np.linspace(0, T1, N); dt = t[1]-t[0]; bb = beta_of_t(t, kq, km)
    t1, t2 = np.meshgrid(t, t, indexing='ij'); lo = np.minimum(t1, t2); hi = np.maximum(t1, t2)
    # beta en lo: indice
    il = np.minimum(t1, t2)/dt; il = np.rint(il).astype(int)
    Psi = np.sqrt(2)*km*bb[il]*np.exp(-km*(hi-lo)/2)
    norm = (np.abs(Psi)**2).sum()*dt*dt/2
    s = np.linalg.svd(Psi*dt, compute_uv=False); p = s**2/np.sum(s**2)
    return dict(eta=norm, K=1/np.sum(p**2), p1=p[0], S=-np.sum(p[p > 1e-12]*np.log2(p[p > 1e-12])), T=T1)
if __name__ == '__main__':
    def eta_f(kq, km):
        y1, y2, V2 = kq/2, km, 2*G**2; return 1/(1+y1/y2+y1*(y1+y2)/V2)
    for kq in (0.05, 0.5):
        print('kappa_q/G = %.2f   (km*=sqrt2 G=1.414)'%kq)
        print('  km/G   eta(formula) eta(integral)  K(Schmidt)  p_modo_dominante  S(bits)')
        for km in (0.1, 0.3, 0.7, 1.414, 3.0, 8.0, 20.0):
            r = schmidt(kq, km); print('  %-6.2f %-12.4f %-14.4f %-11.3f %-17.3f %-6.3f'%(km, eta_f(kq, km), r['eta'], r['K'], r['p1'], r['S']), flush=True)
