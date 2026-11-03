"""Simulacion dependiente del tiempo: transmon multinivel x modo, preparacion (2 pulsos) + emision con conteo exacto de clics.
Unidades: tiempo en us, frecuencias en rad/us (2pi*MHz). Marco rotante a la frecuencia del modo (RWA en g; ver check Bloch-Siegert).
Conteo exacto: funcion generadora con campo de conteo lambda (FFT en lambda) -> P(m clics) sin Monte Carlo.
Fase de pulsos: cadena de bloques (m clics) integrada con RK4 (dt=0.25 ns). Fase estatica: expm del Liouvilliano inclinado."""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from dataclasses import dataclass, replace
TP = 2*np.pi

@dataclass
class Cfg:
    g: float = 15.2*TP; alpha: float = -138*TP; Dq: float = None   # Dq = w_q - w (None -> resonancia numerica)
    nq: int = 4; N: int = 6; cap: int = 4
    km: float = 8.3*TP; kint: float = 0.0; nth: float = 0.0; ng: float = 0.0  # nth: ocupacion termica inicial del modo y del bano intrinseco; ng: ocupacion de la guia
    T1: float = 1.14; Tphi: float = np.inf; pe: float = 0.0
    r_fe: float = 2.0   # tasa f->e en unidades de gamma_q (2: escalera; 1: sin escalera)
    g20: float = 0.0    # tasa directa f->g en unidades de gamma_q
    order: int = 2      # 2: |f,0>->|g,2> ; 3: |3,0>->|g,3>
    off: float = 0.0    # desintonia extra de w_q (rad/us) respecto a la resonancia

def sub_index(nq, N, cap):
    return [q*N+n for q in range(nq) for n in range(N) if q+n <= cap]

def raw_ops(nq, N):
    S = np.diag(np.sqrt(np.arange(1, nq)), 1).astype(complex); A = np.diag(np.sqrt(np.arange(1, N)), 1).astype(complex)
    return np.kron(np.eye(nq), A), np.kron(S, np.eye(N)), np.kron(np.diag(np.arange(nq)).astype(complex), np.eye(N)), np.kron(np.eye(nq), np.diag(np.arange(N)).astype(complex))

def Hparts(cfg, Dq):
    a, s, nqo, _ = raw_ops(cfg.nq, cfg.N)
    Ek = np.array([k*Dq+cfg.alpha*k*(k-1)/2 for k in range(cfg.nq)])
    H0 = np.kron(np.diag(Ek).astype(complex), np.eye(cfg.N))
    Hg = cfg.g*(s.conj().T@a + s@a.conj().T)
    return H0, Hg

def find_res(cfg):
    """Dq (w_q - w) que minimiza el gap entre los dos estados vestidos con peso en |K,0> y |g,K>."""
    K = cfg.order; N = max(cfg.N, K+1)
    c = replace(cfg, N=N, cap=99)
    def gap(Dq):
        H0, Hg = Hparts(c, Dq); E, V = np.linalg.eigh(H0+Hg)
        i1, i2 = K*N+0, 0*N+K
        w = np.abs(V[i1])**2+np.abs(V[i2])**2; j = np.argsort(w)[-2:]
        return abs(E[j[0]]-E[j[1]])
    x0 = -cfg.alpha*(K-1)/2   # K=2: -alpha/2 ; K=3: -alpha
    xs = np.linspace(x0-0.25*abs(cfg.alpha), x0+0.25*abs(cfg.alpha), 801); gs = [gap(x) for x in xs]; i = int(np.argmin(gs))
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(gap, bounds=(xs[max(i-1, 0)], xs[min(i+1, 800)]), method='bounded', options={'xatol': 1e-10})
    return r.x, r.fun

class Sys:
    def __init__(self, cfg):
        self.cfg = c = cfg
        self.idx = sub_index(c.nq, c.N, c.cap); self.D = D = len(self.idx)
        self.Dq0, self.gap = find_res(c) if c.Dq is None else (c.Dq, np.nan)
        self.Dq = (c.Dq if c.Dq is not None else self.Dq0) + c.off
        a, s, nqo, nm = raw_ops(c.nq, c.N); ix = np.ix_(self.idx, self.idx)
        H0, Hg = Hparts(c, self.Dq)
        self.a, self.s, self.nqo, self.nm, self.H0, self.Hg = [m[ix] for m in (a, s, nqo, nm, H0, Hg)]
        self.ad = self.a.conj().T; self.sd = self.s.conj().T
        gq = 1.0/c.T1 if np.isfinite(c.T1) else 0.0
        # operadores de colapso
        self.counted = [np.sqrt(c.km*(1+c.ng))*self.a]
        un = []
        if c.ng > 0: un.append(np.sqrt(c.km*c.ng)*self.ad)
        if c.kint > 0:
            un.append(np.sqrt(c.kint*(1+c.nth))*self.a)
            if c.nth > 0: un.append(np.sqrt(c.kint*c.nth)*self.ad)
        if gq > 0:
            Q = np.zeros((c.nq, c.nq))
            for k in range(1, c.nq): Q[k-1, k] = np.sqrt(gq*(k if k != 2 else c.r_fe))
            un.append(np.kron(Q, np.eye(c.N))[ix])
            if c.g20 > 0:
                Q2 = np.zeros((c.nq, c.nq)); Q2[0, 2] = np.sqrt(gq*c.g20); un.append(np.kron(Q2, np.eye(c.N))[ix])
        if np.isfinite(c.Tphi): un.append(np.sqrt(2.0/c.Tphi)*self.nqo)   # coherencia g-e decae a 1/Tphi
        self.unc = un
        I = np.eye(D)
        def pre(A): return np.kron(A, I)
        def post(B): return np.kron(I, B.T)
        Ld = np.zeros((D*D, D*D), complex)
        for C in un:
            Cd = C.conj().T; Ld += np.kron(C, C.conj())-0.5*(pre(Cd@C)+post(Cd@C))
        for C in self.counted:
            Cd = C.conj().T; Ld += -0.5*(pre(Cd@C)+post(Cd@C))
        self.J = sum(np.kron(C, C.conj()) for C in self.counted)
        self.Ld = Ld                                   # disipacion sin salto contado
        self.LH0 = -1j*(pre(self.H0)-post(self.H0))
        self.LHg = -1j*(pre(self.Hg)-post(self.Hg))
        self.L0 = Ld+self.LH0+self.LHg                 # estatico (g encendido) sin clic
        self.kd = c.km*(1+c.ng)

    def rho_init(self, qpops=None, nth=None):
        c = self.cfg; qp = np.zeros(c.nq); qp[0] = 1-c.pe; qp[1] = c.pe
        if qpops is not None: qp = np.asarray(qpops, float)
        nth = c.nth if nth is None else nth
        pm = (nth/(1+nth))**np.arange(c.N) if nth > 0 else np.eye(c.N)[0]
        pm = pm/pm.sum(); pj = np.kron(qp, pm)
        return np.diag(pj[self.idx]).astype(complex)

    def ket_index(self, k, n): return self.idx.index(k*self.cfg.N+n)

# ---------------- pulsos -----------------
@dataclass
class Pulse:
    tp: float = 0.024          # duracion de cada pi (us)
    x: tuple = ()  # escala de amplitud 1,2 ; offsets de frecuencia 1,2 (rad/us)
    drag: float = 0.0
    gate: bool = True          # g encendido durante pulsos (caso A); False: g apagado hasta el final (caso B, qubit estacionado)
    nsig: float = 5.0          # sigma = tp/nsig

def env_area(p):
    u = np.linspace(0, 1, 4001); e0 = np.exp(-0.5*(0.5*p.nsig)**2)
    f = (np.exp(-0.5*((u-0.5)*p.nsig)**2)-e0)/(1-e0); return np.trapezoid(f, u)*p.tp, e0

def pulse_coefs(sy, p):
    """Devuelve (gfun(t), [coef_k(t)]) : H(t) = H0 + gfun*Hg + sum_k [coef_k s + conj(coef_k) s^dag]."""
    c = sy.cfg; n = c.order; area, e0 = env_area(p); sg = p.tp/p.nsig
    x = p.x if len(p.x) == 2*n else (1.0,)*n+(0.0,)*n
    A = [x[k]*np.pi/(np.sqrt(k+1)*area) for k in range(n)]
    d = [sy.Dq+c.alpha*k+x[n+k] for k in range(n)]      # transicion k->k+1: Dq + alpha*k
    def mk(k):
        t0 = k*p.tp
        def cf(t):
            u = t-t0
            if u < 0 or u > p.tp: return 0.0
            z = (u-p.tp/2)/sg; f = (np.exp(-0.5*z*z)-e0)/(1-e0); df = -(z/sg)*np.exp(-0.5*z*z)/(1-e0)
            return 0.5*(A[k]*f+1j*p.drag*A[k]*df/abs(c.alpha))*np.exp(1j*d[k]*t)
        return cf
    gf = (lambda t: 1.0) if p.gate else (lambda t: 1.0 if t >= n*p.tp else 0.0)
    return gf, [mk(k) for k in range(n)]

def pulse_funcs(sy, p):
    gf, cfs = pulse_coefs(sy, p)
    def Hdyn(t):
        H = gf(t)*sy.Hg
        for cf in cfs:
            ck = cf(t)
            if ck != 0.0: H = H+ck*sy.s+np.conj(ck)*sy.sd
        return H
    return Hdyn

def evolve_ket(sy, p, dt=2.5e-4):
    Hd = pulse_funcs(sy, p); psi = np.zeros(sy.D, complex); psi[sy.ket_index(0, 0)] = 1
    n = int(round(sy.cfg.order*p.tp/dt)); t = 0.0
    f = lambda t, y: -1j*((sy.H0+Hd(t))@y)
    for _ in range(n):
        k1 = f(t, psi); k2 = f(t+dt/2, psi+dt/2*k1); k3 = f(t+dt/2, psi+dt/2*k2); k4 = f(t+dt, psi+dt*k3)
        psi = psi+dt/6*(k1+2*k2+2*k3+k4); t += dt
    return psi

def pops_ket(sy, psi):
    c = sy.cfg; pk = np.zeros(c.nq)
    for j, i in enumerate(sy.idx): pk[i//c.N] += abs(psi[j])**2
    return pk

def calibrate(sy, p0, nit=250):
    """Ajusta escalas de amplitud y offsets de frecuencia de los n pulsos para maximizar la poblacion en el estado objetivo
    ({|K,0>,|g,K>} con g encendido; |K,0> con g apagado)."""
    K = sy.cfg.order; n = K; i_t = sy.ket_index(K, 0); i_g = sy.ket_index(0, K)
    def cost(x):
        psi = evolve_ket(sy, replace(p0, x=tuple(x))); return 1-(abs(psi[i_t])**2+(abs(psi[i_g])**2 if p0.gate else 0.0))
    x0 = np.array(p0.x if len(p0.x) == 2*n else (1.0,)*n+(0.0,)*n); sc = np.array([0.05]*n+[2*TP]*n)
    r = minimize(lambda z: cost(x0+z*sc), np.zeros(2*n), method='Nelder-Mead', options={'maxiter': nit, 'xatol': 1e-3, 'fatol': 1e-6})
    return replace(p0, x=tuple(x0+r.x*sc)), r.fun

# ---------------- fase de pulsos con cadena de bloques ----------------
def pulse_phase(sy, p, R0, M=5, dt=2.5e-4):
    Hd = pulse_funcs(sy, p); D = sy.D
    R = np.zeros((M, D, D), complex); R[0] = R0
    LbT = (sy.Ld+sy.LH0).T.copy(); n = int(round(sy.cfg.order*p.tp/dt)); t = 0.0; ad = sy.ad; a = sy.a; kd = sy.kd
    def f(t, R):
        H = Hd(t); dR = (R.reshape(M, -1)@LbT).reshape(M, D, D)
        dR += -1j*(H@R-R@H); dR[1:] += kd*(a@R[:-1]@ad); return dR
    for _ in range(n):
        k1 = f(t, R); k2 = f(t+dt/2, R+dt/2*k1); k3 = f(t+dt/2, R+dt/2*k2); k4 = f(t+dt, R+dt*k3); R = R+dt/6*(k1+2*k2+2*k3+k4); t += dt
    return R

def pops_rho(sy, rho):
    c = sy.cfg; d = np.real(np.diag(rho)); pk = np.zeros(c.nq); pn = np.zeros(c.N)
    for j, i in enumerate(sy.idx): pk[i//c.N] += d[j]; pn[i % c.N] += d[j]
    return pk, pn

# ---------------- fase estatica: conteo exacto ----------------
def static_counts(sy, Rm, Tmax=3.0, dT=0.1, Lam=12, L=None, J=None):
    """Rm: (M,D,D) bloques con m clics ya contados (o (1,D,D)). Devuelve T (nT+1,), P (nT+1, Lam) prob de m clics (m=0..Lam-1), rho(t) total."""
    D = sy.D; L = sy.L0 if L is None else L; J = sy.J if J is None else J
    lam = 2*np.pi*np.arange(Lam)/Lam; nT = int(round(Tmax/dT)); G = np.zeros((nT+1, Lam), complex)
    tr = np.arange(D)*(D+1); M = Rm.shape[0]; rho_t = None
    for j, l in enumerate(lam):
        E = expm((L+np.exp(1j*l)*J)*dT)
        v = sum(np.exp(1j*l*m)*Rm[m] for m in range(M)).reshape(-1)
        for k in range(nT+1):
            G[k, j] = v[tr].sum()
            if j == 0:
                if rho_t is None: rho_t = np.zeros((nT+1, D, D), complex)
                rho_t[k] = v.reshape(D, D)
            v = E@v
    ms = np.arange(Lam); P = np.real(np.einsum('kj,jm->km', G, np.exp(-1j*np.outer(lam, ms)))/Lam)
    return np.arange(nT+1)*dT, P, rho_t

def summarize(P):
    return np.stack([P[..., 0], P[..., 1], P[..., 2], 1-P[..., 0]-P[..., 1]-P[..., 2]], axis=-1)

def run_cycle(cfg, p=None, tw_idx=None, Tmax=3.0, dT=0.1, sy=None, gate_clicks=True):
    """Ciclo: estado inicial termico -> 2 pulsos (si p) -> emision. p=None: parte de |f,0> ideal."""
    sy = Sys(cfg) if sy is None else sy
    if p is None:
        R0 = np.zeros((sy.D, sy.D), complex); i = sy.ket_index(cfg.order, 0); R0[i, i] = 1; Rm = R0[None]; prep = None
    else:
        Rm = pulse_phase(sy, p, sy.rho_init()); rho = Rm.sum(0); cp = 1-np.real(np.trace(Rm[0]))
        prep = dict(pk=pops_rho(sy, rho)[0], pf0=np.real(rho[sy.ket_index(2, 0), sy.ket_index(2, 0)]), clicks_pulse=cp)
        if gate_clicks: Rm = rho[None]   # clics durante los pulsos descartados (la ventana de emision empieza tras los pulsos)
    T, P, rho_t = static_counts(sy, Rm, Tmax, dT)
    return dict(T=T, P=summarize(P), Praw=P, prep=prep, sy=sy, rho_t=rho_t)
