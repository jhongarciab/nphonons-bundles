"""bundles_core: teoria exacta y cerrada de bundles de N fonones (emisor de 2 niveles + modo bosonico).

CONVENCIONES (unidades hbar = omega_b = 1; todas las energias/tasas/tiempos en unidades de omega_b)
-------------------------------------------------------------------------------------------------
Hamiltoniano en el marco del laser (rotante):
    H = dL s+s + b+b + lam s+s (b+b+) + Om (s+s+)          [marco de LABORATORIO]
    dL = omega_sigma - omega_L  (desintonia del laser),  lam = acoplo longitudinal,  Om = Rabi.
Base: tensor(QD, cavidad) con QD = (g,x); indice = q*N + m.  s = |g><x|.

Transformacion de polaron U = exp[-lam s+s (b+ - b)]:
    H_p = delta s+s + b+b + Om (s+ D + s D+),   delta = dL - lam^2,   D = exp[-lam (b+ - b)]
con elementos exactos d_{mk} = <m|D|k> = sqrt(k!/m!) (-lam)^{m-k} e^{-lam^2/2} L_k^{(m-k)}(lam^2), m>=k.
Resonancia de n fonones (|g,0> <-> |x~,n>):  dL = lam^2 - n + eps_n   (ver eps_n).

DISIPACION (todas con la misma tasa fisica):
    kap  : perdida del modo,   gam : decaimiento radiativo del emisor,   gph : desfase puro (s+s).
Maestras (marco de polaron):
  'dressed' (vestida, FISICA para kap << frecuencias del sistema): saltos  sqrt(kap) b,  sqrt(gam) s D+,  sqrt(gph) s+s
  'local'   (salto b del laboratorio transformado):                saltos  sqrt(kap)(b + lam s+s), sqrt(gam) s D+, sqrt(gph) s+s
  (la 'local' en polaron == maestra local en el marco de laboratorio; ver tests).
Estadistica de conteo (RENOVACION EXACTA, un solo LU, serie en potencias de la funcion generatriz):
  * ciclos:    fonones (saltos del modo) emitidos entre dos saltos radiativos consecutivos.
  * episodios: fonones emitidos entre dos 'retornos de la cavidad al vacio' (salto 1->0 del modo, clasificado
               por el numero de fonones PRE-salto: P1 (y P0 en la maestra local) cierran, P>=2 cuentan).
               Los terminos cruzados (coherencias P1<->P>=2) se dejan en la evolucion sin conteo (pseudo-desenrollado;
               la dinamica total es la fisica). La validacion operativa es el Monte Carlo (bundles_mc.py).
Pureza: P_n = P(k=n) de la distribucion de episodios (fraccion de episodios con exactamente n fonones).
"""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # 1 hilo BLAS: 50x mas rapido para matrices ~300x300 y respeta el limite de CPU
import numpy as np
from math import exp, sqrt, factorial
from scipy.special import eval_genlaguerre, gammaln
from scipy.linalg import lu_factor, lu_solve, solve
from scipy.optimize import brentq

# ----------------------------------------------------------------------------------------------
# 1. Elementos de matriz del desplazamiento y resonancia analitica
# ----------------------------------------------------------------------------------------------
def dmat(lam, m, k):
    """d_{mk} = <m|D|k>, D = exp[-lam(b+ - b)] (desplazamiento D(alpha=-lam)), exacto via Laguerre. Real; d_{km}=(-1)^{k-m} d_{mk}... ver codigo."""
    if m < k:
        return (-1) ** (k - m) * dmat(lam, k, m)
    a = -lam
    lg = 0.5 * (gammaln(k + 1) - gammaln(m + 1)) - lam ** 2 / 2
    return exp(lg) * a ** (m - k) * eval_genlaguerre(k, m - k, lam ** 2)


def D_matrix(lam, N):
    """Matriz N x N de D truncada (elementos exactos, no el expm del espacio truncado)."""
    return np.array([[dmat(lam, m, k) for k in range(N)] for m in range(N)])


def eps_n(lam, Om, n, M=40):
    """Correccion de 2o orden (polaron + Stark dinamico) a la resonancia, en delta = dL - lam^2 = -n + eps_n:
       eps_n = -Om^2 sum_{m!=n} d_{m0}^2/(m-n) + Om^2 sum_{k>=1} d_{nk}^2/k   ~  2 Om^2/n.
       Limite lam->0: eps_n = 2 Om^2/n = expansion de n - sqrt(n^2 - 4 Om^2) (Bin-III)."""
    Sg = -Om ** 2 * sum(dmat(lam, m, 0) ** 2 / (m - n) for m in range(M) if m != n)
    Sx = -Om ** 2 * sum(dmat(lam, n, k) ** 2 / k for k in range(1, M))
    return Sg - Sx


def delta_res(n, lam, Om):
    """Posicion de la resonancia de n fonones en la desintonia del laser: dL = lam^2 - n + eps_n."""
    return lam ** 2 - n + eps_n(lam, Om, n)


def omega_eff(n, lam, Om):
    """Rabi efectivo n-fonon (Raman): Om_n = Om |d_{n0}| = Om e^{-lam^2/2} lam^n / sqrt(n!)  (Bin et al.)."""
    return Om * abs(dmat(lam, n, 0))


# ----------------------------------------------------------------------------------------------
# 2. Formulas cerradas (modelo de tasas)
# ----------------------------------------------------------------------------------------------
def gamma_c(n, j, kap, gam, gph):
    """Tasa de decoherencia de la coherencia Raman |g,j> <-> |x,j+n|: [(2j+n) kap + gam + gph]/2."""
    return ((2 * j + n) * kap + gam + gph) / 2


def Gamma_j(lam, Om, kap, gam, gph, n, j):
    """Tasa de excitacion Raman incoherente g,j -> x,j+n:  Gamma_j = 2 Om^2 d_{n+j,j}^2 / gamma_c(j)."""
    return 2 * Om ** 2 * dmat(lam, n + j, j) ** 2 / gamma_c(n, j, kap, gam, gph)


def flux_closed(lam, Om, kap, gam, gph, n):
    """Flujo cerrado de fonones J = (n + lam^2) / (1/Gamma_0 + 1/gam); Gamma_0 = 4 Om^2 d_{n0}^2/(n kap + gam + gph)."""
    G0 = Gamma_j(lam, Om, kap, gam, gph, n, 0)
    return (n + lam ** 2) / (1 / G0 + 1 / gam)


def P_floor(lam, kind='exp'):
    """Piso de Franck-Condon. 'exp': e^{-lam^2} (prob. de que el decaimiento no deje fonon residual; cota DURA por ciclo).
       'ep': 1/(2 - e^{-lam^2}) (cota por episodios: cada ciclo aporta 1 episodio puro y, con prob. 1-e^{-lam^2}, otro espurio).
       Ojo: 1/(2-u) >= u, asi que e^{-lam^2} NO es cota dura en conteo por episodios (viola a O(lam^4))."""
    u = np.exp(-lam ** 2)
    return u if kind == 'exp' else 1.0 / (2.0 - u)


def closed_parts(lam, Om, kap, gam, gph, n, mmax=8):
    """Factores de impureza (cada uno ~ probabilidad de fallo por ciclo):
    mer = fusion por decaimiento a mitad de cascada y re-excitacion, 1-prod_j(1-p_j r_j)  [modelo previo]
    fcm = perdida Franck-Condon de decaimientos a mitad de cascada que NO se re-excitan: sum_j p_j (1-r_j)(1-d_jj^2)  [NUEVO]
    off = fraccion de excitaciones en las colas lorentzianas de lineas m != n."""
    keep1 = 1.0
    keep2 = 1.0
    for j in range(1, n + 1):
        G = Gamma_j(lam, Om, kap, gam, gph, n, j)
        p = gam / (j * kap + gam); r = G / (G + j * kap); fc = 1 - dmat(lam, j, j) ** 2
        keep1 *= 1 - p * r
        keep2 *= 1 - p * (r + (1 - r) * fc)

    def R(m):
        gc = (m * kap + gam + gph) / 2
        return 2 * Om ** 2 * dmat(lam, m, 0) ** 2 * gc / ((m - n) ** 2 + gc ** 2)
    Rn = R(n)
    Roff = sum(R(m) for m in range(1, mmax) if m != n)
    return 1 - keep1, Roff / (Rn + Roff), 1 - keep2


def P_closed(lam, Om, kap, gam, gph, n, floor='exp', parts=False, fcmid=True):
    """Pureza cerrada por episodios: P_n = floor(lam) * prod_j [1 - p_j (r_j + (1-r_j)(1-d_jj^2))] * (1 - eps_off).
       p_j = gam/(j kap + gam) (decae con j fonones en cavidad), r_j = Gamma_j/(Gamma_j + j kap) (re-excitacion antes de emitir -> fusion),
       (1-d_jj^2) = prob. Franck-Condon de que el decaimiento cambie el numero de fonones, eps_off = R_off/(R_n + R_off) (colas m != n).
       fcmid=False recupera la formula previa (scout B): floor * prod_j (1 - p_j r_j) * (1 - eps_off)."""
    mer, off, fcm = closed_parts(lam, Om, kap, gam, gph, n)
    P = P_floor(lam, floor) * (1 - (fcm if fcmid else mer)) * (1 - off)
    return (P, mer, off, fcm) if parts else P


def Om_max(kap, X, lam, gam, gph, n, floor='exp', Omcap=1.0, fcmid=True):
    """Rabi maximo tal que P_closed >= X a kap dado (P_closed decrece con Om). np.nan si ni Om->0 alcanza X;
       np.inf si P>=X hasta Omcap. OJO: la cerrada solo esta validada para Om <~ 0.3."""
    f = lambda Om: P_closed(lam, Om, kap, gam, gph, n, floor, fcmid=fcmid) - X
    if f(1e-4) < 0:
        return np.nan
    if f(Omcap) >= 0:
        return np.inf
    return brentq(f, 1e-4, Omcap, xtol=1e-9)


def kappa_star(n, lam, Om, gam, gph, floor='exp', lo=1e-4, hi=0.2, ng=120, fcmid=True):
    """kappa que maximiza P_closed a (n, lam, Om, gam, gph) fijos (compromiso fusion ~ gam Om^2/kap^3 vs colas ~ kap^2).
       Devuelve (kappa*, P_closed(kappa*)). Si el maximo esta en el borde hi, devuelve hi (no hay optimo interior)."""
    ks = np.geomspace(lo, hi, ng)
    Ps = np.array([P_closed(lam, Om, k, gam, gph, n, floor, fcmid=fcmid) for k in ks])
    i = int(np.argmax(Ps))
    if i in (0, ng - 1):
        return ks[i], Ps[i]
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda lk: -P_closed(lam, Om, np.exp(lk), gam, gph, n, floor, fcmid=fcmid),
                        bounds=(np.log(ks[i - 1]), np.log(ks[i + 1])), method='bounded', options=dict(xatol=1e-9))
    return float(np.exp(r.x)), -r.fun


# ----------------------------------------------------------------------------------------------
# 3. Construccion de H y de las maestras (matrices densas)
# ----------------------------------------------------------------------------------------------
def _ops(N):
    b1 = np.diag(np.sqrt(np.arange(1, N)), 1).astype(complex)
    s1 = np.array([[0, 1], [0, 0]], dtype=complex)  # |g><x|, base (g,x)
    I2, IN = np.eye(2), np.eye(N)
    return np.kron(I2, b1), np.kron(s1, IN), np.kron(np.diag([0., 1.]), IN), np.kron(I2, IN)


class Model:
    """H (dim 2N), saltos {'b','g','phi'} (con sqrt(tasa) incluido), operador de numero de fonones, operadores de salto de cavidad."""
    def __init__(self, H, cops, nb, b, s, nx, Id, frame, jump):
        self.H, self.cops, self.nb, self.b, self.s, self.nx, self.Id, self.frame, self.jump = H, cops, nb, b, s, nx, Id, frame, jump
        self.d = H.shape[0]
        self._L = None

    @property
    def L(self):
        if self._L is None:
            self._L = liouvillian(self.H, list(self.cops.values()))
        return self._L


def build_lab(dL, lam, Om, kap, gam, gph, N=12):
    """Marco de laboratorio, maestra local: saltos sqrt(kap) b, sqrt(gam) s, sqrt(gph) s+s."""
    b, s, nx, Id = _ops(N)
    H = dL * nx + b.conj().T @ b + lam * nx @ (b + b.conj().T) + Om * (s + s.conj().T)
    c = {'b': np.sqrt(kap) * b, 'g': np.sqrt(gam) * s}
    if gph > 0:
        c['phi'] = np.sqrt(gph) * nx
    return Model(H, c, b.conj().T @ b, b, s, nx, Id, 'lab', 'local')


def build_pol(dL, lam, Om, kap, gam, gph, N=10, jump='dressed', fc=True):
    """Marco de polaron. jump='dressed' (b) | 'local' (b + lam s+s; ver signo abajo). fc=False: decaimiento s sin D+ (diagnostico)."""
    b, s, nx, Id = _ops(N)
    D = np.kron(np.eye(2), D_matrix(lam, N))
    H = (dL - lam ** 2) * nx + b.conj().T @ b + Om * (s.conj().T @ D + s @ D.conj().T)
    bj = b if jump == 'dressed' else b + lam * nx   # signo: con H_p ~ s+ D + s D+ (D=exp[-lam(b+-b)]) el salto local de laboratorio se transforma en b + lam s+s (con b - lam s+s no coincide con el laboratorio: error de signo del codigo previo)
    c = {'b': np.sqrt(kap) * bj, 'g': np.sqrt(gam) * (s @ D.conj().T if fc else s)}
    if gph > 0:
        c['phi'] = np.sqrt(gph) * nx
    return Model(H, c, b.conj().T @ b, b, s, nx, Id, 'pol', jump)


def liouvillian(H, cops):
    """Superoperador densa, vec en orden columna: vec(A rho B) = (B^T kron A) vec(rho)."""
    d = H.shape[0]
    I = np.eye(d)
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for c in cops:
        cdc = c.conj().T @ c
        L += np.kron(c.conj(), c) - 0.5 * np.kron(I, cdc) - 0.5 * np.kron(cdc.T, I)
    return L


def jump_super(C):
    """Parte de salto  rho -> C rho C^+  como superoperador."""
    return np.kron(C.conj(), C)


def _trvec(d):
    return np.eye(d).reshape(-1, order='F')


def steady_state(L, d):
    """Estado estacionario (matriz d x d) resolviendo L rho = 0, tr rho = 1 (se sustituye la fila (0,0), redundante)."""
    A = L.copy()
    A[0, :] = _trvec(d)
    rhs = np.zeros(d * d, dtype=complex)
    rhs[0] = 1
    return solve(A, rhs).reshape(d, d, order='F')


def flux_exact(m: Model, rho=None):
    """Flujo exacto de fonones = tasa de saltos de cavidad = Tr[C_b^+ C_b rho]."""
    if rho is None:
        rho = steady_state(m.L, m.d)
    Cb = m.cops['b']
    return float(np.real(np.trace(Cb.conj().T @ Cb @ rho)))


# ----------------------------------------------------------------------------------------------
# 4. Estadistica de conteo exacta (renovacion)
# ----------------------------------------------------------------------------------------------
def _renewal(L, End, Count, d, K, shift, rho=None):
    """Distribucion exacta P(k), k=0..K, de eventos 'Count' (+shift si el evento que cierra tambien cuenta) entre dos eventos 'End'.
    L0 = L - End - Count;  A = -L0;  P(k) = Tr[End (A^-1 Count)^{k-shift} A^-1 r_s],  r_s = End rho_ss / Tr(End rho_ss).
    Devuelve (P, T_medio_entre_End, flujo=<k>/T)."""
    tr = _trvec(d)
    A = lu_factor(-(L - End - Count))
    rho = (steady_state(L, d) if rho is None else rho).reshape(-1, order='F')
    rate = np.real(tr @ (End @ rho))
    rs = End @ rho / rate
    P = np.zeros(K + 1)
    v = lu_solve(A, rs)
    for k in range(K + 1 - shift):
        P[k + shift] = np.real(tr @ (End @ v))
        v = lu_solve(A, Count @ v)
    T = 1.0 / rate
    return P, T, (np.arange(K + 1) * P).sum() / T


def cycle_dist(m: Model, K=12, rho=None):
    """Distribucion de fonones por ciclo (entre saltos radiativos). Devuelve (P[0..K], T_ciclo, flujo<k>/T)."""
    Jg = jump_super(m.cops['g'])
    Jb = jump_super(m.cops['b'])
    return _renewal(m.L, Jg, Jb, m.d, K, 0, rho)


def episode_dist(m: Model, K=12, rho=None):
    """Distribucion de fonones por episodio (entre retornos de la cavidad al vacio). Devuelve (P[0..K], T_ep, flujo<k>/T).
    Clasificacion por numero de fonones pre-salto: P1 (+P0 en 'local') cierra el episodio, P>=2 cuenta."""
    N = m.d // 2
    nn = np.kron(np.eye(2), np.diag(np.arange(N)))
    P0 = np.kron(np.eye(2), np.diag([1.] + [0.] * (N - 1)))
    P1 = np.kron(np.eye(2), np.diag([0., 1.] + [0.] * (N - 2)))
    Pg = np.eye(m.d) - P0 - P1
    Cb = m.cops['b']
    closers = P1 + P0 if m.jump == 'local' else P1
    End = jump_super(Cb @ closers)
    Cnt = jump_super(Cb @ Pg)
    return _renewal(m.L, End, Cnt, m.d, K, 1, rho)


def purity(m: Model, n, kind='episode', K=None):
    K = K or (2 * n + 6)
    P = (episode_dist if kind == 'episode' else cycle_dist)(m, K)[0]
    return float(P[n])


def exact_point(n, lam, Om, kap, gam, gph, N=None, dL=None, jump='dressed', kind='episode', K=None, flux=True):
    """Evalua pureza exacta (episodios o ciclos) y flujo exacto en la resonancia (dL = delta_res por defecto).
    N por defecto: n + 5. Devuelve dict(P, flux, dist, T)."""
    N = N or (n + 6)
    dL = delta_res(n, lam, Om) if dL is None else dL
    m = build_pol(dL, lam, Om, kap, gam, gph, N, jump)
    K = K or (2 * n + 6)
    rho = steady_state(m.L, m.d)
    P, T, F = (episode_dist if kind == 'episode' else cycle_dist)(m, K, rho)
    out = dict(P=float(P[n]), dist=P, T=T, flux_renewal=F, flux=flux_exact(m, rho), sumP=float(P.sum()))
    return out


# ----------------------------------------------------------------------------------------------
# 5. Modelo de tasas (Pauli) con conteo de episodios: nivel intermedio semianalitico (sistema lineal de 2M estados)
# ----------------------------------------------------------------------------------------------
def rate_model(lam, Om, kap, gam, gph, n, M=None, K=None, raman='lorentz'):
    """Ecuacion maestra de poblaciones sobre (q,m), q in {g,x}, m=0..M-1 (marco de polaron, maestra vestida).
    Transiciones: (i) Raman estimulado g,m' <-> x,m con tasa  R = 2 Om^2 d_{mm'}^2 gc/(Delta^2+gc^2), Delta = m-m'-n,
    gc=((m+m')kap+gam+gph)/2  (raman='lorentz': cubre resonancia y colas lorentzianas); raman='scout': resonante 2Om^2 d^2/gc para m=m'+n
    y desfase inducido gph (Om d/Delta)^2 fuera de resonancia (modelo previo); (ii) emision m->m-1 a m kap (cuenta; m=1->0 cierra episodio);
    (iii) decaimiento radiativo x,m -> g,m' a gam d_{mm'}^2 (Franck-Condon).
    Devuelve dict(P=P_ep[0..K], flux, T, dist)."""
    M = M or n + 8
    K = K or 2 * n + 6
    S = 2 * M
    idx = lambda q, m: q * M + m
    W = np.zeros((S, S))           # W[j,i]: i -> j
    def add(i, j, r):
        W[j, i] += r; W[i, i] -= r
    for m in range(M):
        for mp in range(M):
            d2 = dmat(lam, m, mp) ** 2
            D = m - mp - n
            gc = ((m + mp) * kap + gam + gph) / 2
            if raman == 'lorentz':
                r = 2 * Om ** 2 * d2 * gc / (D ** 2 + gc ** 2)
            else:
                r = 2 * Om ** 2 * d2 / gc if D == 0 else gph * (Om ** 2) * d2 / D ** 2
            add(idx(0, mp), idx(1, m), r); add(idx(1, m), idx(0, mp), r)
            add(idx(1, m), idx(0, mp), gam * d2)
    Cnt = np.zeros((S, S)); End = np.zeros((S, S))
    for q in (0, 1):
        for m in range(1, M):
            i, j = idx(q, m), idx(q, m - 1)
            r = m * kap
            W[i, i] -= r
            (End if m == 1 else Cnt)[j, i] += r
    one = np.ones(S)
    Wt = W + End + Cnt
    A = np.vstack([Wt[1:], one])
    p = np.linalg.solve(A, np.r_[np.zeros(S - 1), 1.0])
    flux = kap * sum(m * (p[idx(0, m)] + p[idx(1, m)]) for m in range(M))
    rate = one @ (End @ p)
    rs = End @ p / rate
    Ai = np.linalg.inv(-W)         # W ya incluye salida por End y Cnt en la diagonal; W = L0 (sin entradas de End/Cnt)
    P = np.zeros(K + 1); v = Ai @ rs
    for k in range(1, K + 1):
        P[k] = one @ (End @ v); v = Ai @ (Cnt @ v)
    return dict(P=P, flux=flux, T=1 / rate, dist=P)
