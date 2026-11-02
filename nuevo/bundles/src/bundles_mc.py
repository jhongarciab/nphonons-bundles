"""bundles_mc: Monte Carlo de trayectorias cuanticas (integrador propio, exacto entre saltos) y agrupacion operativa en bundles.

Como funciona (rapido y sin ODE): en el marco de polaron el H_eff = H - i/2 sum C+C es independiente del tiempo, luego la evolucion
sin salto es U = expm(-i H_eff dt) exacta. Todas las trayectorias avanzan en bloque (matmul d x ntraj) con paso dt; cuando la norma
cae por debajo del numero aleatorio r se localiza el salto dentro del paso (interpolacion exponencial de la norma, error < dt << tau_c),
se evoluciona exactamente hasta ese instante, se elige el canal con peso ||C_c psi||^2 y se re-alinea con la rejilla.
Canales: 0 = salto de cavidad (fonon), 1 = decaimiento radiativo (foton), 2 = desfase puro.

Agrupacion operativa: dos eventos de fonon consecutivos de una trayectoria pertenecen al mismo bundle si su separacion < tau_c.
Pureza P_n^{MC}(tau_c) = (#bundles con exactamente n fonones)/(#bundles). Se descartan el primer bundle (transitorio) y el ultimo si
termina a menos de tau_c del final de la trayectoria.
"""
import numpy as np
from scipy.linalg import expm


def run_mc(model, T, ntraj, seed=1234, dt=2.0, burn=3000.0):
    """Simula ntraj trayectorias de duracion T desde |g,0>. Devuelve dict con tiempos de evento por canal.
    ev[c] = lista (len ntraj) de arrays de tiempos del canal c (solo t >= burn)."""
    d = model.d
    names = list(model.cops.keys())
    assert names[0] == 'b' and names[1] == 'g'
    C = [model.cops[k] for k in names]
    Heff = model.H - 0.5j * sum(c.conj().T @ c for c in C)
    U = expm(-1j * Heff * dt)
    rng = np.random.default_rng(seed)
    Psi = np.zeros((d, ntraj), complex)
    Psi[0, :] = 1.0
    r = rng.random(ntraj)
    ev_t = [[] for _ in C]
    ev_i = [[] for _ in C]
    nsteps = int(round(T / dt))
    for step in range(nsteps):
        Pn = U @ Psi
        nrm = (Pn.real ** 2 + Pn.imag ** 2).sum(0)
        jmp = np.nonzero(nrm < r)[0]
        Psi_old = Psi
        Psi = Pn
        for i in jmp:
            tcur = step * dt
            psi0 = Psi_old[:, i].copy()
            tl = dt
            pn = Pn[:, i]
            while True:
                n0 = float(np.vdot(psi0, psi0).real)
                n1 = float(np.vdot(pn, pn).real)
                if n1 >= r[i]:
                    Psi[:, i] = pn
                    break
                s = tl * np.log(r[i] / n0) / np.log(n1 / n0)
                s = min(max(s, 0.0), tl)
                ps = expm(-1j * Heff * s) @ psi0
                w = np.array([np.vdot(c @ ps, c @ ps).real for c in C])
                ch = rng.choice(len(C), p=w / w.sum())
                newp = C[ch] @ ps
                psi0 = newp / np.linalg.norm(newp)
                tcur += s
                tl -= s
                ev_t[ch].append(tcur)
                ev_i[ch].append(i)
                r[i] = rng.random()
                pn = expm(-1j * Heff * tl) @ psi0 if tl > 0 else psi0
                if tl <= 0:
                    Psi[:, i] = psi0
                    break
    out = dict(T=T, ntraj=ntraj, burn=burn, dt=dt, names=names)
    ev = []
    for c in range(len(C)):
        t = np.array(ev_t[c]); ii = np.array(ev_i[c], dtype=int)
        ev.append([np.sort(t[(ii == k) & (t >= burn)]) for k in range(ntraj)])
    out['ev'] = ev
    return out


def cluster_sizes(times_list, tau, burn, T):
    """Tamaños de bundle (nº de fonones) por agrupacion temporal con separacion < tau."""
    sizes = []
    for t in times_list:
        if len(t) == 0:
            continue
        br = np.nonzero(np.diff(t) >= tau)[0]
        starts = np.r_[0, br + 1]
        ends = np.r_[br, len(t) - 1]
        sz = ends - starts + 1
        keep = np.ones(len(sz), bool)
        if t[starts[0]] - burn < tau:
            keep[0] = False
        if T - t[ends[-1]] < tau:
            keep[-1] = False
        sizes.append(sz[keep])
    return np.concatenate(sizes) if sizes else np.zeros(0, int)


def mc_stats(res, n, taus, kmax=12):
    """Para cada tau: pureza P_n, error binomial, histograma de tamanos, nº de bundles; y flujo de fonones (media, error entre trayectorias)."""
    ev0 = res['ev'][0]
    Teff = res['T'] - res['burn']
    cnt = np.array([len(t) for t in ev0], float)
    flux = cnt.mean() / Teff
    flux_err = cnt.std(ddof=1) / np.sqrt(len(cnt)) / Teff
    out = dict(flux=flux, flux_err=flux_err, taus=np.array(taus), P=[], Perr=[], hist=[], nb=[])
    for tau in taus:
        sz = cluster_sizes(ev0, tau, res['burn'], res['T'])
        h = np.bincount(sz, minlength=kmax + 1)[:kmax + 1]
        nb = len(sz)
        p = h[n] / max(nb, 1)
        out['P'].append(p); out['Perr'].append(np.sqrt(p * (1 - p) / max(nb, 1))); out['hist'].append(h); out['nb'].append(nb)
    for k in ('P', 'Perr', 'hist', 'nb'):
        out[k] = np.array(out[k])
    out['rate_rad'] = np.mean([len(t) for t in res['ev'][1]]) / Teff
    return out


# ---- predicciones analiticas sencillas del efecto operativo de tau_c ----
def split_prob(n, kap, tau):
    """Prob. de que un bundle de n fonones se parta por un hueco >= tau (huecos ~ Exp(j kap), j=1..n-1)."""
    return 1 - np.prod([1 - np.exp(-j * kap * tau) for j in range(1, n)]) if n > 1 else 0.0


def merge_prob(n, lam_unused, Om_unused, kap, gam, G0, tau):
    """Prob. de que el hueco entre el ultimo fonon de un bundle y el primero del siguiente sea < tau:
    suma de tres exponenciales: decaimiento (rate gam), re-excitacion (rate G0), primera emision (rate n kap). CDF hipoexponencial."""
    r = np.array([gam, G0, n * kap], float)
    # CDF de suma de exponenciales con tasas distintas: 1 - sum_i prod_{j!=i} r_j/(r_j-r_i) e^{-r_i tau}
    # (tasas casi iguales -> perturbar ligeramente)
    r = r * (1 + 1e-6 * np.arange(3))
    cdf = 1.0
    for i in range(3):
        coef = np.prod([r[j] / (r[j] - r[i]) for j in range(3) if j != i])
        cdf -= coef * np.exp(-r[i] * tau)
    return float(cdf)


def P_operational_pred(P_ep, n, kap, gam, G0, tau):
    """Prediccion aproximada de P^{MC}(tau): P_ep (1-q)(1-s)/(1+s), q = prob. de fusion de bundles consecutivos, s = prob. de particion."""
    q = merge_prob(n, None, None, kap, gam, G0, tau)
    s = split_prob(n, kap, tau)
    return P_ep * (1 - q) * (1 - s) / (1 + s)
