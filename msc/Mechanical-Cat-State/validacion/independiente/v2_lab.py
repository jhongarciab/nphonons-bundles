"""V2: modelo completo de Ma en el MARCO DE LABORATORIO.

H(t) = w a†a + (wq/2) σz + (a + a†)(gx σx + gz σz) + Ω (σ+ e^{-i wp t} + σ- e^{+i wp t})
Disipador: κ D[σ-], κ = 2Γ = 0.03 (Ma usa Γ[2σ-ρσ+ - ...]). Sin pérdida del oscilador.

Método (distinto del original, que usó marco rotante con Floquet sobre T = 4π/wp y potencias por cuadrados):
  - H(t) es periódico con T_p = 2π/wp en el laboratorio. Se calcula el propagador del
    superoperador sobre UN período T_p integrando la ecuación maestra (qutip.propagator).
  - ρ(nT_p) se obtiene aplicando ese propagador vector a vector, período a período (sin potencias).
  - Muestreo estroboscópico en t = nT_p (fase 0 del drive). En esos instantes el marco que rota
    a wp/2 coincide con el de laboratorio salvo a -> (-1)^n a, lo que deja invariante el espacio
    del código {|α>, |-α>} y ⟨a²⟩; P_e es invariante.
  - Al final se integra un período más con 40 muestras para el promedio y el rango de micromovimiento.
Uso: python v2_lab.py wp N Gt_max [salida.npz]
"""
import sys, time
import numpy as np
import qutip as qt

w, wq, g, th, Om, Gam = 6.0, 12.0, 0.3, np.pi / 4, 0.06, 0.015
kap = 2 * Gam
gx, gz = -g * np.sin(th), g * np.cos(th)
G = 2 * gx * gz / w
alpha2 = Om / G                      # -4 -> α = 2i
alpha = np.sqrt(complex(alpha2))
OPTS = dict(atol=1e-12, rtol=1e-10, nsteps=10**6)


def operadores(N):
    a = qt.tensor(qt.destroy(N), qt.qeye(2))
    sm = qt.tensor(qt.qeye(N), qt.sigmam())   # |e>=basis(2,0), |g>=basis(2,1)
    sz = qt.tensor(qt.qeye(N), qt.sigmaz())
    sx = qt.tensor(qt.qeye(N), qt.sigmax())
    return a, sm, sz, sx


def hamiltoniano(N, wp):
    a, sm, sz, sx = operadores(N)
    H0 = w * a.dag() * a + 0.5 * wq * sz + (a + a.dag()) * (gx * sx + gz * sz)
    return [H0, [Om * sm.dag(), lambda t: np.exp(-1j * wp * t)],
                [Om * sm, lambda t: np.exp(1j * wp * t)]], [np.sqrt(kap) * sm]


def observables(N):
    a, sm, sz, sx = operadores(N)
    ca, cb = qt.coherent(N, alpha), qt.coherent(N, -alpha)
    # base ortonormal del código: gatos par e impar
    cp, cm = (ca + cb).unit(), (ca - cb).unit()
    Pc = qt.tensor(cp * cp.dag() + cm * cm.dag(), qt.qeye(2))
    par = qt.tensor((1j * np.pi * qt.num(N)).expm(), qt.qeye(2))
    Pe = sm.dag() * sm
    return dict(Pc=Pc, Pe=Pe, a2=a * a, par=par)


def validar(rho):
    M = rho.full()
    return (abs(np.trace(M) - 1), np.linalg.norm(M - M.conj().T), np.linalg.eigvalsh((M + M.conj().T) / 2).min())


def medir(rho, obs):
    return {k: qt.expect(o, rho) for k, o in obs.items()}


def main():
    wp, N, Gtmax = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    out = sys.argv[4] if len(sys.argv) > 4 else f"v2_wp{wp:.4f}_N{N}.npz"
    Tp = 2 * np.pi / wp
    H, c = hamiltoniano(N, wp)
    t0 = time.time()
    U = qt.propagator(H, Tp, c, options=OPTS).full()       # superoperador (vec columna)
    tprop = time.time() - t0
    obs = observables(N)
    D = 2 * N
    rho0 = qt.ket2dm(qt.tensor(qt.basis(N, 0), qt.basis(2, 1)))   # vacío ⊗ |g>
    v = rho0.full().reshape(-1, order='F')
    nmax = int(np.ceil(Gtmax / Gam / Tp))
    paso = max(1, nmax // 600)
    reg, peor = [], [0, 0, 0]
    for n in range(nmax + 1):
        if n % paso == 0 or n == nmax:
            rho = qt.Qobj(v.reshape(D, D, order='F'), dims=rho0.dims)
            m = medir(rho, obs)
            tr, he, mi = validar(rho)
            peor = [max(peor[0], tr), max(peor[1], he), min(peor[2], mi)]
            reg.append([n * Tp * Gam, m['Pc'].real, m['Pe'].real, abs(m['a2']), m['a2'].real, m['a2'].imag, m['par'].real])
        v = U @ v
    reg = np.array(reg)
    # micromovimiento: un período más desde el último estado estroboscópico, 40 muestras
    rhoN = qt.Qobj(v.reshape(D, D, order='F'), dims=rho0.dims)
    ts = np.linspace(0, Tp, 41) + nmax * Tp + Tp
    # nota: v ya avanzó un período extra tras el último registro; se usa como inicio en t=(nmax+1)Tp
    res = qt.mesolve(H, rhoN, ts, c, options=OPTS)
    # se miden en el marco rotante (oscilador a wp/2, qubit a wp): ρ_rot = R ρ_lab R†,
    # R = exp[i t (wp/2)(a†a + σz)]; en t = nT_p coincide con el laboratorio (fase global).
    a, sm, sz, sx = operadores(N)
    gen = (wp / 2) * (a.dag() * a + sz)
    micro = {k: [] for k in obs}
    for t, r in zip(ts, res.states):
        R = (1j * t * gen).expm()
        rr = R * r * R.dag()
        for k, o in obs.items():
            micro[k].append(qt.expect(o, rr))
    micro = {k: np.array(x) for k, x in micro.items()}
    np.savez(out, reg=reg, micro_Pc=micro['Pc'], micro_Pe=micro['Pe'], micro_a2=micro['a2'],
             micro_par=micro['par'], peor=peor, tprop=tprop, wp=wp, N=N)
    f = reg[-1]
    print(f"wp={wp} N={N} Γt={f[0]:.1f} (estrob. t=nT_p) Pc={f[1]:.5f} Pe={f[2]:.5f} |a2|={f[3]:.4f} a2={f[4]:+.3f}{f[5]:+.3f}i par={f[6]:+.4f}")
    for k in ('Pc', 'Pe', 'a2', 'par'):
        x = micro[k]; x = np.abs(x) if k == 'a2' else x.real
        print(f"   micro {k}: prom={x[:-1].mean():.5f} min={x.min():.5f} max={x.max():.5f}")
    print(f"   peor |Tr-1|={peor[0]:.1e} ||ρ-ρ†||={peor[1]:.1e} min eig={peor[2]:.1e}; t_prop={tprop:.0f}s")


if __name__ == '__main__':
    main()
