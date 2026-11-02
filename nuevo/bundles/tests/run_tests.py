"""Pruebas automaticas (compatibles con pytest: funciones test_*; sin pytest: python3 tests/run_tests.py)."""
import os, sys, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from bundles_core import *
from bundles_mc import *

B = dict(lam=.14, Om=.1, kap=.003, gam=2e-4, gph=4e-4)

def test_dmat_vs_expm():
    N = 60; b = np.diag(np.sqrt(np.arange(1, N)), 1)
    for lam in (0.1, 0.3, 0.7):
        D = expm(-lam * (b.T - b))
        assert np.abs(D[:8, :8] - D_matrix(lam, 8)).max() < 1e-12

def test_eps_limit_lambda0_binIII():
    for n in (1, 2, 3):
        for Om in (0.02, 0.05):
            e = eps_n(0.0, Om, n); assert abs(e - 2 * Om**2 / n) < 1e-13
            assert abs(e - (n - np.sqrt(n**2 - 4 * Om**2))) < 3 * Om**4 / n**3   # Bin-III hasta O(Om^4)

def test_resonance_matches_flux_peak():
    for n, Om in ((2, 0.1), (2, 0.2), (3, 0.2)):
        f = lambda dL: -flux_exact(build_pol(dL, B['lam'], Om, B['kap'], B['gam'], B['gph'], n + 6))
        d0 = delta_res(n, B['lam'], Om)
        r = minimize_scalar(f, bounds=(d0 - 0.01, d0 + 0.01), method='bounded', options=dict(xatol=1e-7))
        assert abs(r.x - d0) < 5e-4, (n, Om, r.x - d0)

def test_flux_zero_when_Om_to_zero():
    f = [exact_point(2, B['lam'], Om, B['kap'], B['gam'], B['gph'], N=8)['flux'] for Om in (1e-4, 1e-3, 2e-3)]
    assert f[0] < 1e-8 and abs(f[2] / f[1] / 4 - 1) < 0.03     # J ~ Om^2 mientras Gamma_0 << gam

def test_normalization_and_renewal_flux():
    for jump in ('dressed', 'local'):
        r1 = exact_point(2, **{k: B[k] for k in ('lam', 'Om', 'kap', 'gam', 'gph')}, N=9, jump=jump, K=16)
        assert abs(r1['sumP'] - 1) < 1e-5
        r2 = exact_point(2, **{k: B[k] for k in ('lam', 'Om', 'kap', 'gam', 'gph')}, N=9, jump=jump, kind='cycle', K=16)
        assert abs(r2['sumP'] - 1) < 1e-6
        assert abs(r2['flux_renewal'] / r2['flux'] - 1) < 1e-5      # teorema de renovacion: <k>/T = flujo
    # episodios (dressed): <k>/T = flujo
    r1 = exact_point(2, **{k: B[k] for k in ('lam', 'Om', 'kap', 'gam', 'gph')}, N=9, K=16)
    assert abs(r1['flux_renewal'] / r1['flux'] - 1) < 1e-4

def test_N_invariance():
    P = [exact_point(2, B['lam'], B['Om'], B['kap'], B['gam'], B['gph'], N=N, K=10)['P'] for N in (7, 9, 12)]
    assert max(P) - min(P) < 1e-6, P
    P = [exact_point(3, B['lam'], 0.2, 0.0015, B['gam'], B['gph'], N=N, K=12)['P'] for N in (9, 11, 13)]
    assert max(P) - min(P) < 1e-6, P

def test_lab_equals_polaron_local():
    dL = delta_res(2, B['lam'], B['Om'])
    ml = build_lab(dL, B['lam'], B['Om'], B['kap'], B['gam'], B['gph'], N=14)
    mp = build_pol(dL, B['lam'], B['Om'], B['kap'], B['gam'], B['gph'], N=10, jump='local')
    Pl, Tl, _ = cycle_dist(ml, 8); Pp, Tp, _ = cycle_dist(mp, 8)
    assert np.abs(Pl - Pp).max() < 1e-7 and abs(Tl / Tp - 1) < 1e-7
    assert abs(flux_exact(ml) / flux_exact(mp) - 1) < 1e-7
    md = build_pol(dL, B['lam'], B['Om'], B['kap'], B['gam'], B['gph'], N=10, jump='dressed')
    assert abs(cycle_dist(md, 8)[0][2] - Pp[2]) > 0.1     # la vestida SI difiere (nube DC)

def test_local_sign_regression():
    # el salto local en polaron debe ser b + lam s+s (con H ~ s+ D + s D+); con b - lam s+s difiere (error del codigo previo)
    import bundles_core as bc
    dL = delta_res(2, B['lam'], B['Om']); b, s, nx, Id = bc._ops(10)
    mp = build_pol(dL, B['lam'], B['Om'], B['kap'], B['gam'], B['gph'], N=10, jump='dressed')
    mp.cops['b'] = np.sqrt(B['kap']) * (b - B['lam'] * nx); mp._L = None
    ml = build_lab(dL, B['lam'], B['Om'], B['kap'], B['gam'], B['gph'], N=14)
    assert np.abs(cycle_dist(ml, 8)[0] - cycle_dist(mp, 8)[0]).max() > 5e-4

def test_rate_model_close_to_exact():
    r = exact_point(2, B['lam'], B['Om'], B['kap'], B['gam'], B['gph'], N=9)
    q = rate_model(B['lam'], B['Om'], B['kap'], B['gam'], B['gph'], 2)
    assert abs(q['P'][2] - r['P']) < 2e-3 and abs(q['flux'] / r['flux'] - 1) < 0.02

def test_closed_vs_exact_base():
    r = exact_point(2, B['lam'], B['Om'], B['kap'], B['gam'], B['gph'], N=9)
    assert abs(P_closed(B['lam'], B['Om'], B['kap'], B['gam'], B['gph'], 2) / r['P'] - 1) < 0.03
    assert abs(flux_closed(B['lam'], B['Om'], B['kap'], B['gam'], B['gph'], 2) / r['flux'] - 1) < 0.05

def test_floor_bounds():
    for n, Om, kap in ((2, .05, .01), (2, .1, .005), (3, .2, .0015)):
        for lam in (0.1, 0.2):
            c = exact_point(n, lam, Om, kap, B['gam'], B['gph'], kind='cycle')
            e = exact_point(n, lam, Om, kap, B['gam'], B['gph'])
            assert c['P'] <= np.exp(-lam**2) + 1e-9          # cota dura por ciclos
            assert e['P'] <= P_floor(lam, 'ep') + 1e-6        # cota por episodios

def test_Om_max_roundtrip():
    for n in (2, 3):
        for kap in (0.002, 0.005):
            Om = Om_max(kap, 0.93, B['lam'], B['gam'], B['gph'], n)
            if np.isfinite(Om):
                assert abs(P_closed(B['lam'], Om, kap, B['gam'], B['gph'], n) - 0.93) < 1e-7

def test_cluster_sizes():
    t = [np.array([100., 110, 120, 1000, 1010, 5000, 5005, 5006, 5007])]
    s = cluster_sizes(t, 50., 0., 6000.)
    assert list(s) == [3, 2, 4] or list(s) == [2, 4]   # el primero se descarta si empieza a < tau de burn(0): 100>50 -> se conserva
    s2 = cluster_sizes(t, 50., 0., 5030.)               # el ultimo termina a <tau del final -> se descarta
    assert list(s2) == [3, 2]

def test_mc_flux_and_cycle_consistency():
    m = build_pol(delta_res(2, B['lam'], B['Om']), B['lam'], B['Om'], B['kap'], B['gam'], B['gph'], 9)
    r = run_mc(m, 1.5e5, 80, seed=7)
    st = mc_stats(r, 2, [5 / B['kap']])
    J = flux_exact(m)
    assert abs(st['flux'] - J) < 4 * st['flux_err'], (st['flux'], J, st['flux_err'])
    # tasa radiativa MC vs exacta
    Jg = np.real(np.trace(m.cops['g'].conj().T @ m.cops['g'] @ steady_state(m.L, m.d)))
    assert abs(st['rate_rad'] / Jg - 1) < 0.15

if __name__ == '__main__':
    fails = 0
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            t = time.time()
            try:
                f(); print('OK  ', k, '%.1fs' % (time.time() - t))
            except Exception:
                fails += 1; print('FAIL', k); traceback.print_exc()
    print('fallos:', fails); sys.exit(1 if fails else 0)
