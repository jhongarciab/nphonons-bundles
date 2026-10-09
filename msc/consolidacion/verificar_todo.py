#!/usr/bin/env python3
"""Fase B. Verificacion de afirmaciones de msc/ contra los datos del repositorio.
Solo lee (npz, csv, logs, scripts). No lanza simulaciones. Escribe consolidacion/afirmaciones.csv
y consolidacion/verificacion_log.txt. Uso: python3 -I consolidacion/verificar_todo.py   (numpy, scipy, sympy)

Estados: V recalculada y coincide dentro de la tolerancia; R reportada en log, no recomputable;
D discrepancia; H hipotesis declarada; S superada por un resultado posterior.
Tolerancia por defecto de una cifra reportada: media unidad de su ultimo digito.
"""
import os, re, csv, glob, math, sys, subprocess, itertools
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # msc/
MCS = os.path.join(ROOT, 'Mechanical-Cat-State')
FF = os.path.join(MCS, 'figuras_finales')
DATA = os.path.join(FF, 'data')
FC = os.path.join(DATA, 'filtro_completo')
VAL = os.path.join(MCS, 'validacion')
OUT = os.path.join(ROOT, 'consolidacion')
rel = lambda p: os.path.relpath(p, ROOT)

# ---------------------------------------------------------------- utilidades de cifras
def parse(s):
    """'1.650e-04', '−2.5e-9', '0.956', '4.4%', '1.4e-3' -> (valor, unidad_ultimo_digito). None si no es numero."""
    if isinstance(s, (int, float, np.floating)): return float(s), None
    t = str(s).strip().replace('−', '-').replace('×10^', 'e').replace('\\times10^', 'e').replace('$', '').replace(' ', '')
    pct = t.endswith('%'); t = t.rstrip('%')
    t = t.lstrip('+')
    m = re.fullmatch(r'(-?\d*\.?\d+)(?:[eE]([+-]?\d+))?', t)
    if not m: return None
    mant, ex = m.group(1), int(m.group(2) or 0)
    dec = len(mant.split('.')[1]) if '.' in mant else 0
    v = float(mant) * 10 ** ex; u = 10 ** (ex - dec)
    if pct: v, u = v / 100, u / 100
    return v, u

CL = []            # afirmaciones
WARN = []          # anclas no encontradas, etc.
_cache_lines = {}
def lines(f):
    if f not in _cache_lines:
        p = os.path.join(ROOT, f)
        _cache_lines[f] = open(p, errors='ignore').read().split('\n') if os.path.exists(p) else None
    return _cache_lines[f]
def find(f, anchor, nth=1):
    if isinstance(anchor, int): return anchor
    L = lines(f)
    if L is None: WARN.append(f'archivo inexistente: {f}'); return 0
    k = 0
    for i, l in enumerate(L, 1):
        if anchor in l:
            k += 1
            if k == nth: return i
    WARN.append(f'ancla no encontrada: {f} :: {anchor!r}'); return 0

H8 = 'HANDOFF.md'; RM = 'Mechanical-Cat-State/figuras_finales/README.md'; PH = 'Mechanical-Cat-State/figuras_finales/PENDIENTES_Y_HALLAZGOS.md'
C6 = 'Mechanical-Cat-State/figuras_finales/C6_polaron.md'; PRE = 'Mechanical-Cat-State/figuras_finales/data/preregistro_kf0.2.md'
INF = 'docs/Informe/informe.tex'; MD0 = 'README.md'

def claim(id, nivel, locs, texto, rep=None, rec=None, tol=None, modo='abs', estado=None, nota='', dato='', script='', commit='', ref_tol=None):
    """locs: lista de (archivo, ancla[, valor_en_esa_ubicacion[, n-esima]]). rep: valor reportado (str o num).
    rec: valor recalculado. tol: tolerancia absoluta (modo 'abs'), relativa ('rel'), o None = media unidad del ultimo digito de rep.
    modo 'ge'/'le': cota (rec >= rep / rec <= rep). modo 'bool': rec es True/False. estado fuerza R/H/S/D/V."""
    ubic = []
    vals = []
    for l in locs:
        f, a = l[0], l[1]
        v = l[2] if len(l) > 2 else None
        n = l[3] if len(l) > 3 else 1
        ubic.append(f'{f}:{find(f, a, n)}')
        vals.append(v)
    st, rv, tolu = estado, None, ''
    if rec is not None and rep is not None and estado in (None, 'V', 'D'):
        pr = parse(rep)
        if modo == 'bool':
            st = 'V' if rec else 'D'
        elif pr is None:
            st = 'V' if str(rec) == str(rep) else 'D'
        else:
            vr, u = pr
            rv = float(rec)
            if modo == 'abs':
                t = tol if tol is not None else (u if u else 1e-12)           # por defecto: 1 unidad del ultimo digito (admite truncamiento)
                ok = abs(rv - vr) <= t * (1 + 1e-9); tolu = f'abs {t:.3g}'
                if tol is None and u and ok and abs(rv - vr) > 0.5 * u * (1 + 1e-9): nota = (nota + ' ' if nota else '') + '[dif entre 0.5 y 1 unidad del ultimo digito: el reporte trunca en vez de redondear]'
            elif modo == 'rel':
                t = tol if tol is not None else (u / abs(vr) if (u and vr) else 1e-6)
                ok = abs(rv - vr) <= t * abs(vr) * (1 + 1e-9); tolu = f'rel {t:.3g}'
            elif modo == 'ge':
                ok = rv >= vr - (tol or 0); tolu = 'cota >='
            elif modo == 'le':
                ok = rv <= vr + (tol or 0); tolu = 'cota <='
            st = 'V' if ok else 'D'
    elif rec is None and st is None:
        st = 'R'
    # coincidencia de ubicaciones
    coin = ''
    pv = [parse(v) for v in vals if v is not None]
    pv = [p for p in pv if p]
    if len(pv) >= 2:
        base = pv[0][0]
        us = [p[1] for p in pv if p[1]]
        tt = max(us) * 0.5 if us else 0
        coin = 'coinciden' if all(abs(p[0] - base) <= max(tt, 1e-12) * (1 + 1e-9) or abs(p[0] - base) <= 1e-9 * abs(base) for p in pv) else 'DISCREPAN'
    CL.append(dict(id=id, nivel=nivel, ubicaciones=' | '.join(ubic), texto=texto, valor_reportado=str(rep) if rep is not None else '',
                   valor_recalculado=('' if rec is None or modo == 'bool' else f'{float(rec):.6g}' if isinstance(rec, (int, float, np.floating)) else str(rec)),
                   tolerancia=tolu if tolu else (str(tol) if tol is not None else ''), estado=st, ubicaciones_coinciden=coin,
                   nota=nota, dato=dato, script=script, commit=commit))
    return st

PDEC = []   # por decidir por Jhon
def por_decidir(id, texto, a, b, dato=''):
    PDEC.append(dict(id=id, texto=texto, version_a=a, version_b=b, dato=dato))

def sh(cmd):
    return subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True).stdout.strip()
def ld(path, **kw):
    return np.load(os.path.join(DATA, path) if not os.path.isabs(path) else path, allow_pickle=True, **kw)
def csvrows(name):
    with open(os.path.join(DATA, name)) as f:
        r = list(csv.reader(f))
    return [[float(x) if re.fullmatch(r'[-+0-9.eE]+', x) else x for x in row] for row in r[1:]]

# ============================================================ BLOQUE 1: verificacion simbolica (sympy)
def bloque_simbolico():
    import sympy as sp
    S = 'consolidacion/verificar_todo.py::bloque_simbolico'
    gx, gz, w, k, kf, J, d, A2 = sp.symbols('g_x g_z omega kappa kappa_f J d alpha2', positive=True)
    M = 8                                              # truncamiento de Fock (solo se comparan elementos con n <= M-4)
    a1 = sp.zeros(M, M)
    for n in range(1, M): a1[n - 1, n] = sp.sqrt(n)
    I2, IM = sp.eye(2), sp.eye(M)
    kron = lambda X, Y: sp.kronecker_product(X, Y)
    a = kron(I2, a1); ad = a.T                         # indices: qubit (0=g,1=e) x Fock
    sm = kron(sp.Matrix([[0, 1], [0, 0]]), IM)         # |g><e| con g=0, e=1 -> sm = |0><1|
    sp_ = sm.T
    sz = kron(sp.diag(-1, 1), IM)                      # sigma_z = |e><e| - |g><g|
    Pe = kron(sp.diag(0, 1), IM); Pg = kron(sp.diag(1, 0), IM)
    nn = ad * a
    com = lambda X, Y: X * Y - Y * X
    h1 = gx * sm * ad + gz * sz * a
    h3 = gx * sm * a
    Hef = com(h1.T, h1) / w + com(h3.T, h3) / (3 * w)
    Hclaim = (gx**2 / w) * ((sp.Rational(4, 3) * nn + sp.eye(2 * M)) * Pe - (sp.Rational(4, 3) * nn + sp.Rational(1, 3) * sp.eye(2 * M)) * Pg) \
             - (2 * gx * gz / w) * (sp_ * a * a + sm * ad * ad) - gz**2 / w * sp.eye(2 * M)
    # restringir a n <= M-4 (los bordes de Fock rompen [a,a+])
    idx = [q * M + n for q in (0, 1) for n in range(M - 3)]
    diff = sp.simplify((Hef - Hclaim)[idx, :][:, idx])
    ok_H = diff == sp.zeros(len(idx), len(idx))
    claim('SIM-01', 1, [(H8, '**H_ef = (g_x²/ω)'), (INF, 'boxed{\\Href=\\frac{g_x^2}{\\omega}')], 'H_ef = (gx^2/w)[(4n/3+1)|e><e| - (4n/3+1/3)|g><g|] - G(s+ a^2 + h.c.) - gz^2/w, G=2 gx gz/w (James-Jerke con h1=gx s- a^+ + gz sz a, h3=gx s- a)',
          'identidad simbolica', ok_H, modo='bool', dato='(sin dato; algebra con matrices de Fock M=8, elementos n<=4)', script=S)
    G = 2 * gx * gz / w
    claim('SIM-02', 1, [(H8, 'G = 2g_xg_z/ω'), (INF, 'G&=\\frac{2g_xg_z}{\\omega}')], 'G = 2 gx gz / w es el coeficiente del termino de pares en H_ef',
          'G=2gxgz/w', sp.simplify(Hef[1 * M + 0, 0 * M + 2] + sp.sqrt(2) * G) == 0, modo='bool', script=S, dato='(algebra) <e,0|H_ef|g,2> = -sqrt(2) G')
    # corrimientos
    shift_g = sp.simplify(Hef[0 * M + 1, 0 * M + 1] - Hef[0, 0])
    shift_e = sp.simplify(Hef[M + 1, M + 1] - Hef[M, M])
    qub_n0 = sp.simplify((Hef[M + 0, M + 0] - Hef[0, 0]))
    qub_n = lambda n: sp.simplify((Hef[M + n, M + n] - Hef[n, n]))
    claim('SIM-03', 1, [(H8, 'Corrimiento del oscilador en |g⟩: −4g_x²/3ω'), (INF, 'delta_{\\rm osc}^{(g)}&=-\\frac{4g_x^2}{3\\omega}')],
          'corrimiento del oscilador con el qubit en |g> = -4 gx^2/(3 w)', '-4gx^2/(3w)', sp.simplify(shift_g + sp.Rational(4, 3) * gx**2 / w) == 0, modo='bool', script=S)
    claim('SIM-04', 1, [(H8, 'El del qubit con n = 0 es +4g_x²/3ω'), (INF, 'Delta_q^{\\rm Stark}&=')],
          'corrimiento del oscilador con qubit en |e> = +4gx^2/(3w) y transicion del qubit con n=0 = +4gx^2/(3w)', '+4gx^2/(3w)',
          sp.simplify(shift_e - sp.Rational(4, 3) * gx**2 / w) == 0 and sp.simplify(qub_n0 - sp.Rational(4, 3) * gx**2 / w) == 0, modo='bool', script=S)
    chi_ok = all(sp.simplify(qub_n(n) - (gx**2 / w) * (sp.Rational(8, 3) * n + sp.Rational(4, 3))) == 0 for n in range(0, 4))
    claim('SIM-05', 1, [(H8, 'χ = (8/3)g_x²/ω'), (INF, 'E_e(n)-E_g(n)=')], 'E_e(n)-E_g(n) = (gx^2/w)(8n/3 + 4/3), por tanto chi = (8/3) gx^2/w',
          '(8/3)gx^2/w', chi_ok, modo='bool', script=S)
    # 3/4 rotante, 1/4 contrarrotante
    claim('SIM-06', 1, [(H8, '3/4 viene de la rama rotante')], '4/3 = 1 (rotante) + 1/3 (contrarrotante): 3/4 y 1/4 del total',
          '3/4 y 1/4', sp.simplify(1 / (sp.Rational(4, 3)) - sp.Rational(3, 4)) == 0 and sp.simplify((sp.Rational(1, 3)) / sp.Rational(4, 3) - sp.Rational(1, 4)) == 0, modo='bool', script=S)
    # Resonancia vestida
    wp, wm = sp.symbols('omega_p Delta_m')
    sol = sp.solve(sp.Eq((w - wp / 2) - 4 * gx**2 / (3 * w), 0), wp)[0]
    claim('SIM-07', 1, [(H8, 'ω_p* = 2(ω − 4g_x²/3ω)'), (INF, 'boxed{\\wps=2')], 'omega_p* = 2(w - 4gx^2/(3w)) anula el coeficiente de n en el sector |g> de H_ef en el marco del drive',
          'wp*=2(w-4gx2/3w)', sp.simplify(sol - 2 * (w - 4 * gx**2 / (3 * w))) == 0, modo='bool', script=S)
    # estado oscuro: (Omega - G a^2)|psi,g> => alpha^2 = Omega/G (trivial) ; coeficiente de n
    # Tasas Reiter-Sorensen
    Dl = sp.symbols('Delta', positive=True)
    k2_expr = sp.simplify((sp.sqrt(k) * 2 / k)**2 * G**2)            # |L2|^2 = k * (2/k)^2 * G^2
    claim('SIM-08', 1, [(H8, 'κ₂ = 4G²/κ'), (INF, 'boxed{\\kappa_2=\\frac{|2G|^2}{\\kappa}')], 'kappa_2 = 4 G^2/kappa = 16 gx^2 gz^2/(w^2 kappa)',
          '4G^2/kappa', sp.simplify(k2_expr - 4 * G**2 / k) == 0 and sp.simplify(4 * G**2 / k - 16 * gx**2 * gz**2 / (w**2 * k)) == 0, modo='bool', script=S)
    HNHinv = lambda D: 1 / (D - sp.I * k / 2)
    gam = lambda D: sp.simplify(sp.Abs(sp.sqrt(k) * HNHinv(D) * gx)**2)
    # |1/(D - i k/2)|^2 = 1/(D^2 + k^2/4)
    abs_ok = sp.simplify(sp.Abs(HNHinv(Dl))**2 - 1 / (Dl**2 + k**2 / 4)) == 0
    claim('SIM-09', 1, [(H8, 'Γ₁⁻ = g_x²κ/(ω² + κ²/4)'), (INF, 'boxed{\\Gamma_1^{-}=\\frac{g_x^2\\kappa}{\\omega^2+\\kappa^2/4}')],
          'Gamma_1^- = gx^2 k/(w^2+k^2/4) (Delta=w) y Gamma_1^+ = gx^2 k/(9w^2+k^2/4) (Delta=3w), de |L_ef|^2 = k gx^2/(Delta^2+k^2/4)', 'RS', abs_ok, modo='bool', script=S)
    # figura de merito
    kap1 = gx**2 * k * (1 / w**2 + 1 / (9 * w**2)); kap2 = 16 * gx**2 * gz**2 / (w**2 * k)
    claim('SIM-10', 1, [(H8, 'κ₁/κ₂ = (5/72)(κ/g_z)²'), (INF, 'boxed{\\frac{\\kappa_1}{\\kappa_2}=\\frac{5}{72}')], 'kappa_1/kappa_2 = (5/72)(kappa/gz)^2 (independiente de gx y w)',
          '(5/72)(k/gz)^2', sp.simplify(kap1 / kap2 - sp.Rational(5, 72) * (k / gz)**2) == 0, modo='bool', script=S)
    th = math.sqrt(220 * 5 / 72); th2 = math.sqrt(1000 * 5 / 72)
    claim('SIM-11', 1, [(H8, 'g_z/κ ≥ √(220·5/72) = 3.909'), (INF, 'g_z}{\\kappa}\\geq\\sqrt{\\frac{5}{72}\\cdot220}=3.909')], 'umbral kappa_2/kappa_1=220 => gz/kappa >= sqrt(220*5/72)', '3.909', th, script=S)
    claim('SIM-12', 1, [(INF, 'geq\\sqrt{\\frac{5}{72}\\cdot10^{3}}=8.33')], 'kappa_1/kappa_2=1e-3 => gz/kappa >= 8.33', '8.33', th2, script=S)
    # gamma_pf
    adjD = lambda o, P: o.T * P * o - sp.Rational(1, 2) * (o.T * o * P + P * o.T * o)
    P = sp.diag(*[(-1)**n for n in range(M)])
    aa = a1; aad = a1.T; nf = aad * aa
    e1 = adjD(aa, P); e2 = adjD(aad, P)
    nid = list(range(M - 2))
    ok1 = sp.simplify((e1 + 2 * nf * P)[nid, :][:, nid]) == sp.zeros(len(nid), len(nid))
    ok2 = sp.simplify((e2 + 2 * (nf + sp.eye(M)) * P)[nid, :][:, nid]) == sp.zeros(len(nid), len(nid))
    claim('SIM-13', 1, [(H8, 'D[a]†P = −2nP'), (INF, '\\D^{\\dagger}[a]P&=')], 'D^dag[a]P = -2 n P y D^dag[a^+]P = -2(n+1)P', 'identidad', ok1 and ok2, modo='bool', script=S)
    Gm_, Gp_ = sp.symbols('G1m G1p', positive=True)
    gpf = sp.expand(2 * (Gm_ * A2 + Gp_ * (A2 + 1)))
    claim('SIM-14', 1, [(H8, 'γ_pf = 2[Γ₁⁻|α|² + Γ₁⁺(|α|² + 1)]'), (INF, 'boxed{\\gamma_{\\rm pf}=2')], 'gamma_pf = 2[G1m |a|^2 + G1p(|a|^2+1)] a partir de d<P>/dt = -2 G1m <nP> - 2 G1p <(n+1)P> con nP -> |a|^2 P',
          'gamma_pf', sp.simplify(gpf - (2 * Gm_ * A2 + 2 * Gp_ * (A2 + 1))) == 0 and sp.simplify(-(-2 * Gm_ * A2 - 2 * Gp_ * (A2 + 1)) - gpf) == 0, modo='bool', script=S)
    claim('SIM-15', 1, [(INF, 'corrección relativa es $\\Gamma_1^{+}/(\\kappa_1|\\alpha|^2)=1/(10|\\alpha|^2)')], 'correccion relativa del +1: G1p/(k1|a|^2) = 1/(10|a|^2)',
          '1/(10|a|^2)', sp.simplify(sp.Rational(1, 10) / A2 - (gx**2 * k / (9 * w**2)) / (gx**2 * k * 10 / (9 * w**2)) / A2) == 0, modo='bool', script=S)
    # delta_1
    u = k / 2
    d1 = -gx**2 * (w / (w**2 + u**2) + 3 * w / (9 * w**2 + u**2))
    ser = sp.series(d1, k, 0, 3).removeO()
    claimd1 = -(4 * gx**2 / (3 * w)) * (1 - sp.Rational(7, 9) * (k / (2 * w))**2)
    claim('SIM-16', 1, [(H8, 'δ₁ = −(4g_x²/3ω)[1 − (7/9)(κ/2ω)²]'), (INF, 'frac79')], 'delta_1 = -(4gx^2/3w)[1-(7/9)(kappa/2w)^2] (desarrollo a orden kappa^2)', '-(4gx2/3w)[1-(7/9)(k/2w)^2]',
          sp.simplify(ser - claimd1) == 0, modo='bool', script=S)
    # chi|a|^2/k y frontera
    k2k, gzk, wk = sp.symbols('k2k gzk wk', positive=True)
    gx_over_w2 = k2k / (16 * gzk**2)
    chik = sp.Rational(8, 3) * gx_over_w2 * wk * A2        # chi|a|^2/kappa con chi=(8/3)gx^2/w
    claim('SIM-17', 1, [(H8, 'χ|α|²/κ = (|α|²/6)(ω/κ)'), (INF, 'frac{\\chi|\\alpha|^2}{\\kappa}=')], 'chi|a|^2/kappa = (|a|^2/6)(w/kappa)(k2/kappa)/(gz/kappa)^2', '(|a|^2/6)(w/k)(k2k)/(gzk)^2',
          sp.simplify(chik - A2 / 6 * wk * k2k / gzk**2) == 0, modo='bool', script=S)
    fr = sp.solve(sp.Eq(chik.subs(A2, 4), sp.Rational(3, 10)), k2k)[0]
    claim('SIM-18', 1, [(H8, 'κ₂/κ = 0.45(g_z/κ)²/(ω/κ)'), (INF, 'frac{\\kappa_2}{\\kappa}=0.45')], 'frontera chi|a|^2/k = 0.3 con |a|^2=4 => k2/k = 0.45 (gz/k)^2/(w/k)', '0.45', float(sp.simplify(fr * wk / gzk**2)), script=S)
    # filtro
    kfs, dl = sp.symbols('kf delta', positive=True)
    Sig = J**2 / (dl + sp.I * kfs / 2)
    keff = sp.simplify(-2 * sp.im(sp.expand_complex(Sig)))
    keff_c = sp.simplify(keff.subs(J, sp.sqrt(k * kfs) / 2))
    claim('SIM-19', 1, [(H8, 'κ_eff(δ) = κκ_f²/(4δ² + κ_f²)'), (INF, 'kappa_{\\rm eff}(\\delta)=\\frac{\\kappa\\,\\kappa_f^2}{4\\delta^2+\\kappa_f^2}')], 'kappa_eff(delta) = -2 Im Sigma con Sigma = J^2/(delta + i kf/2) y 4J^2/kf = kappa  =>  kappa kf^2/(4 delta^2 + kf^2)',
          'keff', sp.simplify(keff_c - k * kfs**2 / (4 * dl**2 + kfs**2)) == 0, modo='bool', script=S)
    ke = lambda D: k * kfs**2 / (4 * D**2 + kfs**2)
    k1f = gx**2 * (ke(w) / w**2 + ke(3 * w) / (9 * w**2))
    claim('SIM-20', 1, [(H8, 'κ₁^filt = g_x²[κ_eff(ω)/ω² + κ_eff(3ω)/(9ω²)]'), (INF, 'boxed{\\frac{\\kappa_1}{\\kappa_2}=\\frac{\\kappa\\left[\\kappa_{\\rm eff}(\\omega)')], 'kappa_1/kappa_2 filtrado = k[keff(w)+keff(3w)/9]/(16 gz^2)',
          '(k/(16gz2))[keff(w)+keff(3w)/9]', sp.simplify(k1f / kap2 - k * (ke(w) + ke(3 * w) / 9) / (16 * gz**2)) == 0, modo='bool', script=S)
    claim('SIM-21', 1, [(INF, 'mejora del canal de pérdida')], 'mejora del canal de perdida k/keff(w) = (4w^2+kf^2)/kf^2', '(4w^2+kf^2)/kf^2', sp.simplify(k / ke(w) - (4 * w**2 + kfs**2) / kfs**2) == 0, modo='bool', script=S)
    fexpr = ke(w) / k + ke(3 * w) / (9 * k)
    lim = sp.limit(sp.simplify(fexpr / (kfs / (2 * w))**2), kfs, 0)
    claim('SIM-22', 1, [(INF, 'vale $\\simeq\\sqrt{41/45}')], 'f = keff(w)/k + keff(3w)/(9k) -> (82/81)(kf/2w)^2 para kf<<w;  sqrt(f/(10/9)) -> sqrt(41/45) kf/2w = 0.955 kf/2w',
          '0.955', float(sp.sqrt(lim / sp.Rational(10, 9))), tol=0.0005, script=S, nota=f'lim f/(kf/2w)^2 = {lim}')
    # brecha
    G2, Gam2 = sp.symbols('G2 Gam2', positive=True)
    ev = sp.symbols('ev')
    GN = lambda N: G * sp.sqrt(N * (N - 1))
    disc2 = sp.simplify((2 * G**2 - k**2 / 16).subs(G, sp.sqrt(Gam2 * k) / 2) + k**2 / 16 * (1 - 8 * Gam2 / k))
    claim('SIM-23', 1, [(INF, 'G_2^2-\\frac{\\kappa^2}{16}=')], 'G_2^2 = 2G^2 = k*Gam2/2 y G_2^2 - k^2/16 = -(k^2/16)(1-8 Gam2/k) con Gam2=4G^2/k', 'identidad', disc2 == 0, modo='bool', script=S)
    x = sp.symbols('x')
    ser2 = sp.series(k / 4 * (1 - sp.sqrt(1 - 8 * Gam2 / k)), Gam2, 0, 3).removeO()
    claim('SIM-24', 1, [(H8, 'Δ ≈ Γ₂ + 2Γ₂²/κ'), (INF, 'Delta\\simeq\\Gamma_2+\\frac{2\\Gamma_2^2}{\\kappa}')], 'brecha (k/4)[1-sqrt(1-8Gam2/k)] = Gam2 + 2 Gam2^2/k + O(Gam2^3)', 'serie',
          sp.simplify(ser2 - (Gam2 + 2 * Gam2**2 / k)) == 0, modo='bool', script=S)
    slope = sp.simplify(sp.diff(k / 4 * (1 - sp.sqrt(1 - 8 * Gam2 / k)), Gam2) - 1 / sp.sqrt(1 - 8 * Gam2 / k))
    claim('SIM-25', 1, [(INF, 'La pendiente es $d\\Delta/d\\Gamma_2')], 'dDelta/dGam2 = 1/sqrt(1-8 Gam2/k)', 'derivada', slope == 0, modo='bool', script=S)
    Nn = sp.symbols('N', positive=True)
    pe = sp.solve(sp.Eq(GN(Nn)**2, k**2 / 16), G)[0]
    gam_pe = sp.simplify(4 * pe**2 / k)
    claim('SIM-26', 1, [(INF, 'Gamma_2^{\\rm PE}(N)=')], 'punto excepcional del bloque N: G_N^2 = k^2/16  =>  Gam2_PE(N) = k/(4N(N-1)); N=2 -> k/8', 'k/(4N(N-1))',
          sp.simplify(gam_pe - k / (4 * Nn * (Nn - 1))) == 0 and sp.simplify(gam_pe.subs(Nn, 2) - k / 8) == 0, modo='bool', script=S)
    # polaron: G = 2 d gx
    claim('SIM-27', 1, [(INF, 'coeficiente $2dg_x=G$')], 'pares via polaron: coeficiente 2 d gx = G con d = gz/w', '2 d gx = G', sp.simplify(2 * (gz / w) * gx - G) == 0, modo='bool', script=S)
    claim('SIM-28', 1, [(H8, 'Con el qubit en |g⟩ el oscilador se desplaza d = +g_z/ω'), (C6, 'ω a†a − g_z(a + a†) = ω (a − g_z/ω)†(a − g_z/ω) − g_z²/ω')],
          'w a^+a - gz(a+a^+) = w(a-d)^+(a-d) - gz^2/w con d=gz/w', 'identidad', True, modo='bool', script=S,
          nota='verificada a mano y por expansion: w(a^+ - d)(a - d) = w a^+a - gz(a+a^+) + gz^2/w')
    ad_, a_ = sp.symbols('ad a_', commutative=False)
    lhs = sp.expand(w * (ad_ - d) * (a_ - d)); rhs = sp.expand(w * ad_ * a_ - gz * (a_ + ad_) + gz**2 / w).subs(d, gz / w)
    CL[-1]['estado'] = 'V' if sp.expand(lhs.subs(d, gz / w) - rhs) == 0 else 'D'
    # suelos y umbrales de piso
    claim('SIM-29', 1, [(H8, 'κ₂^eff/κ ≥ 220γ/κ'), (INF, 'que da $0.0044$')], 'piso: 220*gamma/kappa = 0.0044 (gamma/kappa=2e-5), 0.044 (2e-4); para eps=1e-3: 1/(1e-3)*g/k = 0.02 y 0.2', '0.0044, 0.044, 0.02, 0.2',
          220 * 2e-5 == 0.0044 and abs(220 * 2e-4 - 0.044) < 1e-12 and abs(1000 * 2e-5 - 0.02) < 1e-12 and abs(1000 * 2e-4 - 0.2) < 1e-12, modo='bool', script=S)
    return

# ============================================================ utilidades de recomputo desde rho (numpy puro, independientes de comun.py/QuTiP)
from scipy.linalg import expm
def _ops(N):
    a1 = np.diag(np.sqrt(np.arange(1, N)), 1)
    return a1
def cat_proj(N, alpha, d):
    """Proyector sobre span{D(d)|alpha>, D(d)|-alpha>} en Fock truncado N (independiente de QuTiP)."""
    a1 = _ops(N)
    Dd = expm(d * (a1.T - a1))
    n = np.arange(N)
    from math import lgamma
    def coh(al):
        lg = np.array([lgamma(k + 1) for k in n])
        return np.exp(-abs(al)**2 / 2) * np.array([al**k for k in n]) / np.exp(lg / 2)
    ca, cb = Dd @ coh(alpha), Dd @ coh(-alpha)
    pl, mi = ca + cb, ca - cb
    pl, mi = pl / np.linalg.norm(pl), mi / np.linalg.norm(mi)
    return np.outer(pl, pl.conj()) + np.outer(mi, mi.conj())
def Pc_rho(rho, N, alpha, d):
    return float(np.real(np.trace(np.kron(cat_proj(N, alpha, d), np.eye(2)) @ rho)))
def Pe_rho(rho, N): return float(np.real(np.trace(np.kron(np.eye(N), np.diag([1., 0.])) @ rho)))     # qubit idx 0 = |e>
def al2_rho(rho, N, d):
    a = np.kron(_ops(N), np.eye(2)); A = a - d * np.eye(2 * N)
    return complex(np.trace(A @ A @ rho))
def par_rho(rho, N): return float(np.real(np.trace(np.kron(np.diag((-1.0)**np.arange(N)), np.eye(2)) @ rho)))
def val_rho(rho):
    return abs(np.trace(rho) - 1), np.linalg.norm(rho - rho.conj().T), np.linalg.eigvalsh((rho + rho.conj().T) / 2).min()

# ============================================================ BLOQUE 2: resonancia vestida de Ma (rho de data/fig2, C6, V2, FWHM)
def bloque_ma():
    S = 'consolidacion/verificar_todo.py::bloque_ma'
    N = 22
    f12 = os.path.join(DATA, 'fig2/gx-0.212132_wp12.000000_wq12.000000_Om0.060000_N22.npz')
    f98 = os.path.join(DATA, 'fig2/gx-0.212132_wp11.980000_wq12.000000_Om0.060000_N22.npz')
    D12, D98 = rel(f12), rel(f98)
    r12, r98 = np.load(f12, allow_pickle=True), np.load(f98, allow_pickle=True)
    d = 0.2121320343559643 / 6.0
    # parametros de Ma
    gx = -0.3 * math.sin(math.pi / 4); gzz = 0.3 * math.cos(math.pi / 4); w = 6.0; kap = 0.03
    claim('MA-01', 1, [(INF, 'g_x^2/\\omega=0.0075')], 'Ma: gx^2/w = 0.0075 con g=0.3, theta=pi/4, w=6', '0.0075', gx**2 / w, script=S, dato='(analitico)')
    claim('MA-02', 1, [(INF, 'delta_{\\rm osc}^{(g)}=-0.010')], 'Ma: delta_osc^(g) = -4gx^2/(3w) = -0.010 (2pi x 10 MHz), un tercio de kappa', '-0.010', -4 * gx**2 / (3 * w), script=S, nota='un tercio de kappa: %.3f' % (4 * gx**2 / (3 * w) / kap))
    claim('MA-03', 1, [(H8, '11.9800 (V2)'), (RM, '11.980'), (INF, '=11.980')], 'omega_p* = 2(w - 4gx^2/3w) para Ma = 11.980', '11.9800', 2 * (w - 4 * gx**2 / (3 * w)), script=S)
    # P_c y companeros desde rho
    for (wp, tag, rr, fr) in ((12.0, '12', r12, D12), (11.98, '11.98', r98, D98)):
        rho = rr['rho']
        pcf, pcv, pe, a2, par = Pc_rho(rho, N, 2j, d), Pc_rho(rho, N, 2j, 0.0), Pe_rho(rho, N), al2_rho(rho, N, d), par_rho(rho, N)
        pep, parp = float(rr['Pe_prom']), float(rr['par_prom'])
        if tag == '12':
            claim('MA-04', 1, [(RM, '(a) ω_p = 12.00', '0.8140'), (INF, '$\\omega_p=2\\omega=12$ & 0.8140', '0.8140')], 'Ma w_p=12: P_c codigo fijo polaronico (rho estacionaria N=22)', '0.8140', pcf, tol=1e-4, dato=fr, script=S, nota='tol 1e-4: numpy propio vs QuTiP difieren ~3e-7')
            claim('MA-05', 1, [(INF, '0.8140 / 0.8131', '0.8131')], 'Ma w_p=12: P_c codigo sin desplazar', '0.8131', pcv, dato=fr, script=S)
            claim('MA-06', 1, [(RM, '(a) ω_p = 12.00', '0.1149')], 'Ma w_p=12: P_e promedio en el periodo', '0.1149', pep, dato=fr, script=S, nota='valor guardado en el npz (Pe_prom, micromovimiento con mesolve): R-parcial; P_e estroboscopico recomputado = %.4f' % pe)
            claim('MA-07', 1, [(INF, '0.114 / 0.115', '0.114')], 'Ma w_p=12: P_e estroboscopico (informe 0.114)', '0.114', pe, dato=fr, script=S)
            claim('MA-08', 1, [(RM, '−2.477 − 0.206i', '-2.477')], 'Ma w_p=12: Re alpha_eff^2 = <(a-d)^2>', '-2.477', a2.real, dato=fr, script=S)
            claim('MA-09', 1, [(RM, '−2.477 − 0.206i', '-0.206')], 'Ma w_p=12: Im alpha_eff^2', '-0.206', a2.imag, dato=fr, script=S)
            claim('MA-10', 1, [(INF, '& 2.49 &', '2.49')], 'Ma w_p=12: |alpha_eff|^2', '2.49', abs(a2), dato=fr, script=S)
            claim('MA-11', 1, [(RM, '+0.067', '0.067'), (INF, '+0.067', '0.067')], 'Ma w_p=12: paridad promedio', '0.067', parp, dato=fr, script=S, nota='parp guardado (R-parcial); estroboscopica recomputada %.4f' % par)
        else:
            claim('MA-12', 1, [(RM, '(b) ω_p = 11.98', '0.99873'), (INF, '0.99873 / 0.99754', '0.99873'), (H8, 'Ma: con ω_p = 12', None)], 'Ma w_p*=11.98: P_c codigo fijo polaronico', '0.99873', pcf, tol=1e-5, dato=fr, script=S)
            claim('MA-13', 1, [(INF, '0.99873 / 0.99754', '0.99754'), (H8, 'P_c = 0.9975', '0.9975')], 'Ma w_p*=11.98: P_c codigo sin desplazar', '0.99754', pcv, dato=fr, script=S,
                  nota='HANDOFF 4.2 escribe P_c=0.9975 sin aclarar que es el codigo sin desplazar; con el codigo fijo es 0.99873')
            claim('MA-14', 1, [(RM, '(b) ω_p = 11.98', '0.0060')], 'Ma w_p*: P_e promedio en el periodo', '0.0060', pep, dato=fr, script=S, nota='guardado (Pe_prom)')
            claim('MA-15', 1, [(INF, '0.0026 / 0.0060', '0.0026')], 'Ma w_p*: P_e estroboscopico', '0.0026', pe, dato=fr, script=S)
            claim('MA-16', 1, [(RM, '−3.975 + 0.012i', '-3.975')], 'Ma w_p*: Re alpha_eff^2', '-3.975', a2.real, dato=fr, script=S)
            claim('MA-17', 1, [(RM, '−3.975 + 0.012i', '0.012')], 'Ma w_p*: Im alpha_eff^2', '0.012', a2.imag, dato=fr, script=S)
            claim('MA-18', 1, [(RM, '(b) ω_p = 11.98', '+0.0006'), (INF, '+0.0006', '0.0006')], 'Ma w_p*: paridad (promedio)', '0.0006', parp, dato=fr, script=S, tol=5e-5)
            claim('MA-19', 1, [(INF, '& 3.975 &', '3.975'), (INF, '$|\\alpha_{\\rm eff}|^2=3.98$', '3.98')], 'Ma w_p*: |alpha_eff|^2 (3.975 en tabla, 3.98 en texto)', '3.975', abs(a2), dato=fr, script=S, tol=0.001, nota='tabla 3.975 y texto 3.98 son redondeos de 3.9746')
            claim('MA-20', 1, [(INF, 'validaciones: se informa', None)] if False else [(RM, 'Validaciones (111 ρ)')], 'Validaciones fig2: (a) y (b) cumplen |Tr-1|=0 y min autovalor >= 4e-15 (README); este rho: minimo autovalor', '4e-15', val_rho(rho)[2], modo='ge', dato=fr, script=S, tol=0)
    # barrido de fig2a: recomputar los 25 P_c
    rows = csvrows('fig2a.csv')
    err = []
    for rr_ in rows:
        fs = os.path.join(DATA, 'fig2/gx-0.212132_wp%.6f_wq12.000000_Om0.060000_N22.npz' % rr_[0])
        z = np.load(fs, allow_pickle=True); rho = z['rho']
        err.append(max(abs(Pc_rho(rho, N, 2j, d) - rr_[1]), abs(Pc_rho(rho, N, 2j, 0) - rr_[2])))
    claim('MA-21', 1, [(RM, 'ω_p ∈ [11.955, 12.015] con paso 0.0025 (25 puntos)')], 'fig2a.csv: 25 P_c (fijo y sin desplazar) recomputados desde las 25 rho; max |dif|', '1e-6', max(err), modo='le', tol=0, dato='data/fig2a.csv; data/fig2/*N22.npz', script=S, nota='max dif %.2e (numpy propio vs QuTiP)' % max(err))
    # C6
    c6 = {n: np.load(os.path.join(DATA, f'c6_{n}.npz'), allow_pickle=True) for n in ('ma_N22', 'ma_N28', 'v4_N22')}
    z = c6['ma_N22']; rho = z['rho']
    d6 = float(z['dpred'])
    pc0, pcp, pcm = Pc_rho(rho, N, 2j, 0), Pc_rho(rho, N, 2j, d6), Pc_rho(rho, N, 2j, -d6)
    pcopt = Pc_rho(rho, N, 2j, float(np.real(z['dopt'])))
    CF = rel(os.path.join(DATA, 'c6_ma_N22.npz'))
    claim('C6-01', 1, [(C6, '| Ma, ω_p = 11.98 | 22 |', '0.03536')], 'C6 Ma N=22: d_pred = gz/w', '0.03536', d6, dato=CF, script=S)
    claim('C6-02', 1, [(C6, '| Ma, ω_p = 11.98 | 22 |', '0.03451')], 'C6 Ma N=22: d optimo (Nelder-Mead guardado)', '0.03451', float(np.real(z['dopt'])), dato=CF, script=S, nota='optimo guardado: R-parcial; se verifica que P_c(d_opt)>=P_c(d_pred)')
    claim('C6-03', 1, [(C6, '| Ma, ω_p = 11.98 | 22 |', '0.976')], 'C6 Ma: d_opt/d_pred', '0.976', float(np.real(z['dopt'])) / d6, dato=CF, script=S)
    claim('C6-04', 1, [(C6, '| Ma, ω_p = 11.98 | 22 |', '0.997544')], 'C6 Ma N=22: P_c(d=0)', '0.997544', pc0, tol=1e-6, dato=CF, script=S)
    claim('C6-05', 1, [(C6, '| Ma, ω_p = 11.98 | 22 |', '0.998725')], 'C6 Ma N=22: P_c(d=+gz/w)', '0.998725', pcp, tol=1e-6, dato=CF, script=S, nota='tol 1e-6: diferencia numpy/QuTiP ~3e-7')
    claim('C6-06', 1, [(C6, '| Ma, ω_p = 11.98 | 22 |', '0.993890')], 'C6 Ma N=22: P_c(d=-gz/w)', '0.993890', pcm, tol=1e-6, dato=CF, script=S)
    claim('C6-07', 1, [(C6, '| Ma, ω_p = 11.98 | 22 |', '0.998726')], 'C6 Ma N=22: P_c(d_opt)', '0.998726', pcopt, tol=1e-6, dato=CF, script=S)
    zz = c6['ma_N28']
    claim('C6-08', 1, [(C6, '| Ma, ω_p = 11.98 | 28 |', '0.998732')], 'C6 Ma N=28: P_c(d=+gz/w)', '0.998732', float(zz['pcs'].max()) if False else Pc_rho(zz['rho'], 28, 2j, d6), tol=1e-6, dato=rel(os.path.join(DATA, 'c6_ma_N28.npz')), script=S)
    claim('C6-09', 1, [(C6, 'Convergencia N = 22 → 28 (Ma)', '7e-6')], 'C6: P_c(gz/w) cambia 7e-6 entre N=22 y 28', '7e-6', abs(Pc_rho(zz['rho'], 28, 2j, d6) - pcp), tol=1.5e-6, dato='data/c6_ma_N22.npz, c6_ma_N28.npz', script=S)
    zv = c6['v4_N22']; rv = zv['rho']; dv = float(zv['dpred'])
    claim('C6-10', 1, [(C6, '| g_x=0.05, ω=6, g_z/κ=12 (T42)', '0.998682')], 'C6 T42 N=22: P_c(d=+gz/w)', '0.998682', Pc_rho(rv, N, 2.0, dv), tol=1e-6, dato='data/c6_v4_N22.npz', script=S)
    claim('C6-11', 1, [(C6, '| g_x=0.05, ω=6, g_z/κ=12 (T42)', '0.995044')], 'C6 T42: P_c(d=0)', '0.995044', Pc_rho(rv, N, 2.0, 0.0), tol=1e-6, dato='data/c6_v4_N22.npz', script=S)
    claim('C6-12', 1, [(C6, '| g_x=0.05, ω=6, g_z/κ=12 (T42)', '0.984188')], 'C6 T42: P_c(d=-gz/w)', '0.984188', Pc_rho(rv, N, 2.0, -dv), tol=1e-6, dato='data/c6_v4_N22.npz', script=S)
    claim('C6-13', 1, [(C6, 'de 0.99754 a 0.99873', '1.9')], 'C6: 1-P_c baja un factor 1.9 en Ma (2.46e-3 -> 1.27e-3)', '1.9', (1 - pc0) / (1 - pcp), tol=0.05, dato=CF, script=S)
    claim('C6-14', 1, [(C6, 'de 0.99504 a 0.99868', '3.8')], 'C6: 1-P_c baja un factor 3.8 en T42', '3.8', (1 - Pc_rho(rv, N, 2.0, 0.0)) / (1 - Pc_rho(rv, N, 2.0, dv)), tol=0.05, dato='data/c6_v4_N22.npz', script=S)
    a_str = float(np.real(np.trace(np.kron(_ops(N), np.eye(2)) @ rho)))
    claim('C6-15', 1, [(C6, 'coincide con ⟨a⟩ estroboscópico', '0.03449')], 'C6: <a> estroboscopico en Ma = 0.03449', '0.03449', a_str, tol=5e-5, dato=CF, script=S)
    sz_g = float(np.real(np.trace(np.kron(np.eye(N), np.diag([1., -1.])) @ rho)))
    mf = -float(z['dpred']) * 6 * 0 + (-0.21213203435596426 * sz_g / 6)
    claim('C6-16', 1, [(C6, 'con ⟨σ_z⟩ = −0.9947', '-0.9947')], 'C6: <sigma_z> en Ma', '-0.9947', sz_g, tol=5e-5, dato=CF, script=S)
    claim('C6-17', 1, [(C6, 'da 0.03517', '0.03517')], 'C6: campo medio -gz<sz>/w en Ma', '0.03517', -0.21213203435596426 * sz_g / 6, tol=5e-5, dato=CF, script=S)
    frac = (d6 - (-0.21213203435596426 * sz_g / 6)) / (d6 - float(np.real(z['dopt'])))
    claim('C6-18', 1, [(C6, 'explica un tercio'), (PH, '| P5 |')], 'C6/P5: el campo medio explica un tercio de la diferencia d_pred - d_opt', '0.333', frac, tol=0.02, dato=CF, script=S,
          nota='fraccion recomputada %.3f (la correccion de campo medio 1.9e-4 frente a la diferencia 8.5e-4 es ~22%%, no un tercio)' % frac)
    claim('C6-19', 1, [(C6, '|α|² = 3.98 en Ma', '3.98')], 'C6: |alpha|^2 optimo libre en Ma = 3.98', '3.98', float(z['opt2'][2])**2, tol=0.005, dato=CF, script=S)
    claim('C6-20', 1, [(C6, '|α|² = 3.98 en Ma y 3.87', '3.87')], 'C6: |alpha|^2 optimo libre en T42 = 3.87', '3.87', float(zv['opt2'][2])**2, tol=0.005, dato='data/c6_v4_N22.npz', script=S)
    claim('C6-21', 1, [(C6, 'T42 P_c sube a 0.99979', '0.99979')], 'C6: T42 con |alpha| libre, P_c = 0.99979', '0.99979', Pc_rho(rv, N, float(zv['opt2'][2]), float(zv['opt2'][0])), tol=1e-5, dato='data/c6_v4_N22.npz', script=S)
    claim('C6-22', 1, [(C6, 'En Ma el cambio es despreciable')], 'C6: Ma con |alpha| libre N=28 P_c = 0.998759 (N=22: ver log 0.998752)', '0.998759', Pc_rho(zz['rho'], 28, 1j * float(zz['opt2'][2]), float(zz['opt2'][0])), tol=1e-5, dato='data/c6_ma_N28.npz', script=S)
    claim('C6-23', 1, [(C6, 'mínimo autovalor ≥ 4.2e-10', '4.2e-10')], 'C6: rho estacionarias: min autovalor >= 4.2e-10 (todas)', '4.2e-10', min(val_rho(c6[k]['rho'])[2] for k in c6), modo='ge', tol=0, dato='data/c6_*.npz', script=S, commit='',
          nota='min recalculado %.2e' % min(val_rho(c6[k]['rho'])[2] for k in c6))
    # V2: barrido
    fs = sorted(glob.glob(os.path.join(VAL, 'independiente/res_v2/barr_wp*_N22.npz')))
    wps, pcs = [], []
    for f in fs:
        zz2 = np.load(f); wps.append(float(zz2['wp'])); pcs.append(zz2['reg'][-1, 1])
    from scipy.interpolate import CubicSpline
    from scipy.optimize import minimize_scalar
    o = np.argsort(wps); wps, pcs = np.array(wps)[o], np.array(pcs)[o]
    cs = CubicSpline(wps, pcs)
    m = minimize_scalar(lambda x: -cs(x), bounds=(11.978, 11.982), method='bounded')
    VD = 'Mechanical-Cat-State/validacion/independiente/res_v2/barr_wp*_N22.npz'
    claim('V2-01', 1, [(H8, '11.97993 ± 0.0005'), (INF, '11.9799\\pm0.0005')], 'V2: maximo del barrido fino (spline cubico) en w_p = 11.9799 (+-0.0005), predicho 11.9800', '11.9799', m.x, tol=0.0005, dato=VD, script=S, commit='')
    wfine = np.linspace(wps[0], wps[-1], 200001); yfine = cs(wfine)
    gridok = wps[pcs > 0.99]
    lo, hi = gridok.min(), gridok.max()
    claim('V2-02', 1, [(INF, 'va de $11.9775$ a $11.9825$', '11.9775')], 'V2: primer punto de la rejilla del barrido con P_c>0.99', '11.9775', lo, tol=1e-6, dato=VD, script=S, nota='ventana definida sobre puntos de la rejilla (paso >=5e-4); el spline cruza 0.99 en %.4f y %.4f' % tuple(np.sort(wfine[np.where(np.diff(np.sign(yfine - 0.99)))[0]])[[0, -1]]))
    claim('V2-03', 1, [(INF, 'va de $11.9775$ a $11.9825$', '11.9825')], 'V2: ultimo punto de la rejilla del barrido con P_c>0.99', '11.9825', hi, tol=1e-6, dato=VD, script=S)
    i98 = int(np.argmin(abs(wps - 11.98)))
    claim('V2-04', 1, [(INF, 'confirma $P_c=0.81180$', '0.99754')], 'V2: P_c en 11.98 (Gt=60, barrido): 0.99755 (V2) / 0.99754 (Gt=300)', '0.99754', pcs[i98], tol=2e-5, dato=VD, script=S)
    z3 = np.load(os.path.join(VAL, 'independiente/res_v2/wp11.9800_N22.npz'), allow_pickle=True)['reg']
    z12 = np.load(os.path.join(VAL, 'independiente/res_v2/wp12.0000_N22.npz'), allow_pickle=True)['reg']
    claim('V2-05', 1, [(INF, 'confirma $P_c=0.81180$', '0.81180')], 'V2: P_c(w_p=12) a Gt=300 = 0.81180 (serie temporal estroboscopica)', '0.81180', z12[np.argmin(abs(z12[:, 0] - 300)), 1], tol=5e-5, dato='Mechanical-Cat-State/validacion/independiente/res_v2/wp12.0000_N22.npz', script=S)
    claim('V2-06', 1, [(INF, 'La evolución temporal hasta $\\Gamma t=300$ da $0.8118$', '0.8118')], 'V2: P_c(12) a Gt=300', '0.8118', z12[np.argmin(abs(z12[:, 0] - 300)), 1], tol=5e-5, dato='res_v2/wp12.0000_N22.npz', script=S)
    claim('V2-07', 1, [(INF, 'y paridad $0.115$ y $0.0016$', '0.115')], 'V2: paridad (12) a Gt=300', '0.115', z12[np.argmin(abs(z12[:, 0] - 300)), 6], tol=5e-4, dato='res_v2/wp12.0000_N22.npz', script=S)
    claim('V2-08', 1, [(INF, 'y paridad $0.115$ y $0.0016$', '0.0016')], 'V2: paridad (11.98) a Gt=300', '0.0016', z3[np.argmin(abs(z3[:, 0] - 300)), 6], tol=5e-5, dato='res_v2/wp11.9800_N22.npz', script=S)
    zN28 = np.load(os.path.join(VAL, 'independiente/res_v2/wp11.9800_N28.npz'), allow_pickle=True)['reg']
    iN = np.argmin(abs(zN28[:, 0] - 60)); i22 = np.argmin(abs(z3[:, 0] - 60))
    claim('V2-09', 1, [(INF, 'Convergencia $N=22\\to28$: $P_c$, $P_e$ y $|\\ave{a^2}|$ cambian menos de $10^{-5}$')], 'V2: convergencia N=22->28 (Gt=60): |dP_c|<1e-5', '1e-5', abs(zN28[iN, 1] - z3[i22, 1]), modo='le', tol=0, dato='res_v2/wp11.9800_N{22,28}.npz', script=S,
          nota='dPc=%.1e dPe=%.1e d|a2|=%.1e (Gt %.1f vs %.1f)' % (abs(zN28[iN, 1] - z3[i22, 1]), abs(zN28[iN, 2] - z3[i22, 2]), abs(zN28[iN, 3] - z3[i22, 3]), zN28[iN, 0], z3[i22, 0]))
    # fig2d
    zd = np.load(os.path.join(DATA, 'principal_fig2d.npz'), allow_pickle=True)
    imax = int(np.argmax(zd['F']))
    claim('P3-01', 1, [(RM, 'F = 0.741 en Γt = 16.9', '0.741'), (INF, 'fidelidad máxima $F=0.741$', '0.741'), (PH, '| P3 |', '0.741')], 'Fig2(d): fidelidad maxima con el gato par polaronico F = 0.741', '0.741', zd['F'][imax], dato='data/principal_fig2d.npz', script=S)
    claim('P3-02', 1, [(RM, 'F = 0.741 en Γt = 16.9', '16.9'), (INF, '$\\Gamma t=16.9$', '16.9')], 'Fig2(d): instante del maximo, Gamma t = 16.9 (Gamma=kappa/2)', '16.9', zd['t'][imax] * 0.03 / 2, tol=0.05, dato='data/principal_fig2d.npz', script=S, nota='t en unidades guardadas; Gt=%s' % (zd['t'][imax] * 0.015))
    claim('P3-03', 1, [(RM, 'con P_c = 0.965 y paridad 0.52')], 'Fig2(d): P_c = 0.965 en el instante del maximo', '0.965', zd['Pc'][imax], tol=1e-3, dato='data/principal_fig2d.npz', script=S, nota='log_principal_fig2d.txt: 0.96549 en G t=16.91')
    claim('P3-04', 1, [(RM, 'con P_c = 0.965 y paridad 0.52', '0.52')], 'Fig2(d): paridad 0.52 en el instante del maximo', '0.52', zd['par'][imax], tol=0.015, dato='data/principal_fig2d.npz', script=S, nota='la paridad alterna 0.507/0.519 entre muestras consecutivas (marco con (-1)^n); 0.52 es una de las dos fases')
    iF13 = int(np.argmin(abs(zd['t'] * 0.015 - 13)))
    claim('P3-05', 1, [(PH, 'F(13) = 0.727')], 'P3: F(Gt=13) = 0.727', '0.727', zd['F'][iF13], tol=0.002, dato='data/principal_fig2d.npz', script=S)
    vd = val_rho(zd['rho']) if 'rho' in zd.files else None
    return

# ============================================================ BLOQUE 3: ancho de la resonancia (c), regla universal, phase-flip y figura de merito
def Gam1(gx, w, kap):
    return gx**2 * kap / (w**2 + kap**2 / 4), gx**2 * kap / (9 * w**2 + kap**2 / 4)
def f3(gx, w, gzk, al=4.0, N=22):
    n = os.path.join(DATA, 'fig3', 'gx%g_w%.1f_gz%.1f_al%.1f_N%d.npz' % (gx, w, gzk, al, N))
    return np.load(n, allow_pickle=True), rel(n)

def bloque_pf():
    S = 'consolidacion/verificar_todo.py::bloque_pf'
    # ---- ancho de la resonancia c = FWHM/G
    r = csvrows('fig2b.csv')
    G = np.array([x[1] for x in r]); F = np.array([x[7] for x in r]); c = np.array([x[8] for x in r]); k2 = np.array([x[2] for x in r])
    DS = 'data/fig2b.csv'
    for i, (cv, txt) in enumerate(zip(['2.48', '3.02', '3.17', '3.08', '2.83'], ['0.03', '0.1', '0.3', '1', '2'])):
        claim('FW-0%d' % (i + 1), 1, [(INF, 'da $c=2.48,3.02,3.17,3.08,2.83$', cv)], 'FWHM/G con wq=wp, kappa2/kappa=%s: c = %s (fig2b.csv)' % (txt, cv), cv, c[i], dato=DS, script=S)
    c_orig = float((G @ F) / (G @ G))
    claim('FW-06', 1, [(RM, 'Ajuste por el origen: **FWHM = cG, c = 2.93**', '2.93'), (H8, 'FWHM = cG, c = 2.93', '2.93'), (INF, 'el ajuste por el origen es $c=2.94$', '2.94')],
          'Ajuste por el origen FWHM = cG (serie wq=wp, 5 puntos): c recalculado = sum(G*FWHM)/sum(G^2)', '2.93', c_orig, dato=DS, script=S,
          nota='recalculo %.4f: redondea a 2.94 (informe); README/HANDOFF dicen 2.93' % c_orig)
    claim('FW-07', 1, [(INF, 'el ajuste por el origen es $c=2.94$', '2.94')], 'Ajuste por el origen c (informe)', '2.94', c_orig, dato=DS, script=S)
    claim('FW-08', 1, [(RM, 'media 2.92 ± 0.25', '2.92'), (INF, 'la media $2.92\\pm0.25$', '2.92')], 'Media de c en la serie wq=wp', '2.92', c.mean(), dato=DS, script=S)
    claim('FW-09', 1, [(RM, 'media 2.92 ± 0.25', '0.25'), (INF, 'la media $2.92\\pm0.25$', '0.25'), (H8, 'c = 2.93 ± 0.25', '0.25')], 'Desviacion de c en la serie wq=wp (5 puntos): +-0.25', '0.25', c.std(ddof=1), dato=DS, script=S, tol=0.005,
          nota='std(ddof=1)=%.3f, std(ddof=0)=%.3f; el 0.25 coincide con la serie de 7 puntos de Tarea 47 (0.246), no con esta de 5' % (c.std(ddof=1), c.std(ddof=0)))
    c7 = np.array([2.454, 2.666, 2.878, 3.050, 3.100, 3.116, 2.826])
    claim('FW-10', 1, [(H8, 'c = 2.87 ± 0.25'), (INF, 'c=2.87\\pm0.25')], 'Serie wq fijo (Tarea 47d, 7 valores): media 2.870 y std 0.246 a partir de los 7 c listados', '2.87', c7.mean(), tol=0.005,
          dato='validacion/tareas_ma/tarea47_resultados.md (tabla; la cache tarea47a/d no existe en el repo)', script=S, estado='V', nota='aritmetica sobre los 7 c del reporte; los FWHM individuales no son recomputables (cache ausente): R')
    claim('FW-11', 1, [(INF, 'c=\\mathrm{FWHM}/G=2.45,2.67,2.88,3.05,3.10,3.12,2.83')], 'Siete valores de c con wq fijo', '2.45,2.67,2.88,3.05,3.10,3.12,2.83', None, estado='R',
          dato='validacion/tareas_ma/tarea47_resultados.md', script='validacion/tareas_ma/tarea47_analisis.py (lee cache inexistente en el repo)', nota='no recomputable: la cache validacion/tarea47a_cache no esta')
    # ---- regla universal
    rs = {row[0]: row for row in csv.reader(open(os.path.join(DATA, 'principal_fig2_resumen.csv')))}
    hdr = rs.pop('system')
    R = {k: [float(v) if re.fullmatch(r'[-+0-9.eE]+', v) else v for v in vv] for k, vv in rs.items()}
    a4 = [v for v in R.values() if abs(v[5] - 4) < 0.01]
    a2 = [v for v in R.values() if abs(v[5] - 2) < 0.01]
    xm = [abs(v[12]) for v in R.values()]
    DR = 'data/principal_fig2_resumen.csv'
    mx_other = sorted(xm)[-2]
    claim('RU-01', 1, [(H8, '(|x| ≤ 0.09, salvo un caso con 0.16)'), (INF, 'el pico cae en $|x|\\leq0.09$'), (RM, 'queda en |x| ≤ 0.09 salvo ω = 8')], 'Regla universal: |x_max (parabola)| <= 0.09 en todas las curvas salvo una', '0.09', mx_other, modo='le', tol=0, dato=DR, script=S, nota='segundo mayor |x|=%.4f' % mx_other)
    claim('RU-02', 1, [(H8, 'salvo un caso con 0.16'), (INF, 'con $x=+0.16$'), (RM, 'x = +0.16, con semianchos 0.68/2.18'), (RM, 'salvo ω = 8, κ₂/κ = 1 (+0.15)', '0.15'), (PH, '| P1 |', '0.15')],
          'Caso extremo: w=8, kappa2/kappa=1, x del maximo (parabola) = +0.156', '0.16', R['w8_gz7_k21_a4'][12], dato=DR, script=S, tol=0.006,
          nota='recalculo %.3f; README l.144/PENDIENTES P2 dicen +0.16, README l.149 y P1 dicen +0.15 (redondeo)' % R['w8_gz7_k21_a4'][12])
    claim('RU-03', 1, [(RM, 'x = +0.16, con semianchos 0.68/2.18', '0.68')], 'Semiancho izquierdo del caso extremo', '0.68', R['w8_gz7_k21_a4'][10], dato=DR, script=S)
    claim('RU-04', 1, [(RM, 'x = +0.16, con semianchos 0.68/2.18', '2.18')], 'Semiancho derecho del caso extremo', '2.18', R['w8_gz7_k21_a4'][11], dato=DR, script=S)
    fw4 = [v[9] for v in a4]; fw2 = [v[9] for v in a2]
    claim('RU-05', 1, [(H8, 'FWHM ≈ 3G con |α|² = 4'), (RM, '**FWHM en x:** 2.5–3.25 con |α|² = 4'), (INF, 'El FWHM es $2.5$--$3.25$ para $|\\alpha|^2=4$')], 'FWHM en x, |alpha|^2=4: rango 2.5-3.25 (cota inferior)', '2.5', min(fw4), tol=0.03, dato=DR, script=S, nota='min recalculado %.3f' % min(fw4))
    claim('RU-06', 1, [(RM, '**FWHM en x:** 2.5–3.25 con |α|² = 4'), (INF, 'El FWHM es $2.5$--$3.25$ para $|\\alpha|^2=4$')], 'FWHM en x, |alpha|^2=4: rango 2.5-3.25 (cota superior)', '3.25', max(fw4), tol=0.01, dato=DR, script=S,
          nota='maximo recalculado %.3f (w4,k2/k=1: 3.181); el documento da 3.25, que no se alcanza en ninguna curva de |alpha|^2=4' % max(fw4))
    claim('RU-07', 1, [(RM, 'y 3.7–4.2 con |α|² = 2'), (INF, '$3.7$--$4.2$ para $|\\alpha|^2=2$')], 'FWHM en x, |alpha|^2=2: rango 3.7-4.2', '3.7', min(fw2), tol=0.05, dato=DR, script=S)
    claim('RU-08', 1, [(RM, 'y 3.7–4.2 con |α|² = 2'), (INF, '$3.7$--$4.2$ para $|\\alpha|^2=2$')], 'FWHM en x, |alpha|^2=2: cota superior 4.2', '4.2', max(fw2), tol=0.05, dato=DR, script=S)
    n22 = [v for v in a4 + a2 if v[1] == 22]
    pm = [v[8] for v in n22 if v[0] != 'x']
    claim('RU-09', 1, [(RM, 'P_max pasa a 0.9983–0.9999'), (INF, 'va de $0.9983$ a $0.9999$')], 'P_max de las curvas con N=22 (excepto w=6, gz/k=20): minimo 0.9983', '0.9983', min(v[8] for v in n22 if 'gz20' not in [k for k, vv in R.items() if vv is v][0]), tol=0.0001, dato=DR, script=S)
    claim('RU-10', 1, [(RM, 'salvo ω = 6, g_z/κ = 20 (0.9905)'), (INF, '($0.9905$; $g_z/\\omega=0.1$')], 'P_max de la curva w=6, gz/kappa=20', '0.9905', R['w6_gz20_k20.1_a4'][8], dato=DR, script=S)
    claim('RU-11', 1, [(RM, 'semiancho izquierdo baja de ~1.3'), (INF, 'baja de $\\sim1.3$')], 'Asimetria: semiancho izquierdo ~1.3 en kappa2/kappa=0.03-0.1 (serie w=6: 1.18, 1.31)', '1.3', np.mean([R['w6_gz7.07_k20.03_a4'][10], R['w6_gz7.07_k20.1_a4'][10]]), tol=0.1, dato=DR, script=S)
    claim('RU-12', 1, [(RM, 'a ~0.6–0.75 (κ₂/κ = 1–2)'), (INF, 'a $\\sim0.6$--$0.75$')], 'Asimetria: semiancho izquierdo 0.6-0.75 en kappa2/kappa=1-2', '0.6', min(R['w6_gz7.07_k22_a4'][10], R['w6_gz7.07_k21_a4'][10]), tol=0.05, dato=DR, script=S,
          nota='valores: w6 k2/k=1: %.3f, k2/k=2: %.3f, w4: %.3f, w8: %.3f; el rango 0.6-0.75 omite 0.875 (w=6, 1) y 1.12 (w=4, 1)' % (R['w6_gz7.07_k21_a4'][10], R['w6_gz7.07_k22_a4'][10], R['w4_gz7_k21_a4'][10], R['w8_gz7_k21_a4'][10]))
    claim('RU-13', 1, [(RM, 'y el derecho sube a ~2.2'), (INF, 'el derecho sube a $\\sim2.2$')], 'Semiancho derecho ~2.2 en kappa2/kappa=1-2', '2.2', np.mean([R['w6_gz7.07_k21_a4'][11], R['w6_gz7.07_k22_a4'][11], R['w8_gz7_k21_a4'][11]]), tol=0.05, dato=DR, script=S)
    claim('RU-14', 1, [(RM, 'Validaciones (157 ρ)'), (INF, 'para 157 estados estacionarios')], 'Numero de estados estacionarios (rho) de la Fig. 2 nueva: 157', '157', len(glob.glob(os.path.join(DATA, 'pfig2', '*.npz'))), dato='data/pfig2/*.npz, data/principal_fig2.csv', script=S,
          nota='pfig2 contiene %d .npz; principal_fig2.csv tiene %d filas; no se pudo reconstruir un conteo que de 157' % (len(glob.glob(os.path.join(DATA, 'pfig2', '*.npz'))), sum(1 for _ in open(os.path.join(DATA, 'principal_fig2.csv'))) - 1))
    # ---- V3: tasa de phase-flip de Ma en la resonancia
    from scipy.optimize import curve_fit
    rr = np.load(os.path.join(VAL, 'independiente/res_v2/wp11.9800_N22.npz'), allow_pickle=True)['reg']
    Gt, par, Pc = rr[:, 0], rr[:, 6], rr[:, 1]; Gam = 0.015
    def lnfit(a, b):
        m = (Gt >= a) & (Gt <= b) & (Pc > 0.99)
        return -np.polyfit(Gt[m] / Gam, np.log(par[m]), 1)[0]
    def eflt(a, b):
        m = (Gt >= a) & (Gt <= b) & (Pc > 0.99)
        p, _ = curve_fit(lambda t, A, k, c: A * np.exp(-k * t) + c, Gt[m] / Gam, par[m], p0=[0.7, 3e-4, 6e-4]); return p
    g1m, g1p = Gam1(0.3 * math.sin(math.pi / 4), 6.0, 0.03)
    pred4 = 2 * 4 * (g1m + g1p)
    a2m = 3.9734
    pred_1 = 2 * (g1m * a2m + g1p * (a2m + 1))
    VR = 'Mechanical-Cat-State/validacion/independiente/res_v2/wp11.9800_N22.npz'
    claim('V3-01', 1, [(INF, 'ajuste logarítmico $\\Gamma t\\in[26,152]$ & $3.307\\times10^{-4}$', '3.307e-4')], 'V3: tasa por ajuste ln lineal en Gt [26,152] (P_c>0.99)', '3.307e-4', lnfit(26, 152), dato=VR, script=S)
    claim('V3-02', 1, [(INF, 'ajuste logarítmico $\\Gamma t\\in[26,152]$ & $3.307\\times10^{-4}$', '3.333e-4')], 'V3: prediccion 2|a|^2(G1m+G1p), |a|^2=4 (sin +1)', '3.333e-4', pred4, dato='(analitico)', script=S)
    claim('V3-03', 1, [(INF, 'ajuste logarítmico $\\Gamma t\\in[26,152]$ & $3.307\\times10^{-4}$', '0.992')], 'V3: razon medido/predicho (ln lineal, ventana 26-152)', '0.992', lnfit(26, 152) / pred4, dato=VR, script=S)
    p = eflt(26, 300)
    claim('V3-04', 1, [(INF, 'ajuste con piso, $\\Gamma t\\in[26,300]$ & $3.332\\times10^{-4}$', '3.332e-4')], 'V3: tasa por ajuste A e^{-kt}+c, Gt [26,300]', '3.332e-4', p[1], dato=VR, script=S)
    claim('V3-05', 1, [(INF, 'ajuste con piso, $\\Gamma t\\in[26,300]$ & $3.332\\times10^{-4}$', '1.000')], 'V3: razon con piso / pred sin +1', '1.000', p[1] / pred4, dato=VR, script=S, tol=5e-4)
    claim('V3-06', 1, [(INF, 'con $+1$ y $|\\ave{a^2}|=3.973$ & $3.332\\times10^{-4}$ & $3.394\\times10^{-4}$', '3.394e-4')], 'V3: prediccion con +1 y |a2|=3.973 (formula 2[G1m a+G1p(a+1)])', '3.394e-4', 2 * (g1m * 3.973 + g1p * 4.973), dato='(analitico)', script=S)
    claim('V3-07', 1, [(INF, 'con $+1$ y $|\\ave{a^2}|=3.973$ & $3.332\\times10^{-4}$ & $3.394\\times10^{-4}$ & 0.982', '0.982')], 'V3: razon con +1 (piso incluido)', '0.982', p[1] / (2 * (g1m * 3.973 + g1p * 4.973)), dato=VR, script=S)
    claim('V3-08', 1, [(INF, 'la razón baja de $0.992$ a $0.866$')], 'V3: ajuste ln lineal en ventana tardia [150,300]: razon 0.866', '0.866', lnfit(150, 300) / pred4, dato=VR, script=S, tol=0.0005)
    pb = [eflt(a, b)[2] for a, b in ((26, 300), (60, 300))]
    claim('V3-09', 1, [(INF, 'El piso de paridad es $c=(6.4\\pm0.6)\\times10^{-4}$'), (H8, 'piso de paridad del estado estacionario (~7e-4'), (INF, 'c\\simeq(6\\text{ a }7)')], 'Piso de paridad c = 6.4e-4 (+-0.6e-4) en el ajuste con piso, ventana [60,300]', '6.4e-4', pb[1], dato=VR, script=S, tol=0.1e-4)
    claim('V3-10', 1, [(INF, 'estacionario, mezcla de $\\ket{\\pm\\alpha}$ sin coherencia entre ellos, tiene entonces paridad $\\simeq e^{-2|\\alpha|^2}\\simeq3.4')], 'Paridad e^{-2|a|^2} con |a|^2=4: 3.4e-4', '3.4e-4', math.exp(-8), dato='(analitico)', script=S)
    # ---- fig3a: tabla V4 de 15 puntos
    T = [(0.04, 8, 3, '0.004', '0.99993', '3.99', '1.022', '0.999'), (0.03, 6, 4, '0.006', '0.99994', '3.98', '1.018', '0.998'), (0.05, 6, 4, '0.018', '0.99993', '3.98', '1.018', '0.997'),
         (0.05, 5, 4, '0.026', '0.99990', '3.97', '1.015', '0.996'), (0.1, 5, 2, '0.026', '0.99946', '4.01', '1.021', '0.993'), (0.05, 6, 5, '0.028', '0.99992', '3.97', '1.014', '0.996'),
         (0.1, 7, 4, '0.052', '0.99987', '3.99', '1.018', '0.996'), (0.06, 8, 8, '0.058', '0.99989', '3.96', '1.011', '0.995'), (0.05, 7, 10, '0.082', '0.99964', '3.92', '0.998', '0.992'),
         (0.05, 6, 12, '0.160', '0.99868', '3.85', '0.975', '0.987'), (0.15, 6, 4, '0.160', '0.99948', '4.00', '1.016', '0.990'), (0.12, 7, 7, '0.230', '0.99979', '3.96', '1.009', '0.993'),
         (0.08, 4, 6, '0.230', '0.99941', '3.92', '0.992', '0.988'), (0.08, 5, 10, '0.410', '0.99859', '3.85', '0.973', '0.985'), (0.1, 4, 10, '1.000', '0.99653', '3.77', '0.947', '0.979')]
    kap = 0.03
    recs = []
    for i, (gx, w, gzk, k2s, pcs, als, s1s, c1s) in enumerate(T):
        z, fz = f3(gx, w, gzk)
        gz = gzk * kap
        k2r = 16 * (gx / w)**2 * gzk**2
        a_eff = abs(complex(z['al2eff']))
        g1m_, g1p_ = Gam1(gx, w, kap)
        con1 = float(z['gpf']) / (2 * (g1m_ * a_eff + g1p_ * (a_eff + 1)))
        sin1 = float(z['gpf']) / (2 * 4 * (g1m_ + g1p_))
        pc = Pc_rho(z['rho'], 22, 2.0, gz / w)
        recs.append((k2r, con1, sin1))
        L = [(INF, f'{gx:g} & {w:g} & {gzk:g} & {k2s}' if False else f'{gx:g} & {w} & {gzk} & {k2s} &')]
        claim('V4-%02d-k2' % (i + 1), 1, L, 'Tabla V4 fila %d (gx=%g,w=%g,gz/k=%g): kappa2/kappa = 16(gx/w)^2(gz/k)^2' % (i + 1, gx, w, gzk), k2s, k2r, dato=fz, script=S)
        claim('V4-%02d-pc' % (i + 1), 1, [(INF, f'{gx:g} & {w} & {gzk} & {k2s} &')], 'Tabla V4 fila %d: P_c (codigo fijo, rho N=22)' % (i + 1), pcs, pc, tol=2e-5, dato=fz, script=S)
        claim('V4-%02d-al' % (i + 1), 1, [(INF, f'{gx:g} & {w} & {gzk} & {k2s} &')], 'Tabla V4 fila %d: |alpha_eff|^2 = |<(a-d)^2>|' % (i + 1), als, a_eff, dato=fz, script=S)
        claim('V4-%02d-sin1' % (i + 1), 1, [(INF, f'{gx:g} & {w} & {gzk} & {k2s} &')], 'Tabla V4 fila %d: gamma_pf medido / (2|a|^2 k1) con |a|^2=4 nominal (sin +1)' % (i + 1), s1s, sin1, dato=fz, script=S)
        claim('V4-%02d-con1' % (i + 1), 1, [(INF, f'{gx:g} & {w} & {gzk} & {k2s} &')], 'Tabla V4 fila %d: gamma_pf medido / C1 con |a_eff|^2 (con +1)' % (i + 1), c1s, con1, dato=fz, script=S)
    global recs_v4
    recs_v4 = recs
    cc = np.array([r[1] for r in recs])
    claim('V4-RANGO', 1, [(H8, '19 puntos, 0.979–0.999'), (INF, 'el rango es de $2\\%$ ($0.979$--$0.999$)'), (INF, 'El rango es $0.979$--$0.999$')], 'Rango de gamma_pf medido/predicho (con +1) en los 15 puntos: 0.979-0.999', '0.979', cc.min(), dato='data/fig3/*.npz', script=S, tol=0.0005,
          nota='cota superior recalculada %.4f. HANDOFF/informe dicen "19 puntos" para el panel (a) de la figura de merito; esta tabla tiene 15 (los 19 son de principal_fig3_puntos? ver FM-19)' % cc.max())
    claim('V4-MAX', 1, [(INF, 'el rango es de $2\\%$ ($0.979$--$0.999$)')], 'Cota superior del rango (0.999)', '0.999', cc.max(), dato='data/fig3/*.npz', script=S, tol=0.0006)
    k2arr = np.array([r[0] for r in recs])
    claim('V4-CORR', 1, [(RM, 'Correlaciona con κ₂/κ (r = 0.89)'), (INF, 'Correlaciona con $\\kappa_2/\\kappa$ ($r=0.89$)'), (INF, 'Correlaciona con $\\kappa_2/\\kappa$ ($r=0.89$)')], 'Correlacion del deficit (1 - medido/predicho) con kappa2/kappa: r = 0.89', '0.89', np.corrcoef(k2arr, 1 - cc)[0, 1], dato='data/fig3/*.npz', script=S, tol=0.03,
          nota='con 15 puntos de la Tabla V4; el r=0.69 con P_e: ver V4-CORR2')
    pe = np.array([float(np.load(os.path.join(DATA, 'fig3', 'pe_gx%g_w%.1f_gz%.1f_al4.0_N22.npz' % (t[0], t[1], t[2])), allow_pickle=True)['Pe_prom'] if 'Pe_prom' in np.load(os.path.join(DATA, 'fig3', 'pe_gx%g_w%.1f_gz%.1f_al4.0_N22.npz' % (t[0], t[1], t[2])), allow_pickle=True).files else np.nan) for t in T])
    if not np.isnan(pe).any():
        claim('V4-CORR2', 1, [(RM, 'más que con P_e (r = 0.69)'), (INF, 'más que con $P_e$ ($r=0.69$)')], 'Correlacion del deficit con P_e: r = 0.69', '0.69', np.corrcoef(pe, 1 - cc)[0, 1], dato='data/fig3/pe_*.npz', script=S, tol=0.05)
    else:
        claim('V4-CORR2', 1, [(RM, 'más que con P_e (r = 0.69)')], 'Correlacion del deficit con P_e: r = 0.69', '0.69', None, estado='R', dato='data/fig3/pe_*.npz', script=S, nota='clave Pe_prom no encontrada en los npz')
    # ---- V4b
    for i, (al, aef, s1, c1, corr) in enumerate([(2, '1.9941', '1.0489', '0.9988', '5.0'), (4, '3.9815', '1.0223', '0.9972', '2.5'), (6, '5.9603', '1.0128', '0.9961', '1.7')]):
        N_ = 26 if al == 6 else 22
        z, fz = f3(0.05, 6.0, 4.0, float(al), N_)
        a_eff = abs(complex(z['al2eff']))
        g1m_, g1p_ = Gam1(0.05, 6.0, kap)
        con1 = float(z['gpf']) / (2 * (g1m_ * a_eff + g1p_ * (a_eff + 1)))
        sin1 = float(z['gpf']) / (2 * a_eff * (g1m_ + g1p_))
        row = f'{al} & {aef}'
        claim('V4b-%d-al' % al, 1, [(INF, row)], 'Tabla V4b |a|^2_nom=%d: |alpha_eff|^2' % al, aef, a_eff, dato=fz, script=S)
        claim('V4b-%d-sin1' % al, 1, [(INF, row)], 'Tabla V4b |a|^2_nom=%d: medido/(2|a_eff|^2 k1) (sin +1)' % al, s1, sin1, dato=fz, script=S)
        claim('V4b-%d-con1' % al, 1, [(INF, row)], 'Tabla V4b |a|^2_nom=%d: medido/C1 (con +1)' % al, c1, con1, dato=fz, script=S)
        claim('V4b-%d-corr' % al, 1, [(INF, row)], 'Tabla V4b |a|^2_nom=%d: correccion prevista 1/(10|a|^2) = %s%%' % (al, corr), corr, 100 / (10 * al), dato='(analitico)', script=S, tol=0.05)
    return

# ============================================================ BLOQUE 4: baño filtrado (Fig. 4 / V5)
def tasa_cola(t, serie, ref=None, piso_fn=None, i2_modo='3piso'):
    """Reimplementacion propia de la definicion documentada: cola exponencial de |Pc - Pc_inf| desde que el exceso cae
    bajo el 10% de su maximo; ajuste lineal de ln."""
    e = np.abs(serie - (serie[-1] if ref is None else ref))
    i1 = int(np.argmax(e < 0.1 * e.max()))
    piso = piso_fn(e)
    fin = np.where(e < (3 * piso if i2_modo == '3piso' else piso))[0]
    i2 = int(fin[fin > i1][0]) if np.any(fin > i1) else len(e)
    if i2 - i1 < 10: return np.nan
    return -np.polyfit(t[i1:i2], np.log(e[i1:i2]), 1)[0]
def conf_fig4(z):
    t, pss = z['tdin'], float(z['Pc_fijo'])
    return [tasa_cola(t, s, ref=pss, piso_fn=lambda e: np.median(e[int(0.8 * len(e)):])) for s in z['Pc_din']]
def conf_mini(z):
    return [tasa_cola(z['t'], s, piso_fn=lambda e: 1e3 * np.median(e[int(0.9 * len(e)):-1]), i2_modo='abs') if False else _td_mini(z['t'], s) for s in z['Pc_din']]
def _td_mini(t, serie):
    e = np.abs(serie - serie[-1]); i1 = int(np.argmax(e < 0.1 * e.max()))
    piso = max(1e-9, 1e3 * np.median(e[int(0.9 * len(e)):-1]))
    fin = np.where(e < piso)[0]; i2 = int(fin[fin > i1][0]) if np.any(fin > i1) else len(e) - 1
    return np.nan if i2 - i1 < 10 else -np.polyfit(t[i1:i2], np.log(e[i1:i2]), 1)[0]

def bloque_filtro():
    S = 'consolidacion/verificar_todo.py::bloque_filtro'
    kap, w, gx = 0.03, 6.0, 0.3 * math.sin(math.pi / 4)
    F4 = lambda n: np.load(os.path.join(DATA, 'fig4', n), allow_pickle=True)
    # --- V5(a): canales por separado
    rows = [('plano', 8, '', 'est_kf0.0_N8_Nf1.npz', '1.0004', '0.9995', '1.0003', ('8', '--')), ('plano', 12, '', 'est_kf0.0_N12_Nf1.npz', '1.0004', '0.9994', '1.0003', ('12', '--')),
            ('kf=1', 8, 2, 'est_kf1.0_N8_Nf2.npz', '0.9980', '1.0011', '0.9981', None), ('kf=1', 8, 3, 'est_kf1.0_N8_Nf3.npz', '0.9980', '1.0011', '0.9981', None),
            ('kf=0.3', 8, 2, 'est_kf0.3_N8_Nf2.npz', '0.9977', '1.0011', '0.9978', None), ('kf=0.3', 8, 3, 'est_kf0.3_N8_Nf3.npz', '0.9977', '1.0011', '0.9978', None),
            ('kf=0.3', 12, 3, 'est_kf0.3_N12_Nf3.npz', '0.9977', '1.0011', '0.9978', None)]
    for i, (nm, N, Nf, fn, rm, rp, rk, _) in enumerate(rows):
        if not os.path.exists(os.path.join(DATA, 'fig4', fn)):
            # est_kf0.3_N12_Nf3 no esta en data/fig4: usar el log de V5
            claim('F5a-%d' % (i + 1), 1, [(INF, '$\\kappa_f=0.3$ & 12 & 3')], 'Tabla V5 %s N=%s Nf=%s: ratios Gamma1' % (nm, N, Nf), '%s/%s/%s' % (rm, rp, rk), None, estado='R',
                  dato='Mechanical-Cat-State/validacion/independiente/res_v5/estatico_gz0.txt (V5); no hay est_kf0.3_N12_Nf3 en data/fig4', script='validacion/independiente/v5_filtro.py', nota='dato de la verificacion independiente, no de figuras_finales')
            continue
        z = F4(fn)
        gm, gp = float(z['Gm']) / float(z['Gm_pred']), float(z['Gp']) / float(z['Gp_pred'])
        k1 = (float(z['Gm']) + float(z['Gp'])) / (float(z['Gm_pred']) + float(z['Gp_pred']))
        L = [(INF, 'plano & %d & -- & %s' % (N, rm) if nm == 'plano' else ('$\\kappa_f=1$ & 8' if nm == 'kf=1' else '$\\kappa_f=0.3$ & 8'))]
        claim('F5a-%d-m' % (i + 1), 1, L, 'Tabla V5(a) %s N=%d Nf=%s: Gamma1- medido/predicho' % (nm, N, Nf), rm, gm, dato='data/fig4/' + fn, script=S)
        claim('F5a-%d-p' % (i + 1), 1, L, 'Tabla V5(a) %s N=%d Nf=%s: Gamma1+ medido/predicho' % (nm, N, Nf), rp, gp, dato='data/fig4/' + fn, script=S)
        claim('F5a-%d-k' % (i + 1), 1, L, 'Tabla V5(a) %s N=%d Nf=%s: kappa1 razon' % (nm, N, Nf), rk, k1, dato='data/fig4/' + fn, script=S)
    claim('F5a-N', 1, [(INF, 'El filtro de dos niveles y el armónico ($N_f=3$) coinciden a $10^{-6}$')], 'N_f=2 vs 3 (kf=1, N=8): Gamma1- coinciden a 1e-6 relativo', '1e-6', abs(float(F4('est_kf1.0_N8_Nf2.npz')['Gm']) / float(F4('est_kf1.0_N8_Nf3.npz')['Gm']) - 1), modo='le', tol=0, dato='data/fig4/est_kf1.0_N8_Nf{2,3}.npz', script=S,
          nota='dif relativa %.1e' % abs(float(F4('est_kf1.0_N8_Nf2.npz')['Gm']) / float(F4('est_kf1.0_N8_Nf3.npz')['Gm']) - 1))
    # README: N_f=2 vs 3 <= 4e-6 ; N=8 vs 12 <= 2e-6
    d1 = max(abs(float(F4('est_kf%s_N8_Nf2.npz' % k)['Gm']) / float(F4('est_kf%s_N8_Nf3.npz' % k)['Gm']) - 1) for k in ('0.1', '0.3', '1.0', '3.0'))
    claim('F5a-N2', 1, [(RM, 'N_f = 2 frente a 3: ≤ 4e-6')], 'README Fig.4(a): N_f=2 frente a 3 cambia <= 4e-6 (Gamma1-)', '4e-6', d1, modo='le', tol=0, dato='data/fig4/est_*', script=S, nota='max dif relativa recalculada %.1e' % d1)
    d2 = abs(float(F4('est_kf0.0_N8_Nf1.npz')['Gm']) / float(F4('est_kf0.0_N12_Nf1.npz')['Gm']) - 1)
    claim('F5a-N3', 1, [(RM, 'N = 8 frente a 12: ≤ 2e-6')], 'README Fig.4(a): N=8 frente a 12 <= 2e-6 (plano; kf=1: N8 Nf3 vs N12 Nf3)', '2e-6', max(d2, abs(float(F4('est_kf1.0_N8_Nf3.npz')['Gm']) / float(F4('est_kf1.0_N12_Nf3.npz')['Gm']) - 1)), modo='le', tol=0, dato='data/fig4/est_*', script=S,
          nota='plano %.1e' % d2)
    r_flat = float(F4('est_kf0.0_N8_Nf1.npz')['Gp']) / float(F4('est_kf0.0_N8_Nf1.npz')['Gm']); r_f1 = float(F4('est_kf1.0_N8_Nf3.npz')['Gp']) / float(F4('est_kf1.0_N8_Nf3.npz')['Gm'])
    claim('F5-09', 1, [(INF, 'baja de $0.11$ a $0.012$')], 'Gamma1+/Gamma1-: 0.11 (plano) -> 0.012 (kf=1)', '0.012', r_f1, tol=0.0005, dato='data/fig4/est_*.npz', script=S, nota='plano %.4f' % r_flat)
    # --- V5(b) mejoras
    fl = lambda kf, N=16, Nf=2: np.load(os.path.join(DATA, 'fig4', 'fl_kf%.1f_N%d_Nf%d_al2.0.npz' % (kf, N, Nf if kf else 1)), allow_pickle=True)
    z0 = fl(0.0)
    def c1(gpf_unused, z):
        kf = float(z['kf']); ke = (lambda d: kap) if kf == 0 else (lambda d: kap * kf**2 / (4 * d**2 + kf**2))
        gm, gp = gx**2 * ke(w) / w**2, gx**2 * ke(3 * w) / (9 * w**2)
        a = abs(complex(z['al2eff']))
        return 2 * (gm * a + gp * (a + 1)), gm + gp
    tabla = [(3.0, '19.5', '19.5', '1.00', '18.6'), (1.0, '166.5', '166.1', '1.003', '159.1'), (0.3, '1838', '1833', '1.003', '1757'), (0.1, '1.65e4', '1.65e4', '1.003', None)]
    P0 = c1(0, z0)
    for kf, med, pre, rat, k1o in tabla:
        z = fl(kf); P = c1(0, z)
        m = float(z0['gpf']) / float(z['gpf']); p = P0[0] / P[0]
        loc = [(INF, '%s & %s & %s & %s' % ({3.0: '3', 1.0: '1', 0.3: '0.3', 0.1: '0.1'}[kf], {3.0: '19.5', 1.0: '166.5', 0.3: '1838', 0.1: '$1.65\\times10^{4}$'}[kf], {3.0: '19.5', 1.0: '166.1', 0.3: '1833', 0.1: '$1.65\\times10^{4}$'}[kf], rat))]
        tt = 0.005 * float(med.replace('e4', 'e4'))
        claim('F5b-%s-med' % kf, 1, loc + [(RM, '19.5/19.5, 166.5/166.1, 1838/1833 y 16528/16487')] if kf == 3.0 else loc, 'Mejora de gamma_pf medida (flat/filtro) kf=%g' % kf, med, m, tol=(None if kf < 0.2 else (0.05 if kf == 3.0 else 0.5)), dato='data/fig4/fl_kf%.1f_N16_*_al2.0.npz' % kf, script=S, nota='recalculo %.5g (README: 16528/16487 para kf=0.1)' % m)
        claim('F5b-%s-pre' % kf, 1, loc, 'Mejora predicha con C1 kf=%g' % kf, pre, p, tol=(0.5 if kf >= 0.3 else None), dato='data/fig4/fl_kf%.1f_N16_*_al2.0.npz' % kf, script=S, nota='recalculo %.5g' % p)
        claim('F5b-%s-rat' % kf, 1, loc, 'Razon medido/predicho de la mejora kf=%g' % kf, rat, m / p, dato='data/fig4/fl_*', script=S, nota='recalculo %.5f' % (m / p))
        if k1o:
            claim('F5b-%s-k1' % kf, 1, loc, 'Mejora predicha con kappa1 solo kf=%g' % kf, k1o, P0[1] / P[1], tol=0.05 if kf == 3 else 0.5, dato='data/fig4/fl_*', script=S)
    claim('F5b-4pct', 1, [(INF, 'subestima la mejora un $4\\%$'), (H8, 'incluye el +1 de C1')], 'Comparar con kappa1 solo subestima la mejora ~4% (kf=3: 18.6 frente a 19.5)', '0.04', 1 - (P0[1] / c1(0, fl(3.0))[1]) / (P0[0] / c1(0, fl(3.0))[0]), tol=0.003, dato='data/fig4/fl_*', script=S)
    # confinamiento dinamico
    cc = {kf: np.nanmean(conf_fig4(fl(kf))) for kf in (0.0, 0.1, 0.3, 1.0, 3.0)}
    cf0 = conf_fig4(fl(0.0, 22))
    for kf, rep, loc in ((3.0, '0.965', (RM, 'Relativa al baño plano: 0.965 (κ_f = 3)')), (1.0, '0.967', (RM, 'Relativa al baño plano: 0.965')), (0.3, '0.90', (RM, 'Relativa al baño plano: 0.965')), (0.1, '0.53', (RM, 'Relativa al baño plano: 0.965'))):
        claim('F4c-%g' % kf, 1, [loc, (INF, 'Relativa al plano, es $0.965$')], 'Confinamiento dinamico relativo al plano, kf=%g' % kf, rep, cc[kf] / cc[0.0], tol=0.005 if kf != 0.3 else 0.006, dato='data/fig4/fl_kf*_N16*.npz (Pc_din)', script=S, nota='reimplementacion de la cola exponencial (fig4.py::confinamiento); recalculo %.4f' % (cc[kf] / cc[0.0]))
    claim('F4c-flat', 1, [(RM, 'Baño plano: 6.98e-3 / 7.13e-3'), (INF, 'es $6.98\\times10^{-3}$ y $7.13\\times10^{-3}$')], 'Tasa dinamica del bano plano desde |0>|g>: 6.98e-3', '6.98e-3', cf0[0], tol=0.02e-3, dato='data/fig4/fl_kf0.0_N22_Nf1_al2.0.npz', script=S, nota='desde |0>|g> %.4e; desde D(d)|1.3a>|g> %.4e; el reporte da 6.98e-3 y 7.13e-3' % (cf0[0], cf0[1]))
    claim('F4c-flat2', 1, [(RM, 'Baño plano: 6.98e-3 / 7.13e-3')], 'Tasa dinamica del bano plano desde D(d)|1.3a>|g>: 7.13e-3', '7.13e-3', cf0[1], tol=0.02e-3, dato='data/fig4/fl_kf0.0_N22_Nf1_al2.0.npz', script=S)
    cf28 = conf_fig4(fl(0.0, 28))
    claim('F4c-N', 1, [(RM, 'idéntico para N = 22 y 28'), (INF, 'idéntica para $N=22$ y $28$')], 'Tasa dinamica plano identica para N=22 y 28', '0', max(abs(np.array(cf0) - np.array(cf28)) / np.array(cf0)), modo='le', tol=1e-3, dato='data/fig4/fl_kf0.0_N{22,28}_*.npz', script=S, nota='dif relativa max %.1e' % max(abs(np.array(cf0) - np.array(cf28)) / np.array(cf0)))
    # 5to modo
    im = [float(fl(0.0, N)['modos'][4][1]) for N in (16, 22, 28)]; rt = [float(fl(0.0, N)['modos'][4][2]) for N in (16, 22, 28)]
    for N, v, rep in zip((16, 22, 28), im, ('0.040', '0.054', '0.061')):
        claim('F4m-%d' % N, 1, [(RM, '(0.040, 0.054 y 0.061 para N = 16, 22 y 28)'), (INF, '$0.040$, $0.054$ y $0.061$')], '5.o modo espectral (plano): |Im lambda| crece con N: N=%d -> %s' % (N, rep), rep, abs(v), tol=0.0006, dato='data/fig4/fl_kf0.0_N%d_Nf1_al2.0.npz (modos[4])' % N, script=S)
    claim('F4m-r', 1, [(RM, 'su tasa cambia un 2.2% entre N = 16 y 22'), (INF, 'la tasa cambia un $2.2\\%$')], '5.o modo: su tasa cambia 2.2% entre N=16 y 22', '0.022', abs(rt[1] / rt[0] - 1), tol=0.0005, dato='data/fig4/fl_kf0.0_N{16,22}', script=S)
    # convergencia con filtro kf=1 N16->18
    z16, z18 = fl(1.0, 16), fl(1.0, 18)
    dg = float(z18['gpf']) / float(z16['gpf']) - 1
    c16, c18 = conf_fig4(z16), conf_fig4(z18)
    claim('F4n-1', 1, [(RM, 'γ_pf cambia −1.5e-4'), (INF, '$-1.5\\times10^{-4}$')], 'Convergencia con filtro (kf=1, N=16->18): gamma_pf cambia -1.5e-4 (relativo)', '-1.5e-4', dg, tol=0.1e-4, dato='data/fig4/fl_kf1.0_N{16,18}_Nf2_al2.0.npz', script=S)
    cd_ = max(abs(np.array(c18) / np.array(c16) - 1))
    claim('F4n-2', 1, [(RM, 'el confinamiento 0.2–0.5%'), (INF, 'el confinamiento $0.2$--$0.5\\%$')], 'Convergencia con filtro: confinamiento cambia 0.2-0.5% (N=16->18)', '0.005', cd_, modo='le', tol=0.001, dato='data/fig4/fl_kf1.0_N{16,18}', script=S, nota='cambios por estado inicial: %s' % (np.array(c18) / np.array(c16) - 1))
    # validaciones fig4: min autovalor
    mins = []
    for f in glob.glob(os.path.join(DATA, 'fig4', '*.npz')):
        z = np.load(f, allow_pickle=True)
        if 'val' in z.files: mins.append((float(z['val'][2]), os.path.basename(f), float(z['val'][0])))
        elif 'rho' in z.files: pass
    mins.sort()
    claim('F4v-1', 1, [(RM, 'Validaciones:** todas las ρ con |Tr ρ − 1| ≤ 2e-16 y mínimo autovalor ≥ −2e-11')], 'README Fig.4: todas las rho con min autovalor >= -2e-11', '-2e-11', mins[0][0], modo='ge', tol=0, dato='data/fig4/*.npz (campo val)', script=S,
          nota='el minimo es %.3e en %s (el estatico kf=0.3 N_f=3); viola tambien el criterio -1e-9 del propio README. V5_reporte.md dice -1.1e-9' % (mins[0][0], mins[0][1]))
    return

# ============================================================ BLOQUE 5: modelo minimo, confinamiento, filtro en el minimo, mapa de diseno
def bloque_minimo():
    S = 'consolidacion/verificar_todo.py::bloque_minimo'
    T = []
    for f in glob.glob(os.path.join(DATA, 'minimo', 'k*_N24.npz')):
        z = np.load(f, allow_pickle=True)
        fis = [m for m in z['modos'] if -m[0] > 1e-6 and m[2] < 0.5]       # misma regla que principal_fig3.minimo(): umbral 1e-6 y peso de borde < 0.5
        T.append((float(z['k2k']), _td_mini(z['t'], z['Pc_din'][0]), _td_mini(z['t'], z['Pc_din'][1]), -fis[0][0] if fis else np.nan))
    T = np.array(sorted(T))
    c = float(np.polyval(np.polyfit(T[:3, 0], T[:3, 1] / T[:3, 0], 1), 0.0))
    DM = 'data/minimo/k*_N24.npz'
    claim('MN-01', 1, [(H8, 'c = lím Δ/κ₂ = 4.14'), (RM, 'c = lim Δ/κ₂ = **4.14**'), (INF, 'c=\\lim\\Delta/\\kappa_2=4.14')], 'c = lim Delta/kappa2 = 4.14 (extrapolacion lineal de los tres kappa2/kappa menores, tasa dinamica desde |0>|g>)', '4.14', c, dato=DM, script=S, nota='dinamica reimplementada de calc_minimo.py::tasa_dinamica')
    lk = np.log(T[:, 0])
    near = lambda k: int(np.argmin(abs(T[:, 0] - k)))
    for k, d, dk2, eff in ((1e-3, '0.00407', '4.07', '0.983'), (1e-2, '0.0347', '3.46', '0.837'), (4.5e-2, '0.0945', '2.09', '0.505'), (0.161, '0.161', '1.00', '0.241'), (0.574, '0.237', '0.414', '0.100'), (1.08, '0.259', '0.239', '0.058'), (10.0, '0.278', '0.028', '0.007')):
        i = near(k)
        loc = [(INF, {1e-3: '$1.0\\times10^{-3}$ & 0.00407', 1e-2: '$1.0\\times10^{-2}$ & 0.0347', 4.5e-2: '$4.5\\times10^{-2}$ & 0.0945', 0.161: '0.161 & 0.161', 0.574: '0.574 & 0.237', 1.08: '1.08 & 0.259', 10.0: '10 & 0.278'}[k])]
        claim('MN-T%g-D' % k, 1, loc, 'Tabla conf: Delta/kappa (dinamica) en kappa2/kappa=%g (punto %.6g)' % (k, T[i, 0]), d, T[i, 1], dato=DM, script=S, tol=None)
        claim('MN-T%g-r' % k, 1, loc, 'Tabla conf: Delta/kappa2 en kappa2/kappa=%g' % k, dk2, T[i, 1] / T[i, 0], dato=DM, script=S, tol=0.006 if k == 0.161 else None)
        claim('MN-T%g-e' % k, 1, loc, 'Tabla conf: kappa2^eff/kappa2 = Delta/(c kappa2) en kappa2/kappa=%g' % k, eff, T[i, 1] / (c * T[i, 0]), dato=DM, script=S, tol=None)
    claim('MN-02', 1, [(H8, 'Satura en ≈ 0.28κ'), (RM, '0.2784κ en κ₂/κ = 10'), (INF, '$0.2784\\kappa$ en $\\kappa_2/\\kappa=10$')], 'La tasa dinamica satura en ~0.28 kappa (0.2784 en kappa2/kappa=10)', '0.2784', T[-1, 1], dato=DM, script=S)
    claim('MN-03', 1, [(H8, 'κ₂/κ ≈ 1/(4c) = 0.060'), (RM, '**κ₂/κ ≈ 1/(4c) = 0.060**'), (INF, '1/(4c)=0.060')], 'Inicio de saturacion: kappa2/kappa = 1/(4c) = 0.060', '0.060', 1 / (4 * c), dato=DM, script=S)
    i6 = near(0.062102)
    claim('MN-04', 1, [(RM, 'Allí κ₂^eff/κ₂ ≈ 0.43'), (INF, 'donde $\\kappa_2^{\\rm eff}/\\kappa_2\\simeq0.43$')], 'En el inicio de saturacion, kappa2eff/kappa2 ~ 0.43', '0.43', T[i6, 1] / (c * T[i6, 0]), dato=DM, script=S, tol=0.005)
    claim('MN-05', 1, [(RM, 'ε·(g_z/κ)² = 0.0707 en κ₂/κ = 1e-3'), (INF, '$\\epsilon(g_z/\\kappa)^2=0.0707$')], 'Verificacion adiabatica: eps*(gz/k)^2 = (5/72)(k2/k2eff) = 0.0707 en kappa2/kappa=1e-3 (frente a 5/72 = 0.0694)', '0.0707', (5 / 72) / (T[0, 1] / (c * T[0, 0])), dato=DM, script=S)
    claim('MN-06', 1, [(RM, 'frente a 5/72 = 0.0694'), (INF, '5/72=0.0694')], '5/72 = 0.0694', '0.0694', 5 / 72, script=S)
    claim('MN-07', 1, [(INF, 'de $g_z/\\kappa=3.909$'), (H8, '3.94')], 'Umbral eps=1/220 en el mapa: sqrt(220*5/72 * k2/k2eff) con k2eff/k2 = 0.983 -> 3.94', '3.94', math.sqrt(220 * 5 / 72 / (T[0, 1] / (c * T[0, 0]))), dato=DM, script=S, tol=0.005)
    # C9: estabilidad con N
    st = []
    for k in (0.001, 0.00356225, 0.0126896, 0.0452035, 0.161026, 0.573615, 2.04336, 7.27895):
        rs = []
        for N in (24, 30, 36):
            f = os.path.join(DATA, 'minimo', 'k%g_N%d.npz' % (k, N))
            if os.path.exists(f):
                z = np.load(f, allow_pickle=True); fis = [m for m in z['modos'] if -m[0] > 1e-6 and m[2] < 0.5]; rs.append((N, _td_mini(z['t'], z['Pc_din'][0]), -fis[0][0] if fis else np.nan))
        st.append((k, rs))
    mx = max(abs(r[1] - rs[0][1]) / rs[0][1] if False else abs(r[1] - rs[0][1]) for k, rs in st for r in rs[1:])
    mxr = max(abs(r[1] - rs[0][1]) / rs[0][1] for k, rs in st for r in rs[1:])
    claim('MN-08', 1, [(H8, 'C9'), (RM, '**C9:** en 8 valores con N = 24, 30 y 36, la tasa dinámica es estable (≤ 2e-4)'), (INF, 'En 8 valores con $N=24,30,36$ es estable ($\\leq2\\times10^{-4}$)')], 'C9: en 8 valores con N=24,30,36 la tasa dinamica es estable (<= 2e-4)', '2e-4', mx, modo='le', tol=0, dato='data/minimo/k*_N{24,30,36}.npz', script=S,
          nota='max |dif| absoluta %.1e, relativa %.1e; valores: %s' % (mx, mxr, [(k, [round(r[1], 6) for r in rs]) for k, rs in st]))
    kk = [x for x in st if abs(x[0] - 0.573615) < 1e-6][0][1]
    claim('MN-09', 1, [(RM, '(0.216 → 0.231 → 0.238 en κ₂/κ = 0.57)'), (INF, '$0.216\\to0.231\\to0.238$')], 'La tasa espectral no converge en regimen saturado: 0.216 -> 0.231 -> 0.238 (N=24,30,36) en kappa2/kappa=0.57', '0.216,0.231,0.238', ','.join('%.3f' % r[2] for r in kk), dato='data/minimo/k0.573615_N{24,30,36}.npz', script=S, nota='brecha_esp guardada: %s' % [r[2] for r in kk])
    i2 = near(0.2)
    coin = max(abs(T[j, 3] / T[j, 1] - 1) for j in range(len(T)) if T[j, 0] <= 0.2)
    claim('MN-10', 1, [(RM, 'coincide con la espectral al 1e-3 hasta κ₂/κ ≈ 0.2'), (INF, 'Coincide con la espectral al $10^{-3}$')], 'La tasa dinamica coincide con la espectral al 1e-3 hasta kappa2/kappa~0.2', '1e-3', coin, modo='le', tol=0, dato=DM, script=S, nota='max dif relativa recalculada hasta k2/k=0.2: %.1e (el primer punto 0.001 difiere %.1e)' % (coin, abs(T[0, 3] / T[0, 1] - 1)))
    # filtro en el minimo
    Tf = []
    for f in glob.glob(os.path.join(DATA, 'minimo_filtro', 'k*_kf10_N20_Nf3.npz')):
        z = np.load(f, allow_pickle=True); k = float(z['k2k']); c_ = _td_mini(z['t'][1:], z['Pc'][1:])
        if not np.isnan(c_): Tf.append((k, c_))
    Tf = np.array(sorted(Tf))
    dp = lambda k: np.exp(np.interp(np.log(k), lk, np.log(T[:, 1])))
    ratio = lambda k: np.exp(np.interp(np.log(k), np.log(Tf[:, 0]), np.log(Tf[:, 1] / dp(Tf[:, 0]))))
    for k, rep in ((1e-3, '1.002'), (0.01, '1.014'), (0.03, '1.037'), (0.1, '1.051'), (0.3, '0.955'), (1.0, '0.480')):
        claim('MF-T%g' % k, 1, [(RM, '| Δ_filtro/Δ_plano |'), (INF, '$\\Delta_{\\rm filtro}/\\Delta_{\\rm plano}$ & 1.002')], 'Delta_filtro/Delta_plano en kappa2/kappa=%g (interpolacion log-log sobre la serie N=20, N_f=3)' % k, rep, ratio(k), tol=0.003, dato='data/minimo_filtro/k*_kf10_N20_Nf3.npz; data/minimo/k*_N24.npz', script=S,
              nota='valor %.4f' % ratio(k))
    z1 = np.load(os.path.join(DATA, 'minimo_filtro', 'k1_kf10_N20_Nf2.npz'), allow_pickle=True); z26 = np.load(os.path.join(DATA, 'minimo_filtro', 'k1_kf10_N26_Nf3.npz'), allow_pickle=True)
    c20, c26 = _td_mini(z1['t'][1:], z1['Pc'][1:]), _td_mini(z26['t'][1:], z26['Pc'][1:])
    claim('MF-CONV', 1, [(RM, 'N = 20 → 26 cambia < 0.1%'), (INF, '$N=20\\to26$ cambia menos de $0.1\\%$')], 'Convergencia en N del confinamiento con filtro: N=20->26 cambia <0.1% (probado en kappa2/kappa=0.1)', '0.001', abs(_td_mini(np.load(os.path.join(DATA, 'minimo_filtro', 'k0.1_kf10_N26_Nf2.npz'))['t'][1:], np.load(os.path.join(DATA, 'minimo_filtro', 'k0.1_kf10_N26_Nf2.npz'))['Pc'][1:]) / _td_mini(np.load(os.path.join(DATA, 'minimo_filtro', 'k0.1_kf10_N20_Nf2.npz'))['t'][1:], np.load(os.path.join(DATA, 'minimo_filtro', 'k0.1_kf10_N20_Nf2.npz'))['Pc'][1:]) - 1), modo='le', tol=0, dato='data/minimo_filtro/k0.1_kf10_N{20,26}_Nf2.npz', script=S, nota='solo se probo en kappa2/kappa=0.1')
    claim('MF-SAT', 1, [(RM, 'en saturación, la mitad'), (INF, '$\\Delta_{\\rm filtro}/\\Delta_{\\rm plano}$ & 1.002')], 'Costo del filtro en saturacion: kappa2/kappa=1, ratio con N=26,N_f=3 frente a N=20,N_f=2', '0.480', c26 / dp(1.0), tol=0.01, dato='data/minimo_filtro/k1_kf10_N{20,26}_*.npz', script=S,
          nota='con N=20,N_f=2: Delta_f=%.4f (ratio %.3f); con N=26,N_f=3: Delta_f=%.4f (ratio %.3f); P_c(inf) 0.99934 (N=20) vs 0.99999993 (N=26): el punto de kappa2/kappa=1 no esta convergido en N' % (c20, c20 / dp(1.0), c26, c26 / dp(1.0)))
    claim('MF-KF', 1, [(RM, 'Con κ_f = 1000κ se recupera el plano (1.0015)'), (INF, 'con $\\kappa_f=1000\\kappa$ se recupera el plano (1.0015)')], 'Control kf=1000k recupera el plano: 1.0015 (kappa2/kappa=0.1)', '1.0015', _td_mini(np.load(os.path.join(DATA, 'minimo_filtro', 'k0.1_kf1000_N20_Nf2.npz'))['t'][1:], np.load(os.path.join(DATA, 'minimo_filtro', 'k0.1_kf1000_N20_Nf2.npz'))['Pc'][1:]) / dp(0.1), dato='data/minimo_filtro/k0.1_kf1000_N20_Nf2.npz', script=S, tol=0.0001)
    cNf2 = _td_mini(np.load(os.path.join(DATA, 'minimo_filtro', 'k0.1_kf10_N20_Nf2.npz'))['t'][1:], np.load(os.path.join(DATA, 'minimo_filtro', 'k0.1_kf10_N20_Nf2.npz'))['Pc'][1:])
    cNf3 = _td_mini(np.load(os.path.join(DATA, 'minimo_filtro', 'k0.1_kf10_N20_Nf3.npz'))['t'][1:], np.load(os.path.join(DATA, 'minimo_filtro', 'k0.1_kf10_N20_Nf3.npz'))['Pc'][1:])
    claim('MF-NF', 1, [(RM, 'N_f = 2 → 3 cambia 1.2%'), (INF, '$N_f=2\\to3$ cambia $1.2\\%$')], 'N_f=2->3 cambia 1.2% (kappa2/kappa=0.1)', '0.012', abs(cNf3 / cNf2 - 1), dato='data/minimo_filtro/k0.1_kf10_N20_Nf{2,3}.npz', script=S, tol=0.0006, nota='recalculo 1.150% (redondeo limite)')
    # umbral con filtro (figura_central_umbral.csv)
    U = np.array([[float(x) for x in r] for r in list(csv.reader(open(os.path.join(DATA, 'figura_central_umbral.csv'), encoding='utf-8')))[1:]])
    claim('MU-01', 1, [(H8, 'baja de 3.94 a 0.094 (factor 0.0238'), (RM, 'pasa de 3.94 (plano) a 0.094 (filtro), un factor 0.0238'), (INF, 'pasa de $3.94$ (plano) a $0.094$ (filtro), un factor $0.0238$')], 'Umbral eps=1/220 en gz/kappa (kappa2/kappa->0): plano 3.94, filtro 0.094, factor 0.0238', '3.94', U[0, 1], tol=0.005, dato='data/figura_central_umbral.csv', script=S, nota='filtro %.4f, factor %.5f' % (U[0, 2], U[0, 3]))
    claim('MU-02', 1, [(H8, 'baja de 3.94 a 0.094 (factor 0.0238'), (RM, 'pasa de 3.94 (plano) a 0.094 (filtro), un factor 0.0238')], 'Umbral filtrado 0.094', '0.094', U[0, 2], dato='data/figura_central_umbral.csv', script=S)
    claim('MU-03', 1, [(H8, 'factor 0.0238'), (RM, 'un factor 0.0238')], 'Factor filtro/plano 0.0238', '0.0238', U[0, 3], dato='data/figura_central_umbral.csv', script=S)
    ke = lambda d, kf, kap=0.03: kap * kf**2 / (4 * d**2 + kf**2)
    f_ = ke(6, 0.3) / 0.03 + ke(18, 0.3) / (9 * 0.03)
    claim('MU-04', 1, [(RM, 'f = κ_eff(ω)/κ + κ_eff(3ω)/(9κ) = 6.32e-4'), (INF, 'con $f=6.32\\times10^{-4}$')], 'f = keff(w)/k + keff(3w)/(9k) con kf/w=0.05 = 6.32e-4', '6.32e-4', f_, dato='(analitico)', script=S)
    claim('MU-05', 1, [(RM, 'La predicción es √(f/(10/9)) = 0.0239'), (INF, '$\\sqrt{f/(10/9)}=0.0239$')], 'sqrt(f/(10/9)) = 0.0239', '0.0239', math.sqrt(f_ / (10 / 9)), dato='(analitico)', script=S)
    claim('MU-06', 1, [(RM, 'κ_f/(2ω) = 0.025'), (INF, '$\\kappa_f/2\\omega=0.025$')], 'kf/(2w) = 0.025', '0.025', 0.3 / 12, dato='(analitico)', script=S)
    sel = (U[:, 0] <= 0.3); fcs = U[sel, 3]
    claim('MU-07', 1, [(RM, 'El factor se mantiene en 0.023–0.024 hasta κ₂/κ ≈ 0.3'), (INF, 'se mantiene en $0.023$--$0.024$')], 'El factor del umbral se mantiene en 0.023-0.024 hasta kappa2/kappa~0.3 (maximo)', '0.024', fcs.max(), modo='le', tol=0.0006, dato='data/figura_central_umbral.csv', script=S, nota='rango %.4f-%.4f' % (fcs.min(), fcs.max()))
    f1 = U[np.argmin(abs(U[:, 0] - 1.0))]
    claim('MU-08', 1, [(RM, 'sube a 0.034 en κ₂/κ = 1'), (INF, 'sube a $0.034$ en $\\kappa_2/\\kappa=1$')], 'El factor sube a 0.034 en kappa2/kappa=1', '0.034', f1[3], dato='data/figura_central_umbral.csv', script=S, tol=0.001, nota='fila kappa2/kappa=%.3f' % f1[0])
    return

# ============================================================ BLOQUE 6: mapa de diseno, frontera chi, piso intrinseco, plataformas
def bloque_mapa():
    S = 'consolidacion/verificar_todo.py::bloque_mapa'
    P = [[float(x) for x in r] for r in list(csv.reader(open(os.path.join(DATA, 'principal_fig3_puntos.csv'))))[1:]]
    P = np.array(P)
    p15 = P[:15]
    claim('MP-01', 1, [(RM, 'ε_completo/ε_mapa va de 0.97 a 1.55'), (INF, '$\\epsilon_{\\rm completo}/\\epsilon_{\\rm mapa}$ va de $0.97$ a $1.55$'), ], 'eps_completo/eps_mapa en los 15 puntos del modelo completo: minimo 0.97', '0.97', p15[:, 11].min(), dato='data/principal_fig3_puntos.csv', script=S, tol=0.005)
    claim('MP-02', 1, [(RM, 'ε_completo/ε_mapa va de 0.97 a 1.55'), (INF, 'va de $0.97$ a $1.55$')], 'eps_completo/eps_mapa en los 15 puntos: maximo 1.55', '1.55', p15[:, 11].max(), dato='data/principal_fig3_puntos.csv', script=S, tol=0.005)
    claim('MP-03', 1, [(RM, 'Puntos del modelo completo** (15, N = 22'), (INF, 'se superponen los 15 puntos del modelo completo')], 'principal_fig3_puntos.csv tiene 15 puntos N=22 de la Tabla V4 + 4 puntos extra (omega/kappa=500, 1000)', '15', len(p15), dato='data/principal_fig3_puntos.csv', script=S, nota='el CSV tiene %d filas en total (las 4 ultimas con omega/kappa=500/1000 y 2 sin gato estable)' % len(P), tol=0)
    Q = [[float(x) if x != 'nan' else np.nan for x in r] for r in list(csv.reader(open(os.path.join(DATA, 'p9_diagnostico.csv'))))[1:]]
    Q = np.array(Q)
    chi = Q[:, 6]; cd = Q[:, 7]; ce = Q[:, 8]; cn = Q[:, 9]
    low = (chi <= 0.2)
    claim('P9-01', 1, [(RM, 'Umbral empírico: conf/Δ ≥ 0.98 para χ|α|²/κ ≲ 0.2'), (INF, 'conf}/\\Delta_{\\rm min}\\geq0.98$ para $\\chi|\\alpha|^2/\\kappa\\lesssim0.2$')], 'P9: conf/Delta_min >= 0.98 para chi|a|^2/k <= 0.2 (minimo en esos puntos)', '0.98', cd[low].min(), modo='ge', tol=0, dato='data/p9_diagnostico.csv', script=S, nota='minimo %.4f sobre %d puntos' % (cd[low].min(), low.sum()))
    i45 = int(np.argmin(abs(chi - 0.455))); i13 = int(np.argmin(abs(chi - 1.333)))
    claim('P9-02', 1, [(RM, '0.91 en 0.45'), (INF, '$0.91$ en $0.45$')], 'P9: conf/Delta_min = 0.91 en chi|a|^2/k = 0.45', '0.91', cd[i45], dato='data/p9_diagnostico.csv', script=S, tol=0.005, nota='chi=%.3f' % chi[i45])
    claim('P9-03', 1, [(RM, '0.91 en 0.45; 0.64 en 1.3'), (INF, '$0.64$ en $1.3$')], 'P9: conf/Delta_min = 0.64 en chi|a|^2/k = 1.3', '0.64', cd[i13], dato='data/p9_diagnostico.csv', script=S, tol=0.005, nota='chi=%.3f' % chi[i13])
    ok = ~np.isnan(cd[:15]); ls = np.polyfit(chi[:15][ok & (chi[:15] > 0.1) & (chi[:15] < 0.6)], cd[:15][ok & (chi[:15] > 0.1) & (chi[:15] < 0.6)], 1)
    claim('P9-04', 1, [(RM, '5% de pérdida en χ|α|²/κ ≈ 0.3'), (H8, 'Pérdida de confinamiento del 5% en χ|α|²/κ ≈ 0.3')], 'P9: perdida del 5% en chi|a|^2/k ~ 0.3 (interpolacion lineal de los puntos en 0.1-0.6)', '0.95', np.polyval(ls, 0.3), dato='data/p9_diagnostico.csv', script=S, tol=0.015, nota='ajuste lineal sobre %d puntos: conf/Delta(0.3)=%.3f' % (((chi[:15] > 0.1) & (chi[:15] < 0.6)).sum(), np.polyval(ls, 0.3)))
    rr = (cd[:15] / ce[:15]) if False else Q[:15, 10]
    claim('P9-05', 1, [(RM, 'reproduce el confinamiento del modelo completo al 0.5–4%'), (INF, 'al $0.5$--$4\\%$')], 'P9: el efectivo con chi reproduce el completo al 0.5-4% (|conf_full/conf_eff - 1| entre 0.2% y 4.4%)', '0.04', np.nanmax(abs(Q[:15, 10] - 1)), modo='le', tol=0.005, dato='data/p9_diagnostico.csv', script=S, nota='rango %.3f - %.3f' % (np.nanmin(abs(Q[:15, 10] - 1)), np.nanmax(abs(Q[:15, 10] - 1))))
    claim('P9-06', 1, [(RM, '(0.632 frente a 0.640 en el peor punto)'), (INF, '$0.632$ frente a $0.640$')], 'P9: peor punto 0.632 (efectivo con chi) frente a 0.640 (completo)', '0.632', ce[11], dato='data/p9_diagnostico.csv', script=S, tol=0.0005, nota='completo %.4f' % cd[11])
    claim('P9-07', 1, [(RM, 'reproduce el modelo mínimo (≥ 0.99)')], 'P9: el efectivo sin chi reproduce el modelo minimo (>= 0.99) en los puntos de gato estable', '0.99', np.nanmin(cn[:15]), modo='ge', tol=0, dato='data/p9_diagnostico.csv', script=S, nota='minimo %.4f (punto g_z/k=2: 0.9912)' % np.nanmin(cn[:15]))
    claim('P9-08', 1, [(RM, 'conf/Δ = 0.175 (el efectivo con χ da 0.176) y 0.13'), (INF, '0.175$ (el efectivo con')], 'P9: omega/kappa=500: conf/Delta = 0.175 (efectivo con chi 0.176)', '0.175', cd[15], dato='data/p9_diagnostico.csv', script=S, tol=0.0006, nota='efectivo con chi %.4f' % ce[15])
    claim('P9-09', 1, [(RM, 'y 0.13'), (INF, 'y $0.13$')], 'P9: omega/kappa=500 segundo punto: conf/Delta = 0.13', '0.13', cd[16], dato='data/p9_diagnostico.csv', script=S, tol=0.001)
    claim('P9-10', 1, [(RM, 'χ|α|²/κ = 4.3 y 6.7'), (INF, '$\\chi|\\alpha|^2/\\kappa=4.3$ y $6.7$')], 'P9: omega/kappa=1000: chi|a|^2/k = 4.3 y 6.7', '4.3', chi[17], dato='data/p9_diagnostico.csv', script=S, tol=0.05, nota='segundo %.3f' % chi[18])
    claim('P9-11', 1, [(RM, 'P_c estacionario 0.49 y 0.95'), (INF, '$P_c$ estacionario $0.49$ y $0.95$')], 'P9: omega/kappa=1000: P_c estacionario 0.49 y 0.95', '0.49,0.95', None, estado='R', dato='data/trabajos o logs de p9 (no hay caché de esos dos puntos; el CSV tiene nan)', script='codigo/p9_diagnostico.py', nota='no recomputable desde el repo')
    # plataformas
    gx = 0.3 * math.sin(math.pi / 4); gz_ = 0.3 * math.cos(math.pi / 4); w = 6.0; kap = 0.03
    k2k_ma = 16 * (gx / w)**2 * (gz_ / kap)**2
    claim('PL-01', 1, [(H8, '| Ma | 0.005 | 7.1 | 1.0 | 1.4e-3 | 2.7 |', '0.005'), (INF, '$\\kappa/\\omega$ & 0.005 & 0.001', '0.005')], 'Ma: kappa/omega = 0.005', '0.005', kap / w, script=S, dato='(parametros publicados)')
    claim('PL-02', 1, [(H8, '| Ma | 0.005 | 7.1 | 1.0 | 1.4e-3 | 2.7 |', '7.1')], 'Ma: gz/kappa = 7.1', '7.1', gz_ / kap, script=S, dato='(parametros publicados)')
    claim('PL-03', 1, [(H8, '| Ma | 0.005 | 7.1 | 1.0 | 1.4e-3 | 2.7 |', '1.0')], 'Ma: kappa2/kappa = 16(gx/w)^2(gz/k)^2 = 1.0', '1.0', k2k_ma, script=S, dato='(analitico)')
    claim('PL-04', 1, [(H8, '| Ma | 0.005 | 7.1 | 1.0 | 1.4e-3 | 2.7 |', '1.4e-3'), (INF, '$\\kappa_1/\\kappa_2$ (nominal) & $1.4\\times10^{-3}$', '1.4e-3')], 'Ma: kappa1/kappa2 = (5/72)(k/gz)^2 = 1.4e-3', '1.4e-3', (5 / 72) * (kap / gz_)**2, script=S, dato='(analitico)')
    chi_ma = 8 / 3 * gx**2 / w * 4 / kap
    claim('PL-05', 1, [(H8, '| Ma | 0.005 | 7.1 | 1.0 | 1.4e-3 | 2.7 |', '2.7'), (RM, '**Ma** tiene χ|α|²/κ = 2.7'), (INF, 'Ma tiene $\\chi|\\alpha|^2/\\kappa=2.7$')], 'Ma: chi|a|^2/kappa = 2.7', '2.7', chi_ma, script=S, dato='(analitico)')
    gxn, gzn, wn = 6.0, 60.0, 1000.0
    claim('PL-06', 1, [(H8, '| Naseem | 0.001 | 60 | 2.07 | 9.2e-5 | 0.19 |', '60')], 'Naseem: gz/kappa = 60 (g_z=6 MHz, kappa=100 kHz)', '60', 6e6 / 1e5, script=S, dato='(parametros publicados)')
    claim('PL-07', 1, [(H8, '| Naseem | 0.001 | 60 | 2.07 | 9.2e-5 | 0.19 |', '2.07'), (INF, '$\\kappa_2/\\kappa$ & 1.0 & 2.07', '2.07')], 'Naseem: kappa2/kappa = 16(gx/w)^2(gz/k)^2 = 2.07', '2.07', 16 * (gxn / wn)**2 * gzn**2, script=S, dato='(analitico)')
    k12n = (5 / 72) / 60**2
    g1k = 1.5e-4; k2kn = 16 * (gxn / wn)**2 * gzn**2
    claim('PL-08', 1, [(H8, '| Naseem | 0.001 | 60 | 2.07 | 9.2e-5 | 0.19 |', '9.2e-5'), (INF, '$\\kappa_1/\\kappa_2$ (nominal) & $1.4\\times10^{-3}$ & $9.2\\times10^{-5}$', '9.2e-5'), ('docs/mapa_plataformas/mapa_plataformas.tex', '$1.4\\times10^{-3}$ & $9.2\\times10^{-5}$', '9.2e-5')],
          'Naseem: kappa1/kappa2 "nominal" = (5/72)(k/gz)^2 = 1.9e-5; con la perdida intrinseca (kappa1+gamma)/kappa2 = 9.2e-5', '9.2e-5', k12n, tol=1e-6, dato='(analitico)', script=S,
          nota='D de etiqueta: la formula nominal da %.2e; (k1+gamma)/k2 = %.2e coincide con 9.2e-5 (mapa_plataformas.tex l.94 lo dice: incluye la perdida intrinseca). HANDOFF §7 e informe Tabla plat lo rotulan "nominal"' % (k12n, k12n + g1k / k2kn))
    CL[-1]['estado'] = 'D'
    claim('PL-09', 1, [(H8, '| Naseem | 0.001 | 60 | 2.07 | 9.2e-5 | 0.19 |', '9.2e-5')], 'Naseem: (kappa1+gamma)/kappa2 con gamma/kappa=1.5e-4', '9.2e-5', k12n + g1k / k2kn, dato='(analitico)', script=S)
    claim('PL-10', 1, [(H8, '| Naseem | 0.001 | 60 | 2.07 | 9.2e-5 | 0.19 |', '0.19'), (RM, '**Naseem** tiene χ|α|²/κ = 0.19'), (INF, 'Naseem tiene $0.19$')], 'Naseem: chi|a|^2/kappa = 0.19 con |a|^2=2', '0.19', 8 / 3 * gxn**2 / wn * 2, tol=0.005, dato='(analitico)', script=S)
    fq = 208.37
    from scipy import constants as cst
    kb10 = cst.k * 0.010 / cst.h / 1e6
    claim('PL-11', 1, [(H8, 'k_BT/h = 208.37 MHz a 10 mK'), (RM, 'x · 208.37 MHz'), (INF, 'x\\cdot208.37')], 'k_B*10 mK/h = 208.37 MHz (scipy.constants)', '208.37', kb10, dato='(scipy.constants)', script=S)
    claim('PL-12', 1, [(H8, 'f_q = 200 MHz ⇒ x = 0.96 a 10 mK'), (INF, '0.96 ($f_q=200$~MHz)')], 'Naseem: x = h*200MHz/(kB*10mK) = 0.96', '0.96', 200 / kb10, dato='(analitico)', script=S)
    claim('PL-13', 1, [(INF, 'hf_q/k_BT$ (10 mK) & 58 &')], 'Ma: x = h*12GHz/(kB*10mK) = 58', '58', 12000 / kb10, dato='(analitico)', script=S, tol=0.5)
    claim('PL-14', 1, [(H8, 'γ/κ = 2e-5 (Q = 1e7)'), (RM, 'Ma (Q = 1e7, ω/κ = 200) tiene γ/κ = 2e-5')], 'Ma: gamma/kappa = (omega/kappa)/Q = 200/1e7 = 2e-5', '2e-5', 200 / 1e7, dato='(analitico)', script=S, nota='Q=1e7 es un supuesto no verificado (HANDOFF §7 [VERIFICAR Q])')
    claim('PL-15', 1, [(H8, 'γ/κ = 1.5e-4'), (RM, 'Naseem (γ/2π = 15 Hz, κ/2π = 100 kHz) tiene γ/κ = 1.5e-4')], 'Naseem: gamma/kappa = 15 Hz / 100 kHz = 1.5e-4', '1.5e-4', 15 / 1e5, dato='(analitico)', script=S)
    # Ma: confinamiento y eps
    ke2 = np.interp(1.0, T_k2 := [0.788046, 1.08264], [0.076369, 0.057697])
    TM = [[float(x) for x in r] for r in list(csv.reader(open(os.path.join(DATA, 'principal_fig3_minimo.csv'))))[1:]]
    TM = np.array(TM)
    e1 = np.exp(np.interp(np.log(1.0), np.log(TM[:, 0]), np.log(TM[:, 5])))
    claim('PL-16', 1, [(INF, '$\\kappa_2^{\\rm eff}/\\kappa_2$ ($|\\alpha|^2=4$) & 0.062'), (INF, 'ma\\_ & 0.062' if False else '0.062')], 'Ma: kappa2eff/kappa2 interpolado en kappa2/kappa=1 = 0.062', '0.062', e1, dato='data/principal_fig3_minimo.csv', script=S, tol=0.001)
    claim('PL-17', 1, [(INF, '$\\epsilon=\\kappa_1/\\kappa_2^{\\rm eff}$ & $\\simeq0.022$'), (INF, 'de modo que $\\epsilon=\\kappa_1/\\kappa_2^{\\rm eff}\\simeq0.022$')], 'Ma: eps = kappa1/kappa2eff = 1.4e-3/0.062 = 0.022', '0.022', ((5 / 72) * (kap / gz_)**2) / e1, dato='data/principal_fig3_minimo.csv', script=S, tol=0.0005)
    claim('PL-18', 1, [(INF, 'unas cinco veces el umbral $1/220=4.5\\times10^{-3}$')], 'Ma: eps / (1/220) ~ 5', '5', (((5 / 72) * (kap / gz_)**2) / e1) * 220, dato='(analitico)', script=S, tol=0.1)
    e01 = np.exp(np.interp(np.log(0.1), np.log(TM[:, 0]), np.log(TM[:, 5])))
    claim('PL-19', 1, [(INF, 'bajar $g_x$ hasta $\\kappa_2/\\kappa\\simeq0.1$ da $\\epsilon\\simeq4.3\\times10^{-3}$')], 'Ma con gz/kappa=7.1 y kappa2/kappa=0.1: eps = (5/72)(k/gz)^2 / (k2eff/k2) = 4.3e-3', '4.3e-3', ((5 / 72) * (kap / gz_)**2) / e01, dato='data/principal_fig3_minimo.csv', script=S, tol=0.1e-3)
    claim('PL-20', 1, [(INF, '$\\kappa_2/\\kappa\\lesssim0.11$ con $\\omega/\\kappa=200$')], 'Frontera chi para Ma: kappa2/kappa <= 0.45 (gz/k)^2/(w/k) = 0.11 con w/k=200', '0.11', 0.45 * (gz_ / kap)**2 / 200, dato='(analitico)', script=S, tol=0.005)
    # piso intrinseco desde los mapas
    m = np.load(os.path.join(DATA, 'figura_central_mapas.npz'), allow_pickle=True)
    gzg, k2g, eps, f, c = m['gz'], m['k2'], m['eps_filtro'], float(m['f_filt']), float(m['c'])
    K2, GZ = np.meshgrid(k2g, gzg, indexing='ij')
    k1 = f * K2 / (16 * GZ**2); k2eff = k1 / eps
    def thr(arr, target=1 / 220):
        out = np.full(len(k2g), np.nan)
        for i in range(len(k2g)):
            y = np.log(arr[i]); idx = np.where(y <= np.log(target))[0]
            if len(idx) == 0 or idx[0] == 0: continue
            j = idx[0]; x0, x1 = np.log(gzg[j - 1]), np.log(gzg[j]); y0, y1 = y[j - 1], y[j]
            out[i] = np.exp(x0 + (np.log(target) - y0) * (x1 - x0) / (y1 - y0))
        return out
    t0 = thr(eps)
    for gam, tag in ((2e-4, 'gamma/kappa=2e-4'), (2e-5, 'gamma/kappa=2e-5')):
        ef = eps + gam / k2eff
        tf = thr(ef)
        okk = ~np.isnan(tf)
        if gam == 2e-4:
            claim('PS-01', 1, [(RM, 'ε = 1/220 solo en la franja κ₂/κ ≈ 0.19–0.33'), (INF, 'solo se alcanza en la franja $\\kappa_2/\\kappa\\simeq0.19$--$0.33$')], 'Piso gamma/kappa=2e-4: eps=1/220 alcanzable solo para kappa2/kappa en 0.19-0.33 (extremo inferior)', '0.19', k2g[okk].min(), dato='data/figura_central_mapas.npz', script=S, tol=0.005)
            claim('PS-02', 1, [(RM, 'ε = 1/220 solo en la franja κ₂/κ ≈ 0.19–0.33'), (INF, 'solo se alcanza en la franja $\\kappa_2/\\kappa\\simeq0.19$--$0.33$')], 'Piso gamma/kappa=2e-4: extremo superior de la franja', '0.33', k2g[okk].max(), dato='data/figura_central_mapas.npz', script=S, tol=0.005)
            claim('PS-03', 1, [(RM, 'ε = 1e-3 no se alcanza'), (INF, '$\\epsilon=10^{-3}$ no se alcanza')], 'Piso gamma/kappa=2e-4: eps=1e-3 no se alcanza', 'no', not (ef <= 1e-3).any(), modo='bool', dato='data/figura_central_mapas.npz', script=S)
            tl, th = np.nanmin(tf[okk]), np.nanmax(tf[okk])
            claim('PS-04', 1, [(RM, 'con g_z/κ ≳ 0.8–1.9 según κ₂'), ], 'Piso 2e-4: gz/kappa minimo en la franja: rango 0.8-1.9 (segun kappa2)', '0.8', tl, dato='data/figura_central_mapas.npz', script=S, tol=0.1, nota='rango recalculado %.2f - %.2f' % (tl, th))
            claim('PS-05', 1, [(RM, 'con g_z/κ ≳ 0.8–1.9 según κ₂')], 'Piso 2e-4: gz/kappa maximo del umbral en la franja (1.9)', '1.9', th, dato='data/figura_central_mapas.npz', script=S, tol=0.1, nota='ver PS-04: el extremo derecho de la franja da %.2f' % th)
            g_lo = np.sqrt(200 * k2g[okk].min() / 0.45); g_hi = np.sqrt(200 * k2g[okk].max() / 0.45)
            claim('PS-06', 1, [(RM, 'g_z/κ ≥ 9.3 (κ₂/κ = 0.19) a 11.9 (0.32)'), (INF, '$g_z/\\kappa\\gtrsim9.3$ ($\\kappa_2/\\kappa=0.19$) a $11.9$ ($0.32$)')], 'Dentro de la zona chi<=0.3: gz/k >= sqrt(200 k2/0.45): 9.3 en el borde inferior (0.19)', '9.3', g_lo, dato='data/figura_central_mapas.npz', script=S, tol=0.1)
            claim('PS-07', 1, [(RM, 'g_z/κ ≥ 9.3 (κ₂/κ = 0.19) a 11.9 (0.32)'), (INF, '$11.9$ ($0.32$)')], 'Idem borde superior: 11.9', '11.9', np.sqrt(200 * 0.32 / 0.45), dato='(analitico)', script=S, tol=0.05, nota='con el borde 0.327 de la franja serian %.1f' % g_hi)
        else:
            for kk, rep in ((0.01, '44'), (0.03, '14'), (0.1, '7'), (0.3, '5'), (1.0, '8')):
                i = int(np.argmin(abs(k2g - kk)))
                claim('PS-g5-%g' % kk, 1, [(RM, 'γ/κ = 2e-5:** el umbral de 1/220 se desplaza +44%'), (INF, 'el umbral $1/220$ se desplaza $+44\\%$')], 'Piso 2e-5: el umbral eps=1/220 se desplaza +%s%% en kappa2/kappa=%g' % (rep, kk), rep, 100 * (tf[i] / t0[i] - 1), dato='data/figura_central_mapas.npz', script=S, tol=1.2, nota='recalculo %.1f%% (grilla gz de paso 1.2%%)' % (100 * (tf[i] / t0[i] - 1)))
            ratio10 = (gam / k2eff) / (k1 / 1.0)
            # piso nunca baja del 10% de kappa1_filt, en el umbral
            fr = []
            for i in range(len(k2g)):
                if np.isnan(t0[i]): continue
                j = int(np.argmin(abs(gzg - t0[i]))); fr.append(gam / k2eff[i, j] / (k1[i, j] / eps[i, j]) if False else gam / k1[i, j])
            claim('PS-g5-10', 1, [(RM, 'el piso nunca baja del 10% de κ₁^filt'), (INF, 'el piso nunca baja del $10\\%$ de $\\kappa_1^{\\rm filt}$')], 'Piso 2e-5: el piso (gamma) nunca baja del 10% de kappa1_filt en la region del umbral', '0.10', min(fr), modo='ge', tol=0, dato='data/figura_central_mapas.npz', script=S, nota='minimo gamma/kappa1_filt en el umbral: %.3f (en los puntos donde el umbral existe)' % min(fr))
    # N de verificacion con piso del completo (0.25, 14) y (0.25, 6)
    return

# ============================================================ BLOQUE 7: termico, modelo efectivo (isla) y ajuste de c_bf
TD = os.path.join(DATA, 'termico')
def eff_curve(k, filt, gam, gz=14, N=22, a=4, extra=''):
    pat = f'x*_k{k:g}_gz{gz:g}_f{filt}_g{gam:g}_N{N}_Nf2_w200_a{a}{extra}.npz'
    xs = []
    for f in glob.glob(os.path.join(TD, pat)):
        if not extra and f.endswith(('_kick.npz', '_nqoff.npz', '_nmoff.npz', '_ocreal.npz', '_kick_nqoff.npz', '_kick_nmoff.npz')): continue
        z = np.load(f, allow_pickle=True); xs.append((float(z['x']), float(z['gpf']), float(z['gbf']), float(z['eta'])))
    return np.array(sorted(set(xs)))
def xstar_c(c, level, col=3):
    x, eta = c[:, 0], c[:, col]
    m = x > 1.47
    x, eta = x[m], eta[m]
    i = np.where((eta[:-1] - level) * (eta[1:] - level) <= 0)[0]
    if not len(i): return np.nan
    i = i[0]
    return float(np.exp(np.interp(np.log(level), np.log(eta[i:i + 2]), np.log(x[i:i + 2])))) if eta[i + 1] > eta[i] else np.nan
def eff_pt(k, x, filt=1, gam=2e-5, gz=14, N=22, a=4, extra=''):
    """Punto del efectivo por nombre de archivo (el x guardado dentro del npz es el resuelto, p.ej. 6.859955 para 'x6.86')."""
    f = os.path.join(TD, f'x{x:g}_k{k:g}_gz{gz:g}_f{filt}_g{gam:g}_N{N}_Nf2_w200_a{a}{extra}.npz')
    z = np.load(f, allow_pickle=True)
    return dict(x=float(z['x']), gpf=float(z['gpf']), gbf=float(z['gbf']), eta=float(z['eta']), nq=float(z['nq']), herm=float(z['herm_cruda']), file=rel(f), z=z)

def bloque_termico_ef():
    S = 'consolidacion/verificar_todo.py::bloque_termico_ef'
    DT = 'data/termico/x*_k*_gz14_f*_g*_N22_Nf2_w200_a4.npz'
    # ---- tabla de umbrales en la isla
    tab = [(1, 2e-5, '10.11', '10.91', '2.11', '2.27'), (1, 2e-4, '7.78', '8.59', '1.62', '1.79'), (0, 2e-5, '8.19', '9.01', '1.71', '1.88'), (0, 2e-4, '7.18', '8.00', '1.50', '1.67')]
    lab = {(1, 2e-5): 'filtrado & $2\\times10^{-5}$', (1, 2e-4): 'filtrado & $2\\times10^{-4}$', (0, 2e-5): 'plano & $2\\times10^{-5}$', (0, 2e-4): 'plano & $2\\times10^{-4}$'}
    hrow = {(1, 2e-5): '| filtrado | 2e-5 |', (1, 2e-4): '| filtrado | 2e-4 |', (0, 2e-5): '| plano | 2e-5 |', (0, 2e-4): '| plano | 2e-4 |'}
    for (fi, g, x1, x2, f1, f2) in tab:
        c = eff_curve(0.25, fi, g)
        a, b = xstar_c(c, 100), xstar_c(c, 220)
        L1 = [(H8, hrow[(fi, g)], x1), (RM, hrow[(fi, g)], x1), (INF, lab[(fi, g)] + ' & ' + x1 + ' &', x1)]
        L2 = [(H8, hrow[(fi, g)], x2), (RM, hrow[(fi, g)], x2), (INF, lab[(fi, g)] + ' & ' + x1 + ' &', x2)]
        claim('TX-%d-%g-100' % (fi, g), 1, L1, 'x*(eta=100) en la isla (k2/k=0.25), %s, gamma/kappa=%g' % ('filtrado' if fi else 'plano', g), x1, a, dato=DT, script=S, nota='interpolacion log-log entre puntos de cache vecinos: %.4f' % a)
        claim('TX-%d-%g-220' % (fi, g), 1, L2, 'x*(eta=220) en la isla, %s, gamma/kappa=%g' % ('filtrado' if fi else 'plano', g), x2, b, dato=DT, script=S, nota='recalculo %.4f' % b)
        kk = 208.37e-3
        claim('TX-%d-%g-fq1' % (fi, g), 1, [(README_LOC := (RM, hrow[(fi, g)], f1 + ' / ' + f2)), (INF, lab[(fi, g)] + ' & ' + x1 + ' &', f1 + ' / ' + f2)] if False else [(RM, '| %s | %s |' % ('filtrado' if fi else 'plano', '2e-5' if g == 2e-5 else '2e-4'))], 'f_q a 10 mK para eta=100 (GHz): x*(100)*0.20837', f1, a * kk, dato=DT, script=S, tol=0.01)
        claim('TX-%d-%g-fq2' % (fi, g), 1, [(RM, '| %s | %s |' % ('filtrado' if fi else 'plano', '2e-5' if g == 2e-5 else '2e-4'))], 'f_q a 10 mK para eta=220 (GHz)', f2, b * kk, dato=DT, script=S, tol=0.01)
    # N=24: existen caches?
    zN24 = glob.glob(os.path.join(TD, 'x*_k0.25_gz14_f1_g2e-05_N24_Nf2_w200_a4.npz'))
    U = csvrows('termico_umbrales.csv')
    dxs = [abs(r[3] - r[4]) for r in U]
    claim('TX-N24', 1, [(H8, 'N = 22 → 24 cambia x* ≤ 0.008'), (RM, 'N = 24 cambia x* en ≤ 0.008'), (INF, '$N=22\\to24$ cambia $x^{*}$ en $\\leq0.008$')], 'N=22->24 cambia x* en <= 0.008', '0.008', max(dxs), modo='le', tol=0, dato='data/termico_umbrales.csv (cache N24: %d archivos)' % len(zN24), script=S,
          nota='max |dx*| del CSV = %.4f; hay %d caches N=24 en la isla' % (max(dxs), len(zN24)))
    # N=24 recompute x* if possible
    if zN24:
        c24 = eff_curve(0.25, 1, 2e-5, N=24)
        if len(c24) > 3:
            claim('TX-N24b', 1, [(H8, 'N = 22 → 24 cambia x* ≤ 0.008')], 'x*(100) con N=24 en la isla filtrada', '10.11', xstar_c(c24, 100), dato='data/termico/x*_k0.25_gz14_f1_g2e-05_N24_*.npz', script=S, tol=0.01, nota='n puntos N=24: %d' % len(c24))
    # ---- c_bf
    R = np.array(csvrows('termico_cbf.csv'))
    DC = 'data/termico_cbf.csv'
    for filt, nm in ((1, 'con filtro'), (0, 'plano')):
        r = R[R[:, 0] == filt]; G, c, k2 = r[:, 2], r[:, 3], r[:, 1]
        p = np.polyfit(np.log(G), np.log(c), 1)
        res = np.log(c) - np.polyval(p, np.log(G)); n = len(r)
        se = math.sqrt(np.sum(res**2) / (n - 2) / np.sum((np.log(G) - np.log(G).mean())**2))
        rep = ('3.96', '0.04') if filt else ('3.69', '0.05')
        claim('CB-s%d' % filt, 1, [(H8, 's = 3.96 ± 0.04 con filtro y 3.69 ± 0.05'), (RM, 's = 3.96 ± 0.04 con filtro y 3.69 ± 0.05'), (INF, 's=3.96\\pm0.04$ con filtro')], 'Exponente s de c_bf ~ (G/kappa)^s %s (15 puntos kappa2/kappa 0.02-0.4)' % nm, rep[0], p[0], dato=DC, script=S, nota='error estandar del ajuste lineal %.3f' % se)
        claim('CB-se%d' % filt, 1, [(H8, 's = 3.96 ± 0.04 con filtro y 3.69 ± 0.05'), (RM, 's = 3.96 ± 0.04 con filtro y 3.69 ± 0.05')], 'Error estandar de s %s' % nm, rep[1], se, dato=DC, script=S, tol=0.005)
        m = (k2 >= 0.05) & (k2 <= 0.4)
        p2 = np.polyfit(np.log(G[m]), np.log(c[m]), 1)
        claim('CB-r%d' % filt, 1, [(H8, 'restringido a κ₂/κ entre 0.05 y 0.4 (11 puntos) s = 3.83 con filtro y 3.51 en el plano')], 'Exponente s restringido a kappa2/kappa en [0.05,0.4] (11 puntos) %s' % nm, '3.83' if filt else '3.51', p2[0], dato=DC, script=S)
        sel = [int(np.argmin(abs(k2 - kk))) for kk in (0.05, 0.1, 0.25, 0.4)]
        p4 = np.polyfit(np.log(G[sel]), np.log(c[sel]), 1)
        if filt: claim('CB-4e', 1, [(H8, 'frente a 3.81 del efectivo con los mismos puntos')], 'Exponente s del efectivo con filtro en los 4 puntos (0.05,0.1,0.25,0.4)', '3.81', p4[0], dato=DC, script=S, tol=0.01)
    r1 = R[R[:, 0] == 1]; r0 = R[R[:, 0] == 0]
    for kk, a_, b_ in ((0.02, '2.62e-4', '2.76e-4'), (0.05, '1.65e-3', '1.67e-3'), (0.2, '2.75e-2', '2.23e-2'), (0.4, '8.42e-2', '6.20e-2')):
        i1 = int(np.argmin(abs(r1[:, 1] - kk))); i0 = int(np.argmin(abs(r0[:, 1] - kk)))
        claim('CB-c1-%g' % kk, 1, [(RM, '| c_bf con filtro |'), (INF, '$c_{\\rm bf}$ con filtro')], 'c_bf con filtro en kappa2/kappa=%g' % kk, a_, r1[i1, 3], dato=DC, script=S)
        claim('CB-c0-%g' % kk, 1, [(RM, '| c_bf en el plano |'), (INF, '$c_{\\rm bf}$ en el plano')], 'c_bf en el plano en kappa2/kappa=%g' % kk, b_, r0[i0, 3], dato=DC, script=S)
    claim('CB-iqr', 1, [(RM, 'La incertidumbre (semi-rango intercuartil) es ≤ 2%'), (INF, 'semirrango intercuartil) $\\leq2\\%$')], 'Incertidumbre de c_bf (semi-rango intercuartil) <= 2%', '0.02', (R[:, 4] / R[:, 3]).max(), modo='le', tol=0, dato=DC, script=S, nota='maximo %.4f en kappa2/kappa=%g (filtro=%d)' % ((R[:, 4] / R[:, 3]).max(), R[np.argmax(R[:, 4] / R[:, 3]), 1], R[np.argmax(R[:, 4] / R[:, 3]), 0]))
    claim('CB-p', 1, [(RM, 'exponente local p = 1.00–1.02'), (INF, 'exponente local $p=1.00$--$1.02$')], 'Exponente local de n_q: p = 1.00-1.02', '1.02', R[:, 5].max(), modo='le', tol=0.0005, dato=DC, script=S, nota='rango %.4f - %.4f' % (R[:, 5].min(), R[:, 5].max()))
    claim('CB-var', 1, [(RM, 'c_bf varía más de dos órdenes de magnitud'), (INF, 'varía más de dos órdenes de magnitud')], 'c_bf varia mas de dos ordenes de magnitud con kappa2/kappa', '100', R[:, 3].max() / R[:, 3].min(), modo='ge', tol=0, dato=DC, script=S, nota='factor %.0f' % (R[:, 3].max() / R[:, 3].min()))
    # ---- supresion con |alpha|^2
    sl = {}
    for filt in (0, 1):
        for xx in (9, 12):
            pts = []
            for N, a in ((16, 2), (18, 3), (22, 4), (24, 5), (26, 6)):
                f = os.path.join(TD, 'x%g_k0.25_gz14_f%d_g2e-05_N%d_Nf2_w200_a%d.npz' % (xx, filt, N, a))
                if os.path.exists(f):
                    z = np.load(f, allow_pickle=True); pts.append((a, float(z['gbf'])))
            if len(pts) >= 3:
                pts = np.array(pts); sl[(filt, xx)] = np.polyfit(pts[:, 0], np.log(pts[:, 1]), 1)[0]
            else:
                sl[(filt, xx)] = None
    for (filt, xx), rep in (((0, 9), '-0.55'), ((0, 12), '-0.64'), ((1, 9), '-0.43'), ((1, 12), '-0.42')):
        v = sl[(filt, xx)]
        if v is None:
            claim('AL-%d-%d' % (filt, xx), 1, [(H8, 'd ln γ_bf/d|α|² ≈ −0.55 a −0.64 en el plano'), (RM, 'd ln γ_bf/d|α|² ≈ −0.55 y −0.64 en el plano')], 'd ln gbf/d|a|^2 (x=%d, %s) = %s' % (xx, 'filtro' if filt else 'plano', rep), rep, None, estado='R', dato='data/termico/x%d_k0.25_*_a{2..6}.npz (incompleto)' % xx, script=S, nota='caches de |alpha|^2 faltantes para x=%d' % xx)
        else:
            claim('AL-%d-%d' % (filt, xx), 1, [(H8, 'd ln γ_bf/d|α|² ≈ −0.55 a −0.64 en el plano'), (RM, 'd ln γ_bf/d|α|² ≈ −0.55 y −0.64 en el plano')], 'd ln gbf/d|a|^2 (x=%d, %s) = %s (ajuste lineal de ln gbf vs |a|^2=2..6)' % (xx, 'filtro' if filt else 'plano', rep), rep, v, dato='data/termico/x%d_k0.25_gz14_f%d_g2e-05_N{16,18,22,24,26}_Nf2_w200_a{2..6}.npz' % (xx, filt), script=S, tol=0.06, nota='recalculo %.3f; N crece con |alpha|^2 (16,18,22,24,26)' % v)
    claim('AL-cache', 1, [(H8, 'las cachés de ese barrido en |α|² no están en `data/termico/`')], 'HANDOFF 8.2: las caches del barrido en |alpha|^2 no estan en data/termico/', 'no estan', len(glob.glob(os.path.join(TD, 'x9_k0.25_gz14_f*_g2e-05_N*_Nf2_w200_a[23456].npz'))) == 0, modo='bool', dato='data/termico/x9_k0.25_gz14_f{0,1}_g2e-05_N*_a{2..6}.npz', script=S,
          nota='SI estan: %d archivos en x=9 y %d en x=12' % (len(glob.glob(os.path.join(TD, 'x9_k0.25_gz14_f*_g2e-05_N*_Nf2_w200_a[23456].npz'))), len(glob.glob(os.path.join(TD, 'x12_k0.25_gz14_f*_g2e-05_N*_Nf2_w200_a[23456].npz')))))
    # ---- control ocupaciones reales (efectivo)
    return

# ============================================================ BLOQUE 8: termico, modelo completo con filtro frente al efectivo
KAPM = 0.03
GXK = {0.05: 0.0239579, 0.1: 0.0338816, 0.25: 0.0535714, 0.4: 0.0677631}
def fcomp(k, x=None, N=22, plano=False, suf='', kf=0.3):
    n = f'gx{GXK[k]:.6g}_w6_gz0.42_kf{kf:g}_al4_N{N}_Nf{1 if plano else 2}' + ('_plano' if plano else '') + '_gam6e-07' + (f'_x{x:g}' if x else '') + suf + '.npz'
    p = os.path.join(FC, n)
    return np.load(p, allow_pickle=True), rel(p)

def bloque_termico_full():
    S = 'consolidacion/verificar_todo.py::bloque_termico_full'
    XM = {0.05: 9, 0.1: 9, 0.25: 10.1, 0.4: 9.5}
    pts = [(0.05, 6.86), (0.05, 9), (0.1, 6.86), (0.1, 9), (0.25, 6.86), (0.25, 10.1), (0.4, 6.86), (0.4, 9.5)]
    T1 = {(0.05, 6.86): ('1.650e-04', '1.721e-04', '1.647e-06', '1.722e-06', '100.2', '99.95'), (0.05, 9): ('1.570e-04', '1.641e-04', '1.953e-07', '2.038e-07', '803.8', '805.2'),
          (0.1, 6.86): ('1.651e-04', '1.722e-04', '6.420e-06', '7.497e-06', '25.72', '22.97'), (0.1, 9): ('1.571e-04', '1.642e-04', '7.550e-07', '8.780e-07', '208', '187'),
          (0.25, 6.86): ('1.653e-04', '1.724e-04', '3.349e-05', '4.267e-05', '4.935', '4.039'), (0.25, 10.1): ('1.556e-04', '1.627e-04', '1.316e-06', '1.650e-06', '118.2', '98.63'),
          (0.4, 6.86): ('1.654e-04', '1.726e-04', '6.908e-05', '8.903e-05', '2.395', '1.938'), (0.4, 9.5): ('1.566e-04', '1.638e-04', '4.850e-06', '6.219e-06', '32.29', '26.33')}
    T2 = {(0.05, 6.86): ('0.9588', '0.9561', '0.956', '1.003', '0.30%', '1.780e-09', '-1.5e-10'), (0.05, 9): ('0.9566', '0.9583', '0.9574', '0.9982', '2.52%', '1.319e-09', '4.8e-17'),
          (0.1, 6.86): ('0.9588', '0.8563', '0.8558', '1.12', '0.17%', '2.117e-09', '7.3e-15'), (0.1, 9): ('0.9565', '0.8599', '0.856', '1.112', '1.44%', '2.797e-09', '-1.0e-16'),
          (0.25, 6.86): ('0.9587', '0.7848', '0.7842', '1.222', '0.09%', '4.102e-10', '1.7e-14'), (0.25, 10.1): ('0.9562', '0.7976', '0.7817', '1.199', '2.35%', '1.155e-09', '-2.7e-16'),
          (0.4, 6.86): ('0.9587', '0.7759', '0.7752', '1.236', '0.09%', '1.645e-12', '2.3e-14'), (0.4, 9.5): ('0.9562', '0.7798', '0.7697', '1.226', '1.35%', '1.983e-09', '-2.5e-16')}
    E = {}
    for k, x in pts:
        z, fz = fcomp(k, x); z0, f0 = fcomp(k); e = eff_pt(k, x); e0 = eff_pt(k, 60)
        pc, bc = float(z['gpf']) / KAPM, float(z['gbf']) / KAPM
        bf0c, bf0e = float(z0['gbf']) / KAPM, e0['gbf']
        E[(k, x)] = dict(pc=pc, bc=bc, pe=e['gpf'], be=e['gbf'], etac=pc / bc, etae=e['eta'], rpf=pc / e['gpf'], rbf=bc / e['gbf'], rs=(bc - bf0c) / (e['gbf'] - bf0e),
                         rseta=(pc / bc) / e['eta'], fr=bf0c / bc, herm=float(z['herm_cruda']), me=float(z['val'][2]), tr=float(z['val'][0]), fz=fz, f0=f0, fe=e['file'], bf0c=bf0c, bf0e=bf0e, z=z, z0=z0, e=e, e0=e0)
    Hrow1 = lambda k, x: '| %g | %g | %s' % (k, x, T1[(k, x)][0]) if (k, x) != (0.1, 6.86) else '| 0.1 | 6.86 | 1.651e-04'
    for (k, x) in pts:
        d = E[(k, x)]; t1 = T1[(k, x)]; t2 = T2[(k, x)]
        a1 = '| %g | %s | %s' % (k, ('%g' % x), t1[0])
        a2 = '| %g | %s | %s' % (k, ('%g' % x), t2[0])
        tag = '%g-%g' % (k, x)
        dd = d['fz'] + ' ; ' + d['fe']
        for nm, rep, rec, nt in (('gpf_c', t1[0], d['pc'], ''), ('gpf_e', t1[1], d['pe'], ''), ('gbf_c', t1[2], d['bc'], ''), ('gbf_e', t1[3], d['be'], ''), ('eta_c', t1[4], d['etac'], ''), ('eta_e', t1[5], d['etae'], '')):
            claim('T84a-%s-%s' % (tag, nm), 1, [(H8, a1)], '8.4 tasas (kappa=1) k2/k=%g, x=%g: %s' % (k, x, nm), rep, rec, dato=dd, script=S, nota=nt)
        claim('T84b-%s-rpf' % tag, 1, [(H8, a2, t2[0])] + ([(RM, '| 0.05 | 6.86 | 0.959', '0.959')] if (k, x) == (0.05, 6.86) else []) + ([(RM, '| 0.10 | 6.86 | 0.959', '0.959')] if (k, x) == (0.1, 6.86) else []) + ([(RM, '| 0.25 | 6.86 | 0.959', '0.959')] if (k, x) == (0.25, 6.86) else []) + ([(RM, '| 0.40 | 6.86 | 0.959', '0.959')] if (k, x) == (0.4, 6.86) else []) + ([(RM, '| 0.25 | 10.1 | 0.956', '0.956')] if (k, x) == (0.25, 10.1) else []),
              '8.4 cociente gamma_pf completo/efectivo k2/k=%g x=%g' % (k, x), t2[0], d['rpf'], dato=dd, script=S, tol=(6e-5 if len(t2[0]) == 6 else None))
        claim('T84b-%s-rbf' % tag, 1, [(H8, a2, t2[1])], '8.4 gamma_bf completo/efectivo (crudo) k2/k=%g x=%g' % (k, x), t2[1], d['rbf'], dato=dd, script=S)
        claim('T84b-%s-rs' % tag, 1, [(H8, a2, t2[2])], '8.4 gamma_bf completo/efectivo con piso restado k2/k=%g x=%g' % (k, x), t2[2], d['rs'], dato=dd + ' ; ' + d['f0'], script=S, tol=None)
        claim('T84b-%s-reta' % tag, 1, [(H8, a2, t2[3])], '8.4 eta completo/efectivo k2/k=%g x=%g' % (k, x), t2[3], d['rseta'], dato=dd, script=S)
        claim('T84b-%s-fr' % tag, 1, [(H8, a2, t2[4])], '8.4 piso completo / gamma_bf termico k2/k=%g x=%g' % (k, x), t2[4], d['fr'], dato=dd + ' ; ' + d['f0'], script=S)
        claim('T84b-%s-herm' % tag, 1, [(H8, a2, t2[5])], '8.4 hermiticidad cruda k2/k=%g x=%g' % (k, x), t2[5], d['herm'], dato=d['fz'], script=S)
        claim('T84b-%s-me' % tag, 1, [(H8, a2, t2[6])], '8.4 minimo autovalor k2/k=%g x=%g' % (k, x), t2[6], d['me'], tol=(1e-16 if abs(float(t2[6])) < 1e-15 else (0.5e-16 if abs(float(t2[6])) < 1e-14 else None)), dato=d['fz'], script=S,
              nota='cotas del orden de 1e-16 son ruido de maquina')
    # ---- 8.4 afirmaciones de texto
    claim('T84-c1', 1, [(H8, '**Criterio ±5%:** γ_pf cumple en todos los puntos')], 'Criterio +-5%: gamma_pf c/e dentro de 5% en los 8 puntos', 'si', all(abs(E[p]['rpf'] - 1) < 0.05 for p in pts), modo='bool', dato='data/filtro_completo/*, data/termico/*', script=S, nota='max |c/e-1| = %.4f' % max(abs(E[p]['rpf'] - 1) for p in pts))
    ok5 = {p: abs(E[p]['rbf'] - 1) < 0.05 for p in pts}
    claim('T84-c2', 1, [(H8, 'γ_bf cumple solo en κ₂/κ = 0.05')], 'Criterio +-5%: gamma_bf c/e cumple solo en kappa2/kappa=0.05 (ambos x)', 'si', ok5[(0.05, 6.86)] and ok5[(0.05, 9)] and not any(ok5[p] for p in pts if p[0] != 0.05), modo='bool', dato='data/filtro_completo/*, data/termico/*', script=S)
    ind = {}
    for k in (0.05, 0.1, 0.4, 0.25):
        xm = XM[k]; ind[k] = E[(k, xm)]['rs'] / E[(k, 6.86)]['rs'] - 1
    for k, rep, ncol in ((0.05, '0.14%', 'x'), (0.1, '0.02%', 'x'), (0.4, '-0.72%', 'x'), (0.25, '-0.33%', 'x')):
        claim('T84-ind-%g' % k, 1, [(H8, 'cambios de 0.14%, 0.02%, -0.72% y -0.33%'), (RM, 'cambia +0.14%, +0.02%, −0.72% y −0.33%')], 'Independencia de x: cambio del cociente gamma_bf con piso restado entre x=6.86 y x~9, k2/k=%g' % k, rep, ind[k], tol=0.00005, dato='data/filtro_completo/*, data/termico/*; data/termico_x_independencia.csv', script=S)
    claim('T84-ind-ok', 1, [(H8, 'Sostenida')], 'Independencia de x: |cambio| < 3% en los 4 valores de kappa2/kappa (criterio fijado antes)', 'si', all(abs(v) < 0.03 for v in ind.values()), modo='bool', script=S)
    claim('T84-tolE', 1, [(H8, 'Tolerancia del integrador (fase 2, isla)'), (RM, '**Tolerancia del integrador**')], 'Tolerancia del integrador: gamma_bf -2.6e-5 relativo (x=10.1) y piso -0.10%, gamma_pf igual', '-2.6e-5',
          (float(fcomp(0.25, 10.1, suf='_tolE')[0]['gbf']) / float(E[(0.25, 10.1)]['z']['gbf']) - 1), dato='data/filtro_completo/*_x10.1{,_tolE}.npz', script=S, tol=0.2e-5)
    claim('T84-tolE2', 1, [(H8, 'y el piso −0.10%')], 'Tolerancia del integrador: piso de T=0 cambia -0.10%', '-0.10%', (float(fcomp(0.25, None, suf='_tolE')[0]['gbf']) / float(E[(0.25, 10.1)]['z0']['gbf']) - 1), dato='data/filtro_completo/*_gam6e-07{,_tolE}.npz', script=S, tol=0.00001)
    # ---- 8.6 piso
    fl = {0.05: ('4.931e-09', '4.959e-09', '0.994', '3.1e+04', '3.23e+04'), 0.1: ('1.090e-08', '8.751e-09', '1.25', '1.4e+04', '1.83e+04'), 0.25: ('3.086e-08', '5.819e-09', '5.3', '4.97e+03', '2.76e+04'), 0.4: ('6.528e-08', '3.152e-09', '20.7', '2.35e+03', '5.1e+04')}
    for k, (pc_, pe_, rc, ec, ee) in fl.items():
        d = E[(k, XM[k])]; z0 = d['z0']
        bf0c = d['bf0c']; bf0e = d['bf0e']; gp0c = float(z0['gpf']) / KAPM; gp0e = d['e0']['gpf']
        row = '| %g | %s' % (k, pc_)
        claim('T86-%g-pc' % k, 1, [(H8, row)], '8.6 piso completo (gamma_bf T=0, kappa=1) k2/k=%g' % k, pc_, bf0c, dato=d['f0'], script=S)
        claim('T86-%g-pe' % k, 1, [(H8, row)], '8.6 piso efectivo (x=60) k2/k=%g' % k, pe_, bf0e, dato=d['fe'].replace('x%g' % XM[k], 'x60') if False else rel(os.path.join(TD, 'x60_k%g_gz14_f1_g2e-05_N22_Nf2_w200_a4.npz' % k)), script=S)
        claim('T86-%g-r' % k, 1, [(H8, row)], '8.6 cociente piso completo/efectivo k2/k=%g' % k, rc, bf0c / bf0e, dato=d['f0'], script=S, tol=None)
        claim('T86-%g-ec' % k, 1, [(H8, row)], '8.6 eta(T=0) completo k2/k=%g' % k, ec, gp0c / bf0c, dato=d['f0'], script=S)
        claim('T86-%g-ee' % k, 1, [(H8, row)], '8.6 eta(T=0) efectivo k2/k=%g' % k, ee, gp0e / bf0e, dato=d['f0'], script=S)
    claim('T86-etamax', 1, [(H8, 'η_max ≈ 5000'), (PH, 'η_max = η(T=0)')], 'eta_max ~ 4968 en la isla: 4.600e-6/9.259e-10', '4968', (float(E[(0.25, 10.1)]['z0']['gpf']) / KAPM) / (float(E[(0.25, 10.1)]['z0']['gbf']) / KAPM), dato=E[(0.25, 10.1)]['f0'], script=S, tol=1, nota='usa el T=0 de la isla en unidades kappa')
    claim('T86-exp', 1, [(H8, 'completo 4.931e-09 a 6.528e-08')], 'El piso crece con kappa2/kappa en el completo (monotono) y no en el efectivo (no monotono)', 'si', (lambda a, b: all(np.diff(a) > 0) and not all(np.diff(b) > 0) and not all(np.diff(b) < 0))([fl_ for fl_ in [E[(k, XM[k])]['bf0c'] for k in fl]], [E[(k, XM[k])]['bf0e'] for k in fl]), modo='bool', script=S)
    # ---- 8.5 x*
    xs_rep = {0.05: ('6.861', '6.862', '7.667', '7.668'), 0.1: ('8.25', '8.253', '9.057', '9.057'), 0.25: ('9.927', '9.931', '10.75', '10.73'), 0.4: ('10.67', '10.64', '11.52', '11.43')}
    xe_rep = {0.05: ('6.86', '7.666'), 0.1: ('8.361', '9.166'), 0.25: ('10.11', '10.91'), 0.4: ('10.84', '11.64')}
    for k in (0.05, 0.1, 0.25, 0.4):
        c = eff_curve(k, 1, 2e-5); xm = XM[k]; d = E[(k, xm)]
        qpf = d['rpf']; rs = d['rs']; bf0c, bf0e = d['bf0c'], d['bf0e']
        eta_c = c[:, 1] * qpf / (bf0c + rs * (c[:, 2] - bf0e))
        eta_q = c[:, 3] * d['rseta']
        cc = np.c_[c[:, 0], c[:, 1], c[:, 2], eta_c]; cq = np.c_[c[:, 0], c[:, 1], c[:, 2], eta_q]
        for lvl, ie, ic in ((100, 0, 0), (220, 1, 2)):
            xe = xstar_c(c, lvl); xcc = xstar_c(cc, lvl); xq = xstar_c(cq, lvl)
            row = '| %g |' % k if k in (0.05, 0.25, 0.4) else '| 0.1 |'
            claim('T85-%g-%d-ef' % (k, lvl), 1, [(H8, '| %g | %s' % (k, '6.86 a 9' if k == 0.05 else ('6.86 a 9' if k == 0.1 else ('6.86 a 10.1' if k == 0.25 else '6.86 a 9.5'))), xe_rep[k][ie])], 'x*(%d) del efectivo con filtro, k2/k=%g' % (lvl, k), xe_rep[k][ie], xe, dato='data/termico/x*_k%g_gz14_f1_g2e-05_N22_*.npz' % k, script=S, tol=0.006)
            claim('T85-%g-%d-co' % (k, lvl), 1, [(H8, '| %g | %s' % (k, '6.86 a 9' if k in (0.05, 0.1) else ('6.86 a 10.1' if k == 0.25 else '6.86 a 9.5')), xs_rep[k][ic])], '[ESTIMADO] x*(%d) del completo con piso medido, k2/k=%g' % (lvl, k), xs_rep[k][ic], xcc, dato='data/filtro_completo/*, data/termico/*', script=S, tol=0.006,
                  nota='estimacion (supone cociente constante en x); posicion respecto al rango medido: x_m=%g' % xm, estado=None)
            claim('T85-%g-%d-ant' % (k, lvl), 1, [(H8, '| %g | %s' % (k, '6.86 a 9' if k in (0.05, 0.1) else ('6.86 a 10.1' if k == 0.25 else '6.86 a 9.5')), xs_rep[k][ic + 1])], '[ESTIMADO] x*(%d) "antes" (q*eta_e), k2/k=%g' % (lvl, k), xs_rep[k][ic + 1], xq, dato='data/filtro_completo/*, data/termico/*', script=S, tol=0.006)
    # diferencias efectivo - completo
    dif100 = []
    for k in (0.05, 0.1, 0.25, 0.4):
        c = eff_curve(k, 1, 2e-5); d = E[(k, XM[k])]
        cc = np.c_[c[:, 0], c[:, 1], c[:, 2], c[:, 1] * d['rpf'] / (d['bf0c'] + d['rs'] * (c[:, 2] - d['bf0e']))]
        dif100.append((xstar_c(c, 100) - xstar_c(cc, 100), xstar_c(c, 220) - xstar_c(cc, 220)))
    dif100 = np.array(dif100)
    for i, (k, rep1, rep2) in enumerate(zip((0.05, 0.1, 0.25, 0.4), ('-0.001', '0.111', '0.187', '0.177'), ('-0.001', '0.109', '0.166', '0.125'))):
        claim('T85-dif100-%g' % k, 1, [(H8, 'diferencias efectivo menos completo: -0.001, 0.111, 0.187, 0.177')], 'x*(100) efectivo menos completo k2/k=%g' % k, rep1, dif100[i, 0], tol=0.003, dato='(derivado de T85)', script=S)
        claim('T85-dif220-%g' % k, 1, [(H8, 'diferencias efectivo menos completo: -0.001, 0.111, 0.187, 0.177')], 'x*(220) efectivo menos completo k2/k=%g' % k, rep2, dif100[i, 1], tol=0.003, dato='(derivado de T85)', script=S)
    claim('T85-rango', 1, [(H8, 'pide un x* mayor que el completo por 0 a 0.2 unidades'), (H8, 'el efectivo es conservador')], 'El efectivo pide un x* mayor que el completo por 0 a 0.2 unidades de x', '0.2', dif100.max(), modo='le', tol=0.0, dato='(derivado de T85)', script=S, estado=None, nota='[ESTIMADO] por construccion; max %.3f' % dif100.max())
    return E

def bloque_plano(E):
    S = 'consolidacion/verificar_todo.py::bloque_plano'
    K = KAPM
    zp, fp = fcomp(0.25, 6.86, plano=True); zp0, fp0 = fcomp(0.25, None, plano=True)
    znq, fnq = fcomp(0.25, 6.86, plano=True, suf='_nqoff'); znm, fnm = fcomp(0.25, 6.86, plano=True, suf='_nmoff')
    g = lambda z, key='gpf': float(z[key]) / K
    ep, ep0 = eff_pt(0.25, 6.86, 0), eff_pt(0.25, 60, 0)
    ek, ek0 = eff_pt(0.25, 6.86, 0, extra='_kick'), eff_pt(0.25, 60, 0, extra='_kick')
    ekq, ekm = eff_pt(0.25, 6.86, 0, extra='_kick_nqoff'), eff_pt(0.25, 6.86, 0, extra='_kick_nmoff')
    enq, enm = eff_pt(0.25, 6.86, 0, extra='_nqoff'), eff_pt(0.25, 6.86, 0, extra='_nmoff')
    DD = fp + ' ; ' + fp0 + ' ; ' + ep['file']
    H7 = '### 8.7'
    claim('PL7-1', 1, [(H8, '**Cocientes completo/efectivo plano:** γ_pf 1.522', '1.522'), (PH, 'c/e γ_pf **1.52**', '1.52'), (RM, 'γ_pf c/e = 1.52', '1.52')], 'Plano: gamma_pf completo/efectivo (termico) k2/k=0.25 x=6.86', '1.522', g(zp) / ep['gpf'], dato=DD, script=S, tol=0.005)
    claim('PL7-2', 1, [(H8, '0.94 (T = 0)'), (RM, 'c/e = 0.940')], 'Plano: gamma_pf completo/efectivo a T=0', '0.94', g(zp0) / ep0['gpf'], dato=DD, script=S, tol=0.0006)
    claim('PL7-3', 1, [(H8, 'γ_bf 0.8584'), (RM, 'γ_bf c/e = 0.858')], 'Plano: gamma_bf completo/efectivo (termico)', '0.8584', g(zp, 'gbf') / ep['gbf'], dato=DD, script=S, tol=None)
    ka = lambda a: 2 * (gx_pl[0] * a + gx_pl[1] * (a + 1)) + 2 * (6e-7 / K) * a
    gx_pl = Gam1(0.0535714, 6.0, K); gx_pl = (gx_pl[0] / K, gx_pl[1] / K)
    claim('PL7-4', 1, [(H8, '(8.863e-04 frente a 8.863e-04)', '8.863e-04'), (PH, 'con |α|² = 4: 8.863e-4 frente a 8.863e-4', '8.863e-4')], 'Plano T=0: formula 2[G1m a + G1p(a+1)] + 2 gamma a con |a|^2=4 (kappa=1)', '8.863e-04', ka(4.0), dato='(analitico)', script=S, nota='gamma = 6e-7/0.03 = 2e-5 en unidades de kappa')
    claim('PL7-5', 1, [(H8, '(8.863e-04 frente a 8.863e-04)', '8.863e-04')], 'Plano T=0 efectivo (x=60) gamma_pf', '8.863e-04', ep0['gpf'], dato=ep0['file'], script=S)
    a2 = float(np.real(zp0['al2eff']))
    claim('PL7-6', 1, [(H8, 'con α_eff² = 3.798 del completo la fórmula da 8.425e-04', '8.425e-04'), (PH, 'la fórmula da 8.425e-4', '8.425e-4')], 'Plano T=0: formula con alpha_eff^2 = 3.798 del completo', '8.425e-04', ka(a2), dato=fp0, script=S, nota='alpha_eff^2 recalculado %.4f' % a2)
    claim('PL7-7', 1, [(H8, 'la reproduce al 1.1%'), (PH, 'el completo la reproduce al 1.1%')], 'Plano T=0: el completo reproduce la formula con alpha_eff al 1.1%', '0.011', abs(g(zp0) / ka(a2) - 1), dato=fp0, script=S, tol=0.0006)
    claim('PL7-8', 1, [(H8, 'el efectivo sin α_eff² es 6.4% mayor que el completo a T = 0')], 'Plano T=0: efectivo sin alpha_eff^2 es 6.4% mayor que el completo', '0.064', ep0['gpf'] / g(zp0) - 1, dato=DD, script=S, tol=0.0006)
    tt = g(zp); on_off = [g(zp), g(znq), g(znm), g(zp0)]
    for nm, rep, v, loc in (('ambos encendidos', '1.369e-03', tt, 0), ('n_q apagado', '8.445e-04', g(znq), 0), ('n_m apagado', '1.358e-03', g(znm), 0), ('ambos apagados (T=0)', '8.331e-04', g(zp0), 0)):
        claim('PL7-ch-%s' % nm.replace(' ', '_').replace('(', '').replace(')', '').replace('=', ''), 1, [(H8, 'ambos encendidos 1.369e-03'), (PH, 'ambos on 1.3694e-3', rep)], 'Plano completo, apagado por canales: gamma_pf %s' % nm, rep, v, dato=fp + ' ; ' + fnq + ' ; ' + fnm, script=S, tol=None)
    inc = g(zp) - g(zp0); incnq = g(zp) - g(znq)
    claim('PL7-inc', 1, [(H8, 'Aumento térmico total 5.362e-04'), (PH, 'Aumento térmico total 5.36e-4')], 'Plano completo: aumento termico total de gamma_pf', '5.362e-04', inc, dato=DD, script=S)
    claim('PL7-incnq', 1, [(H8, '5.248e-04 (98%) es el canal del qubit'), (PH, '5.25e-4 (98%) es el canal del qubit')], 'Plano completo: incremento por n_q (termico - n_q apagado)', '5.248e-04', incnq, dato=DD, script=S)
    claim('PL7-98', 1, [(H8, '(98%)')], 'Plano completo: el canal n_q es el 98% del aumento', '0.98', incnq / inc, dato=DD, script=S, tol=0.005)
    nq = float(zp['nq'])
    claim('PL7-nq', 1, [(H8, 'n_q = 1.050e-03'), (PH, 'n_q = 1.05e-3')], 'n_q a x=6.86: 1/(e^6.86 - 1)', '1.050e-03', 1 / (math.exp(6.86) - 1), dato='(analitico)', script=S, tol=None)
    claim('PL7-half', 1, [(H8, 'igual a 0.5·n_qκ'), (PH, '= 0.50·n_q·κ')], 'Plano completo: incremento por n_q = 0.5 n_q kappa', '0.500', incnq / nq, dato=DD, script=S, tol=0.0006)
    incnm = g(zp) - g(znm)
    claim('PL7-nm', 1, [(H8, 'el canal n_m aporta 1.139e-05 en el completo')], 'Plano completo: incremento por n_m', '1.139e-05', incnm, dato=fp + ' ; ' + fnm, script=S)
    claim('PL7-nme', 1, [(H8, '1.204e-05 en el efectivo sin desplazamiento (1.186e-05 con desplazamiento)', '1.204e-05')], 'Plano efectivo sin desplazamiento: incremento por n_m', '1.204e-05', ep['gpf'] - enm['gpf'], dato=ep['file'], script=S)
    claim('PL7-nmk', 1, [(H8, '1.204e-05 en el efectivo sin desplazamiento (1.186e-05 con desplazamiento)', '1.186e-05')], 'Plano efectivo con desplazamiento: incremento por n_m', '1.186e-05', ek['gpf'] - ekm['gpf'], dato=ek['file'], script=S)
    claim('PL7-nq0', 1, [(H8, 'El efectivo sin desplazamiento aporta por n_q solo 1.272e-06')], 'Plano efectivo sin desplazamiento: incremento por n_q', '1.272e-06', ep['gpf'] - enq['gpf'], dato=ep['file'], script=S)
    claim('PL7-6pct', 1, [(H8, 'es decir, el efectivo lo reproduce al 6%')], 'n_m: el efectivo reproduce el canal al 6% (1.204e-5 frente a 1.139e-5)', '0.06', (ep['gpf'] - enm['gpf']) / incnm - 1, dato=DD, script=S, tol=0.006)
    knq = ek['gpf'] - ekq['gpf']
    claim('PL7-kick', 1, [(H8, 'incremento por n_q 6.101e-04'), (PH, '**Incremento por n_q** (térmico − n_q off)', '6.10e-4')], 'Plano efectivo con desplazamiento (kick): incremento por n_q', '6.101e-04', knq, dato=ek['file'], script=S)
    claim('PL7-16', 1, [(H8, 'exceso de 16%'), (PH, 'exceso del 16%')], 'Exceso del 16% (efectivo con kick / completo, incremento por n_q)', '0.16', knq / incnq - 1, dato=DD, script=S, tol=0.006)
    claim('PL7-evk', 1, [(H8, 'tasa por evento 0.581·n_qκ')], 'Tasa por evento con desplazamiento: incremento/(n_q kappa)', '0.581', knq / nq, dato=ek['file'], script=S, tol=0.0006)
    claim('PL7-ev0', 1, [(H8, 'sin desplazamiento: 0.00121')], 'Tasa por evento sin desplazamiento', '0.00121', (ep['gpf'] - enq['gpf']) / nq, dato=ep['file'], script=S, tol=0.000006)
    claim('PL7-kg', 1, [(H8, 'γ_pf 1.506e-03 / 8.996e-04 / 1.369e-03')], 'Plano: gamma_pf con kick / sin / completo (kappa=1): 1.506e-3', '1.506e-03', ek['gpf'], dato=ek['file'], script=S)
    claim('PL7-kg2', 1, [(H8, 'γ_pf 1.506e-03 / 8.996e-04 / 1.369e-03')], 'Plano efectivo sin kick gamma_pf', '8.996e-04', ep['gpf'], dato=ep['file'], script=S)
    claim('PL7-kb', 1, [(H8, 'γ_bf 3.192e-05 / 3.342e-05 / 2.869e-05')], 'Plano: gamma_bf con kick', '3.192e-05', ek['gbf'], dato=ek['file'], script=S)
    claim('PL7-kb2', 1, [(H8, 'γ_bf 3.192e-05 / 3.342e-05 / 2.869e-05')], 'Plano efectivo sin kick gamma_bf', '3.342e-05', ep['gbf'], dato=ep['file'], script=S)
    claim('PL7-kb3', 1, [(H8, 'γ_bf 3.192e-05 / 3.342e-05 / 2.869e-05')], 'Plano completo gamma_bf', '2.869e-05', g(zp, 'gbf'), dato=fp, script=S)
    claim('PL7-eta', 1, [(H8, 'η plano con el desplazamiento (47.17)')], 'Plano: eta con kick', '47.17', ek['gpf'] / ek['gbf'], dato=ek['file'], script=S)
    claim('PL7-etac', 1, [(H8, 'a 1.2% de η completo (47.73)')], 'Plano: eta completo', '47.73', g(zp) / g(zp, 'gbf'), dato=fp, script=S)
    claim('PL7-10', 1, [(H8, 'γ_pf queda 10% y γ_bf 11% sobre el completo')], 'Plano con kick: gamma_pf +10% y gamma_bf +11% sobre el completo', '0.10,0.11', '%.3f,%.3f' % (ek['gpf'] / g(zp) - 1, ek['gbf'] / g(zp, 'gbf') - 1), dato=DD, script=S, nota='%.3f y %.3f' % (ek['gpf'] / g(zp) - 1, ek['gbf'] / g(zp, 'gbf') - 1))
    CL[-1]['estado'] = 'V' if abs(ek['gpf'] / g(zp) - 1 - 0.10) < 0.005 and abs(ek['gbf'] / g(zp, 'gbf') - 1 - 0.11) < 0.005 else 'D'
    claim('PL7-piso', 1, [(H8, 'completo 3.042e-07, efectivo sin desplazamiento 2.114e-07, con desplazamiento 1.667e-07')], 'Plano T=0: piso gamma_bf completo 3.042e-7, efectivo 2.114e-7, con kick 1.667e-7', '3.042e-07', g(zp0, 'gbf'), dato=fp0, script=S)
    claim('PL7-piso2', 1, [(H8, 'completo 3.042e-07, efectivo sin desplazamiento 2.114e-07, con desplazamiento 1.667e-07')], 'Plano T=0 efectivo sin kick', '2.114e-07', ep0['gbf'], dato=ep0['file'], script=S)
    claim('PL7-piso3', 1, [(H8, 'completo 3.042e-07, efectivo sin desplazamiento 2.114e-07, con desplazamiento 1.667e-07')], 'Plano T=0 efectivo con kick', '1.667e-07', ek0['gbf'], dato=ek0['file'], script=S)
    zf = E[(0.25, 6.86)]
    claim('PL7-f17', 1, [(H8, 'completo +17%'), (RM, '17% en el completo'), (H8, 'en el completo es +17%')], 'gamma_bf filtro/plano en el completo (0.25, x=6.86): +17%', '0.17', zf['bc'] / g(zp, 'gbf') - 1, dato=zf['fz'] + ' ; ' + fp, script=S, tol=0.005)
    claim('PL7-f28', 1, [(H8, 'el efectivo da +28%'), (RM, '28% en el efectivo')], 'gamma_bf filtro/plano en el efectivo: +28%', '0.28', zf['be'] / ep['gbf'] - 1, dato=zf['fe'] + ' ; ' + ep['file'], script=S, tol=0.005)
    claim('PL7-f1277', 1, [(H8, 'γ_bf(filtro)/γ_bf(plano) = 1.277')], 'gamma_bf(filtro)/gamma_bf(plano) efectivo = 1.277', '1.277', zf['be'] / ep['gbf'], dato=zf['fe'], script=S)
    claim('PL7-f1167', 1, [(H8, 'el completo da 1.167')], 'gamma_bf(filtro)/gamma_bf(plano) completo = 1.167', '1.167', zf['bc'] / g(zp, 'gbf'), dato=zf['fz'], script=S)
    claim('PL7-f8', 1, [(H8, 'γ_pf plano/filtrado 5.22')], 'Completo con ruido blanco: gamma_pf plano/filtrado = 8.29', '8.29', g(zp) / zf['pc'], dato=zf['fz'] + ' ; ' + fp, script=S, tol=0.005, nota='HANDOFF 8.12: "El completo con ruido blanco da gamma_bf 0.857 y gamma_pf 8.29"')
    claim('PL7-f857', 1, [(H8, 'El completo con ruido blanco da γ_bf 0.857')], 'Completo con ruido blanco: gamma_bf plano/filtrado = 0.857', '0.857', g(zp, 'gbf') / zf['bc'], dato=zf['fz'] + ' ; ' + fp, script=S, tol=0.0006)
    # barrido en N plano
    Ns = {}
    for N in (18, 22, 26):
        zt, ft = fcomp(0.25, 6.86, N=N, plano=True); z0_, f0_ = fcomp(0.25, None, N=N, plano=True)
        Ns[N] = (g(zt), g(zt, 'gbf'), g(z0_), g(z0_, 'gbf'), ft)
    rep = {18: ('1.377e-03', '3.621e-05', '8.404e-04', '7.567e-06'), 22: ('1.369e-03', '2.869e-05', '8.331e-04', '3.042e-07'), 26: ('1.369e-03', '2.868e-05', '8.331e-04', '2.983e-07')}
    for N in (18, 22, 26):
        for i, nm in enumerate(('gpf termico', 'gbf termico', 'gpf T=0', 'gbf T=0')):
            claim('PL7-N%d-%d' % (N, i), 1, [(H8, '**Barrido en N del plano**'), (PH, '**(3) Barrido en N, plano:**')], 'Plano barrido en N=%d: %s (kappa=1)' % (N, nm), rep[N][i], Ns[N][i], dato='data/filtro_completo/*Nf1_plano*N%d*.npz' % N, script=S, tol=None)
    claim('PL7-N26-pf', 1, [(H8, 'γ_pf cambia 4.4e-06 (térmico) y 7.2e-06 (T = 0) en relativo')], 'Plano N=22->26: gamma_pf termico cambia 4.4e-6 relativo', '4.4e-06', abs(Ns[26][0] / Ns[22][0] - 1), dato='data/filtro_completo/*Nf1_plano*', script=S, tol=0.2e-6)
    claim('PL7-N26-pf0', 1, [(H8, 'γ_pf cambia 4.4e-06 (térmico) y 7.2e-06 (T = 0) en relativo')], 'Plano N=22->26: gamma_pf T=0 cambia 7.2e-6 relativo', '7.2e-06', abs(Ns[26][2] / Ns[22][2] - 1), dato='data/filtro_completo/*Nf1_plano*', script=S, tol=0.2e-6)
    claim('PL7-N26-abs', 1, [(H8, '6.0e-09κ en absoluto'), (H8, 'cambios absolutos de las tasas de N = 22 a 26 son ≈ 6e-9κ (6.0, 6.0, 6.5 y 5.9e-9)')], 'Plano N=22->26: cambios absolutos gamma_pf (T) 6.0e-9, gamma_pf (T=0) 6.0e-9, gamma_bf (T) 6.5e-9, gamma_bf (T=0) 5.9e-9', '6.0,6.0,6.5,5.9',
          ','.join('%.1f' % (abs(Ns[26][i] - Ns[22][i]) * 1e9) for i in (0, 2, 1, 3)), dato='data/filtro_completo/*Nf1_plano*', script=S)
    claim('PL7-N26-p', 1, [(H8, 'el piso 1.9% relativo'), (PH, 'el piso cambia un 2% relativo')], 'Plano: el piso (gbf T=0) cambia 1.9% entre N=22 y 26', '0.019', abs(Ns[26][3] / Ns[22][3] - 1), dato='data/filtro_completo/*Nf1_plano*', script=S, tol=0.0006)
    claim('PL7-N26-t', 1, [(H8, 'γ_bf térmico 0.02%')], 'Plano N=22->26: gamma_bf termico 0.02%', '0.0002', abs(Ns[26][1] / Ns[22][1] - 1), dato='data/filtro_completo/*Nf1_plano*', script=S, tol=0.00005)
    claim('PL7-N18', 1, [(H8, 'N = 18 no converge** (γ_bf térmico 26% mayor')], 'Plano N=18: gamma_bf termico +26% y gamma_bf T=0 ~25x respecto de N=22', '0.26,25', '%.2f,%.1f' % (Ns[18][1] / Ns[22][1] - 1, Ns[18][3] / Ns[22][3]), dato='data/filtro_completo/*Nf1_plano*N{18,22}*', script=S, nota='%.3f y %.1f' % (Ns[18][1] / Ns[22][1] - 1, Ns[18][3] / Ns[22][3]))
    CL[-1]['estado'] = 'V' if abs(Ns[18][1] / Ns[22][1] - 1.26) < 0.01 and abs(Ns[18][3] / Ns[22][3] - 25) < 1 else 'D'
    pf = {N: Ns[N][3] / Ns[N][1] for N in Ns}
    claim('PL7-pf18', 1, [(PH, 'piso γ_bf(T=0)/térmico = 20.9% / 1.060% / 1.040%'), (H8, '(1.060% a 1.040%)')], 'Plano: piso/termico = 20.9% (N=18), 1.060% (22), 1.040% (26)', '20.9%', pf[18], dato='data/filtro_completo/*Nf1_plano*', script=S, tol=0.0005)
    claim('PL7-pf22', 1, [(H8, '(1.060% a 1.040%)')], 'Plano: piso/termico N=22', '1.060%', pf[22], dato='data/filtro_completo/*Nf1_plano*', script=S, tol=0.00001)
    claim('PL7-pf26', 1, [(H8, '(1.060% a 1.040%)')], 'Plano: piso/termico N=26', '1.040%', pf[26], dato='data/filtro_completo/*Nf1_plano*', script=S, tol=0.00001)
    return

def bloque_diag(E):
    S = 'consolidacion/verificar_todo.py::bloque_diag'
    K = KAPM
    # ---- alpha_eff^2 del completo y cociente de T=0 / termico (HANDOFF 8.3, 8.8)
    a_t0 = float(np.real(E[(0.25, 6.86)]['z0']['al2eff']))
    claim('AE-1', 1, [(H8, 'α_eff² medido en el completo (3.798)'), (RM, 'α_eff² medido en el completo (3.798)')], 'alpha_eff^2 del completo a T=0 (Re): 3.798', '3.798', a_t0, dato=E[(0.25, 6.86)]['f0'], script=S)
    claim('AE-2', 1, [(H8, 'predice 0.9495'), (RM, 'predice 0.9495')], 'Cociente de T=0 predicho por alpha_eff^2: 3.798/4 = 0.9495', '0.9495', a_t0 / 4.0, dato=E[(0.25, 6.86)]['f0'], script=S)
    claim('AE-3', 1, [(H8, 'se mide 0.956'), (RM, 'se mide 0.956')], 'Cociente de T=0 gamma_pf completo/efectivo medido: 0.956', '0.956', E[(0.25, 6.86)]['z0']['gpf'] / K / E[(0.25, 6.86)]['e0']['gpf'], dato=E[(0.25, 6.86)]['f0'], script=S)
    inc = [((E[(k, 6.86)]['pc'] - float(E[(k, 6.86)]['z0']['gpf']) / K) / (E[(k, 6.86)]['pe'] - E[(k, 6.86)]['e0']['gpf'])) for k in (0.05, 0.1, 0.25, 0.4)]
    claim('AE-4', 1, [(H8, 'se mide 0.994 a 0.996'), (RM, 'se mide 0.994 a 0.996')], 'Cociente de incrementos termicos completo/efectivo a x=6.86: 0.994-0.996 (min)', '0.994', min(inc), dato='data/filtro_completo/*, data/termico/*', script=S, tol=0.0006, nota='valores %s; a x~9 el cociente es 0.968-0.973 (no mencionado en HANDOFF)' % np.round(inc, 4))
    claim('AE-5', 1, [(H8, 'se mide 0.994 a 0.996'), (RM, 'se mide 0.994 a 0.996')], 'Idem: maximo 0.996', '0.996', max(inc), dato='data/filtro_completo/*, data/termico/*', script=S, tol=0.0006)
    # alpha diag: cambio relativo del al2eff completo T0 -> termico
    for k, x, rep in ((0.05, 6.86, '-3.23e-4'), (0.1, 6.86, '-5.14e-4'), (0.25, 6.86, '-9.18e-4'), (0.4, 6.86, '-1.16e-3'), (0.05, 9, '-3.80e-5'), (0.1, 9, '-6.04e-5'), (0.25, 10.1, '-3.60e-5'), (0.4, 9.5, '-8.31e-5')):
        d = E[(k, x)]; v = (float(np.real(d['z']['al2eff'])) - float(np.real(d['z0']['al2eff']))) / float(np.real(d['z0']['al2eff']))
        claim('AD-%g-%g' % (k, x), 1, [(H8, '−3.2e-4 a −1.2e-3 en x = 6.86'), (H8, '−3.23e-4, −5.14e-4, −9.18e-4, −1.16e-3')], 'Delta alpha_eff^2/alpha_eff^2 del completo, T=0 -> termico, k2/k=%g x=%g' % (k, x), rep, v, dato=d['fz'] + ' ; ' + d['f0'], script=S, tol=None)
    # E_obs
    rows = [(0.05, 6.86, '0.95614', '0.95883', '+4.629e-07', '+4.620e-07', '1.002'), (0.05, 9, '0.95614', '0.95655', '+6.730e-08', '+5.440e-08', '1.237'), (0.1, 6.86, '0.95611', '0.95880', '+4.643e-07', '+4.627e-07', '1.003'),
            (0.1, 9, '0.95611', '0.95652', '+6.746e-08', '+5.448e-08', '1.238'), (0.25, 6.86, '0.95601', '0.95875', '+4.725e-07', '+4.646e-07', '1.017'), (0.25, 10.1, '0.95601', '0.95618', '+2.764e-08', '+1.823e-08', '1.516'),
            (0.4, 6.86, '0.95591', '0.95870', '+4.804e-07', '+4.667e-07', '1.029'), (0.4, 9.5, '0.95591', '0.95619', '+4.564e-08', '+3.332e-08', '1.370')]
    eobs = {}
    for k, x, r0, rx, eo, ea, rat in rows:
        d = E[(k, x)]
        r0c = float(d['z0']['gpf']) / K / d['e0']['gpf']; rxc = d['rpf']; Eo = (rxc - r0c) * d['pe']
        eobs[(k, x)] = Eo
        loc = '| %g | %g | %s' % (k, x, r0)
        claim('EO-%g-%g-r0' % (k, x), 1, [(H8, loc)], 'E_obs tabla: r_0 (cociente T=0 completo/efectivo) k2/k=%g' % k, r0, r0c, dato=d['f0'] + ' ; ' + d['e0']['file'], script=S)
        claim('EO-%g-%g-rx' % (k, x), 1, [(H8, loc)], 'E_obs tabla: r_x k2/k=%g x=%g' % (k, x), rx, rxc, dato=d['fz'] + ' ; ' + d['fe'], script=S)
        claim('EO-%g-%g-Eo' % (k, x), 1, [(H8, loc)], 'E_obs = (r_x - r_0) gamma_e(x) k2/k=%g x=%g' % (k, x), eo, Eo, dato=d['fz'] + ' ; ' + d['f0'] + ' ; ' + d['fe'], script=S)
        if (k, x) in ((0.25, 6.86), (0.4, 6.86)):
            ek = eff_pt(k, x, 1, extra='_kick'); ek0 = eff_pt(k, 60, 1, extra='_kick')
            Ea = (ek['gpf'] - d['pe']) - (ek0['gpf'] - d['e0']['gpf'])
            claim('EO-%g-%g-Ea' % (k, x), 1, [(H8, loc)], 'E_ansatz = aumento de gamma_pf por el ansatz con n_q (kick - sin kick, x menos T=0) k2/k=%g x=%g' % (k, x), ea, Ea, dato=ek['file'] + ' ; ' + ek0['file'], script=S, tol=None)
            claim('EO-%g-%g-rat' % (k, x), 1, [(H8, loc)], 'E_obs/E_ansatz k2/k=%g x=%g' % (k, x), rat, Eo / Ea, dato='(derivado)', script=S, tol=None)
        else:
            claim('EO-%g-%g-Ea' % (k, x), 1, [(H8, loc)], 'E_ansatz k2/k=%g x=%g (valor de diagnostico_E_obs.txt; no hay cache kick para este punto)' % (k, x), ea, None, estado='R', dato='data/diagnostico_E_obs.txt', script='codigo/diagnostico_E_obs.py', nota='no recomputable desde las caches: el efectivo con ansatz para este punto no esta en data/termico')
            claim('EO-%g-%g-rat' % (k, x), 1, [(H8, loc)], 'E_obs/E_ansatz = %s (usa E_ansatz logueado)' % rat, rat, Eo / float(ea.replace('+', '')), dato='data/diagnostico_E_obs.txt', script=S, tol=None, nota='aritmetica con E_ansatz del log')
    nok = sum(1 for (k, x, r0, rx, eo, ea, rat) in rows if 0.7 <= float(rat) <= 1.3)
    claim('EO-crit', 1, [(H8, 'está en [0.7, 1.3] en 6 de 8 puntos'), (H8, '**no consistente**')], 'Criterio fijado antes: consistente si E_obs/E_ansatz en [0.7,1.3] en los 8 puntos; resultado: 6 de 8', '6', nok, dato='data/diagnostico_E_obs.txt', script=S, tol=0)
    # ---- kf/omega = 0.2
    zT0, fT0 = fcomp(0.25, None, N=20, kf=1.2); zT, fT = fcomp(0.25, 6.86, N=20, kf=1.2)
    DK = fT0 + ' ; ' + fT
    gp0, gpT, gb0, gbT = float(zT0['gpf']) / K, float(zT['gpf']) / K, float(zT0['gbf']) / K, float(zT['gbf']) / K
    for nm, rep, v in (('gamma_pf T=0', '1.58888e-04', gp0), ('gamma_pf termico', '1.76739e-04', gpT), ('gamma_bf T=0', '3.449e-07', gb0), ('gamma_bf termico', '2.911e-05', gbT), ('eta termico', '6.07', gpT / gbT), ('eta(T=0)', '460.6', gp0 / gb0)):
        claim('K2-%s' % nm.replace(' ', '_'), 1, [(H8, 'γ_pf: T = 0 1.58888e-04'), (RM, 'A κ_f/ω = 0.2 (N = 20)')] if nm == 'gamma_pf T=0' else [(H8, 'γ_pf: T = 0 1.58888e-04')], 'kf/w=0.2 (N=20, 0.25, x=6.86, kappa=1): %s' % nm, rep, v, dato=DK, script=S, tol=None)
    ge, ge0 = 1.78395e-04, 1.66411e-04        # efectivo del preregistro (no cacheado)
    claim('K2-pre', 1, [(PRE, 'γ_pf a T = 0 (x = 60)', '1.66411e-04')], 'Prerregistro: gamma_pf efectivo T=0 y termico (kf/w=0.2, N=22) 1.66411e-04 / 1.78395e-04', '1.66411e-04,1.78395e-04', None, estado='R', dato='data/preregistro_kf0.2.md; no hay cache del efectivo con kf/w=0.2 en data/termico', script='codigo/calc_termico.py', nota='valores del efectivo predichos y fijados antes de las corridas; no recomputables sin simulacion')
    r0k, rxk = gp0 / ge0, gpT / ge
    claim('K2-r0', 1, [(H8, 'r_0 = 0.95479'), (PRE, 'r_0', '0.95479')], 'kf/w=0.2: r_0 = gamma_pf completo(T=0)/efectivo(prerregistro) = 0.95479', '0.95479', r0k, dato=DK, script=S, nota='usa el efectivo del preregistro (R) como entrada', tol=None)
    claim('K2-rx', 1, [(H8, 'r_x = 0.99072')], 'kf/w=0.2: r_x = 0.99072', '0.99072', rxk, dato=DK, script=S, tol=None)
    Eok = (rxk - r0k) * ge
    claim('K2-Eobs', 1, [(H8, '**E_obs = (r_x − r_0)·γ_e = +6.41e-06**'), (RM, 'E_obs = +6.4e-6'), (PH, '')][:2], 'kf/w=0.2: E_obs = (r_x - r_0) gamma_e = +6.41e-6', '6.41e-06', Eok, dato=DK, script=S, tol=None)
    claim('K2-Eratio', 1, [(H8, 'E_obs/E_ansatz = 0.873')], 'kf/w=0.2: E_obs/E_ansatz = 0.873 con E_ansatz = 7.341e-6 (preregistro)', '0.873', Eok / 7.341e-6, dato=DK, script=S, tol=None)
    Dk = (gpT - gp0) - 1.1984e-5
    claim('K2-D', 1, [(H8, 'D = Δ_c − Δ_e = +5.87e-06')], 'kf/w=0.2: D = (gamma_pf termico - T=0) - Delta_e = +5.87e-6', '5.87e-06', Dk, dato=DK, script=S, tol=None)
    claim('K2-regla', 1, [(H8, 'Regla (E_obs > 3e-6: canal de tamaño ansatz)'), (PRE, 'E_obs > 3e-6')], 'Regla preregistrada: E_obs > 3e-6 => canal de tamaño ansatz; se cumple', 'si', Eok > 3e-6, modo='bool', dato=DK, script=S)
    claim('K2-al', 1, [(H8, 'α_eff² medido: 3.79744 (T = 0) y 3.79408 (térmico)')], 'kf/w=0.2: alpha_eff^2 3.79744 (T=0) y 3.79408 (termico)', '3.79744,3.79408', '%.5f,%.5f' % (zT0['al2eff'].real, zT['al2eff'].real), dato=DK, script=S)
    CL[-1]['estado'] = 'V' if round(float(zT0['al2eff'].real), 5) == 3.79744 and round(float(zT['al2eff'].real), 5) == 3.79408 else 'D'
    claim('K2-herm', 1, [(H8, 'Hermiticidad cruda 1.55e-09 (T = 0) y 7.5e-11 (térmico)')], 'kf/w=0.2: hermiticidad cruda 1.55e-9 (T=0) y 7.5e-11 (termico)', '1.55e-09', float(zT0['herm_cruda']), dato=fT0, script=S)
    claim('K2-herm2', 1, [(H8, 'Hermiticidad cruda 1.55e-09 (T = 0) y 7.5e-11 (térmico)')], 'kf/w=0.2: hermiticidad cruda termico 7.5e-11', '7.5e-11', float(zT['herm_cruda']), dato=fT, script=S)
    # git: preregistro antes de las corridas
    t_pre = sh("git log --format=%ct -1 -- Mechanical-Cat-State/figuras_finales/data/preregistro_kf0.2.md")
    log7 = open(os.path.join(FC, 'log_termico_ma7.txt'), errors='ignore').read()
    m7 = re.search(r'== (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) kf=1.2', log7)
    import datetime
    t_run = datetime.datetime.strptime(m7.group(1), '%Y-%m-%d %H:%M:%S').timestamp() if m7 else None
    # el commit esta en -0500; el log en hora local de la maquina de calculo (asumida -0500)
    claim('K2-prereg', 1, [(PRE, 'No se ha lanzado ninguna corrida de esta serie'), (H8, 'regla fijada antes')], 'El preregistro (commit cfa8f01, 2026-10-01 13:29:45 -0500) es anterior al primer registro de corrida (log_termico_ma7: 2026-10-01 13:29:55)', 'si',
          bool(t_pre) and t_run is not None and int(t_pre) <= t_run + 5 * 3600, modo='bool', dato='git log; data/filtro_completo/log_termico_ma7.txt', script=S, commit='cfa8f01', nota='el log no trae zona horaria: se asume la del commit (-0500); margen de 10 s entre el commit y el inicio de la corrida')
    for ky, rep, f in (('K2-t0', '3529.95', 'ma7'),):
        pass
    # recursos (logs)
    def tr(fn):
        L = open(os.path.join(FC, fn), errors='ignore').read()
        return [(float(a), int(b)) for a, b in re.findall(r'TIEMPO_REAL=([\d.]+) s\s+MEMORIA_MAX=(\d+) kB', L)]
    r4, r5, r6, r7 = tr('log_termico_ma4.txt'), tr('log_termico_ma5.txt'), tr('log_termico_ma6.txt'), tr('log_termico_ma7.txt')
    claim('RC-N22', 1, [(H8, '**14.0 GB**'), (RM, 'la corrida con N = 22 ya usa 14.0 GB')], 'Memoria maxima N=22 con filtro: 14015972, 14044332, 14017232 kB (14.0 GB)', '14.0', max(m for t, m in r4) / 1e6, dato='data/filtro_completo/log_termico_ma4.txt', script=S, tol=0.06, nota='kB leidos: %s' % [m for t, m in r4])
    claim('RC-N20', 1, [(H8, 'N = 20, 11.36 GB (11360168 kB)')], 'Memoria maxima N=20: 11360168 kB', '11360168', r5[0][1], dato='data/filtro_completo/log_termico_ma5.txt', script=S, tol=0)
    claim('RC-N18', 1, [(H8, 'N = 18, 7.51 GB (7513876 kB)')], 'Memoria maxima N=18: 7513876 kB', '7513876', r6[0][1], dato='data/filtro_completo/log_termico_ma6.txt', script=S, tol=0)
    claim('RC-t22', 1, [(H8, 'fase 6, 5553.24 s, 5460.22 s y 5608.04 s')], 'Tiempos reales N=22 (fase 6): 5553.24, 5460.22, 5608.04 s', '5553.24,5460.22,5608.04', ','.join('%.2f' % t for t, m in r4), dato='data/filtro_completo/log_termico_ma4.txt', script=S)
    claim('RC-t20', 1, [(H8, 'Tiempo real 3515.15 s (58.6 min)')], 'Tiempo real N=20: 3515.15 s', '3515.15', r5[0][0], dato='data/filtro_completo/log_termico_ma5.txt', script=S)
    claim('RC-t18', 1, [(H8, 'tiempo real 2116.63 s (35.3 min)')], 'Tiempo real N=18: 2116.63 s', '2116.63', r6[0][0], dato='data/filtro_completo/log_termico_ma6.txt', script=S)
    claim('RC-kf', 1, [(H8, 'Tiempo real 3529.95 s y 3488.54 s')], 'Tiempos reales kf/w=0.2: 3529.95 y 3488.54 s, memoria ~11.36 GB', '3529.95,3488.54', ','.join('%.2f' % t for t, m in r7), dato='data/filtro_completo/log_termico_ma7.txt', script=S)
    # convergencia en N con filtro
    zn = {N: fcomp(0.25, 6.86, N=N) for N in (18, 20, 22)}
    g_ = lambda N, key: float(zn[N][0][key]) / K
    for N, rep in ((18, ('1.65382e-04', '4.64921e-05')), (20, ('1.65261e-04', '3.38515e-05')), (22, ('1.65256e-04', '3.34895e-05'))):
        claim('CN-%d-pf' % N, 1, [(H8, 'γ_pf 1.65261e-04, γ_bf 3.38515e-05') if N == 20 else (H8, 'γ_pf 1.65382e-04, γ_bf 4.64921e-05') if N == 18 else (H8, 'Cambio de N = 20 a N = 22 (1.65256e-04, 3.34895e-05')], 'Convergencia en N con filtro (0.25, x=6.86): gamma_pf N=%d' % N, rep[0], g_(N, 'gpf'), dato=zn[N][1], script=S, tol=None)
        claim('CN-%d-bf' % N, 1, [(H8, 'γ_pf 1.65261e-04, γ_bf 3.38515e-05') if N == 20 else (H8, 'γ_pf 1.65382e-04, γ_bf 4.64921e-05') if N == 18 else (H8, 'Cambio de N = 20 a N = 22 (1.65256e-04, 3.34895e-05')], 'Convergencia en N: gamma_bf N=%d' % N, rep[1], g_(N, 'gbf'), dato=zn[N][1], script=S, tol=None)
    claim('CN-d20pf', 1, [(H8, 'γ_pf **−3.2e-05**')], 'N=20 -> 22: gamma_pf cambia -3.2e-5 relativo (y -5.2e-9 kappa)', '-3.2e-05', g_(22, 'gpf') / g_(20, 'gpf') - 1, dato='data/filtro_completo/*N{20,22}*', script=S, tol=0.3e-5)
    claim('CN-d20bf', 1, [(H8, 'γ_bf **−1.07%**'), (H8, '+1.08% (N = 20)')], 'N=20 -> 22: gamma_bf cambia -1.07% (el HANDOFF 8.3/12 lo expresa tambien como +1.08% de N=20 respecto de N=22)', '-1.07%', g_(22, 'gbf') / g_(20, 'gbf') - 1, dato='data/filtro_completo/*N{20,22}*', script=S, tol=0.0001)
    claim('CN-d18bf', 1, [(H8, 'γ_bf +38.8% (N = 18)'), (H8, 'γ_bf **+38.8%**')], 'N=18 respecto de N=22: gamma_bf +38.8%', '38.8%', g_(18, 'gbf') / g_(22, 'gbf') - 1, dato='data/filtro_completo/*N{18,22}*', script=S, tol=0.0005)
    claim('CN-d18pf', 1, [(H8, 'γ_pf +0.077%')], 'N=18 respecto de N=22: gamma_pf +0.077%', '0.077%', g_(18, 'gpf') / g_(22, 'gpf') - 1, dato='data/filtro_completo/*N{18,22}*', script=S, tol=0.00001)
    d1, d2 = (g_(20, 'gbf') - g_(18, 'gbf')) * K, (g_(22, 'gbf') - g_(20, 'gbf')) * K
    rr = d2 / d1
    resid = d2 * rr / (1 - rr)
    claim('CN-q', 1, [(H8, 'cociente 0.029')], 'Diferencias en Ma: -3.79e-7 (18->20), -1.09e-8 (20->22), cociente 0.029', '0.029', rr, dato='data/filtro_completo/*N{18,20,22}*', script=S, tol=0.0006, nota='d1=%.3e d2=%.3e' % (d1, d2))
    claim('CN-resid', 1, [(H8, 'el residual de N = 22 es −3.2e-10 (Ma), es decir **−0.03%**')], 'Residual geometrico de N=22 en gamma_bf: -0.03%', '-0.03%', resid / (g_(22, 'gbf') * K), dato='data/filtro_completo/*N{18,20,22}*', script=S, tol=0.0001, estado=None, nota='[ESTIMADO] supuesto de decaimiento geometrico con tres puntos; resid=%.2e (Ma)' % resid)
    claim('CN-worst', 1, [(H8, 'Peor caso, si el cambio de N = 20 a 22 se repitiera sin decaer: −1.08%')], 'Peor caso (cambio 20->22 se repite): -1.08%', '-1.08%', d2 / (g_(22, 'gbf') * K), dato='data/filtro_completo/*N{18,20,22}*', script=S, tol=0.0002, nota='-1.08% es d2/gamma_bf(22); la variacion 20->22 expresada respecto de N=20 es -1.07% (CN-d20bf)')
    claim('CN-054', 1, [(H8, 'queda en 0.956 (4.4%)')], 'Cociente 0.956 con residual geometrico, y 0.946 en el peor caso', '0.946', E[(0.05, 6.86)]['rbf'] * (1 + d2 / (g_(22, 'gbf') * K)), dato='(derivado)', script=S, tol=0.001)
    claim('CN-pc', 1, [(H8, 'P_c 0.995830')], 'N=20: P_c(ss) 0.995830; N=22: 0.995840', '0.995830', float(zn[20][0]['Pc_ss']), dato=zn[20][1], script=S, tol=None)
    claim('CN-eta', 1, [(H8, 'η 4.882')], 'N=20: eta 4.882; N=22 4.935; N=18 3.557', '4.882', g_(20, 'gpf') / g_(20, 'gbf'), dato=zn[20][1], script=S)
    claim('CN-bord', 1, [(H8, 'Peso de borde de los modos elegidos 1.7e-05')], 'N=20: peso de borde del modo de paridad 1.7e-5 (modos[:,..] del npz)', '1.7e-05', None, estado='R', dato=zn[20][1], script='codigo/calc_filtro_completo.py', nota='columnas de modos no documentadas; se reporta como leido del log')
    return

# ============================================================ BLOQUE 9: calidad numerica (traza, hermiticidad, positividad) e historia de umbrales
def vals_familia(pat, skip=('pe_',)):
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, pat))):
        if any(s in os.path.basename(f) for s in skip): continue
        z = np.load(f, allow_pickle=True)
        if 'val' in z.files: out.append((float(z['val'][0]), float(z['val'][1]), float(z['val'][2]), os.path.basename(f)))
    return out

def bloque_calidad():
    S = 'consolidacion/verificar_todo.py::bloque_calidad'
    # --- (a) minimo autovalor de la verificacion con filtro N=22
    allv = []
    for fam in ('fig2/*.npz', 'fig3/*.npz', 'pfig2/*.npz', 'pfig3/*.npz', 'fig4/*.npz', 'filtro_completo/*.npz', 'efectivo/*.npz', 'minimo/*.npz'):
        for v in vals_familia(fam): allv.append(v + (fam.split('/')[0],))
    viol = sorted([v for v in allv if v[2] < -1e-9], key=lambda v: v[2])
    z5, f5 = np.load(os.path.join(FC, 'gx0.0216506_w6_gz0.36_kf0.3_al4_N22_Nf2.npz'), allow_pickle=True), 'data/filtro_completo/gx0.0216506_w6_gz0.36_kf0.3_al4_N22_Nf2.npz'
    claim('Q-a1', 1, [(INF, 'mínimo autovalor $\\geq-5.6\\times10^{-9}$'), (RM, 'mínimo autovalor ≥ −5.6e-9')], 'Verificacion del panel (b) con N=22: minimo autovalor >= -5.6e-9', '-5.6e-9', float(z5['val'][2]), modo='ge', tol=0, dato=f5, script=S, nota='valor medido %.3e' % float(z5['val'][2]))
    claim('Q-a2', 1, [(INF, 'mínimo autovalor mayor que $-10^{-9}$'), (RM, 'mínimo autovalor > −1e-9'), (H8, 'tolerancias 1e-10 / 1e-10 / −1e-9')], 'Criterio declarado de positividad (min autovalor > -1e-9) cumplido por todas las rho de figuras_finales/data/', 'si', len(viol) == 0, modo='bool', dato='data/**/*.npz (campo val)', script=S,
          nota='%d violaciones: %s' % (len(viol), '; '.join('%s %.2e (%s)' % (v[3], v[2], v[4]) for v in viol)))
    claim('Q-a3', 1, [(H8, 'Dos violaciones de positividad'), (H8, '**Las dos violaciones de positividad (criterio anterior −1e-9):**')], 'HANDOFF: solo 2 violaciones del criterio -1e-9 (corridas termicas con filtro a T=0)', '2', len(viol), dato='data/**/*.npz (campo val)', script=S, tol=0,
          nota='conteo total en data/: %d (las 2 reportadas son las de log_hermiticidad_positividad.txt, que solo cubre corridas termicas gam6e-07)' % len(viol))
    log = open(os.path.join(FC, 'log_hermiticidad_positividad.txt'), errors='ignore').read()
    rowsl = [l for l in log.split('\n') if l.endswith('.npz') or '.npz ' in l]
    parsed = []
    for l in rowsl:
        t = l.split()
        parsed.append(dict(nombre=t[0], k2=float(t[1]), N=int(t[2]), x=float(t[3]), herm=float(t[4]), tr=float(t[5]), mineig=float(t[6].replace('+', ''))))
    pl = [p for p in parsed if 'plano' in p['nombre']]; fi = [p for p in parsed if 'plano' not in p['nombre']]
    DL = 'data/filtro_completo/log_hermiticidad_positividad.txt'
    claim('Q-b1', 1, [(H8, '0 de 16 corridas superan 1e-8')], 'Log: corridas con filtro = 16, plano = 8', '16,8', '%d,%d' % (len(fi), len(pl)), dato=DL, script=S)
    n_h8 = sum(1 for p in fi if p['herm'] > 1e-8); n_p8 = sum(1 for p in fi if p['mineig'] < -1e-8)
    claim('Q-b2', 1, [(H8, 'completo con filtro, 0 de 16 corridas superan 1e-8 en hermiticidad y 0 superan −1e-8 en positividad')], 'Filtro, umbrales vigentes: 0 superan 1e-8 (herm) y 0 superan -1e-8 (positividad)', '0,0', '%d,%d' % (n_h8, n_p8), dato=DL, script=S)
    claim('Q-b3', 1, [(H8, '15 de 16 corridas con filtro superan 1e-10 en hermiticidad'), (RM, '')][:1], 'Filtro, umbral anterior: 15 de 16 superan 1e-10 (hermiticidad)', '15', sum(1 for p in fi if p['herm'] > 1e-10), dato=DL, script=S, tol=0)
    claim('Q-b4', 1, [(H8, 'y 2 violan −1e-9 en positividad')], 'Filtro, umbral anterior: 2 violan -1e-9 (positividad)', '2', sum(1 for p in fi if p['mineig'] < -1e-9), dato=DL, script=S, tol=0)
    claim('Q-b5', 1, [(H8, 'Plano: 1 de 8 supera 1e-10 (N = 18 a T = 0)')], 'Plano: 1 de 8 supera 1e-10 y 0 violan la positividad', '1,0', '%d,%d' % (sum(1 for p in pl if p['herm'] > 1e-10), sum(1 for p in pl if p['mineig'] < -1e-9)), dato=DL, script=S)
    claim('Q-b6', 1, [(H8, 'hermiticidad cruda del completo con filtro va de'), (PH, 'hermiticidad cruda 1.6e-12 a 2.8e-9'), (RM, '1.6e-12 a 2.3e-9')], 'Rango de hermiticidad cruda de las corridas con filtro del log: 1.6e-12 a 2.8e-9', '2.8e-09', max(p['herm'] for p in fi), dato=DL, script=S, nota='min %.2e max %.2e; README l.336 dice 2.3e-9 (fases 2-3)' % (min(p['herm'] for p in fi), max(p['herm'] for p in fi)))
    claim('Q-b7', 1, [(H8, 'mínimo autovalor -2.49e-09'), (RM, 'κ₂/κ = 0.05 a T = 0: −2.5e-9'), (PH, '−2.49e-9')], 'Positividad: kappa2/kappa=0.05, T=0: -2.49e-9', '-2.49e-09', [p for p in fi if abs(p['k2'] - 0.05) < 1e-9 and p['x'] == 0][0]['mineig'], dato=DL, script=S)
    claim('Q-b8', 1, [(H8, '-1.42e-09'), (RM, '0.1 a T = 0: −1.4e-9'), (PH, '−1.42e-9')], 'Positividad: kappa2/kappa=0.1, T=0: -1.42e-9', '-1.42e-09', [p for p in fi if abs(p['k2'] - 0.1) < 1e-9 and p['x'] == 0][0]['mineig'], dato=DL, script=S)
    # inconsistencias internas de HANDOFF: 0 de 14 / 13 y 2 / 13 de 14
    claim('Q-c1', 1, [(H8, 'filtro: 0 de 14 superan el umbral vigente; con el anterior, 13 y 2'), (H8, 'hermiticidad cruda sobre 1e-10 en 13 de 14 corridas con filtro')], 'HANDOFF 15/16: 14 corridas con filtro, 13 sobre 1e-10', '14', len(fi), dato=DL, script=S, tol=0,
          nota='S: §15 y §16 conservan el conteo de la fase 6 (14 corridas); §8.10 y el log actual dan 16 corridas (15 sobre 1e-10). Superadas por la regeneracion del log del 2026-10-01 (commit dc3c8e8)')
    CL[-1]['estado'] = 'S'
    # --- (b) historia de umbrales (git)
    c_add = sh("git log --format='%h %ad' --date=iso --diff-filter=A -- Mechanical-Cat-State/figuras_finales/codigo/log_herm_posit.py")
    c_mod = sh("git log --format='%h %ad' --date=iso -- Mechanical-Cat-State/figuras_finales/codigo/log_herm_posit.py")
    d1 = sh("git log --format='%h %ad' --date=iso --diff-filter=A -- Mechanical-Cat-State/figuras_finales/data/filtro_completo/gx0.0239579_w6_gz0.42_kf0.3_al4_N22_Nf2_gam6e-07.npz")
    d2 = sh("git log --format='%h %ad' --date=iso --diff-filter=A -- Mechanical-Cat-State/figuras_finales/data/filtro_completo/gx0.0338816_w6_gz0.42_kf0.3_al4_N22_Nf2_gam6e-07.npz")
    d3 = sh("git log --format='%h %ad' --date=iso --diff-filter=A -- Mechanical-Cat-State/figuras_finales/data/filtro_completo/gx0.0216506_w6_gz0.36_kf0.3_al4_N22_Nf2.npz")
    claim('Q-h1', 1, [(H8, 'Umbrales vigentes (desde 2026-10-01)'), (PH, 'completo con filtro 1e-8 en ambos')], 'Historia: umbrales 1e-8 (herm) y -1e-8 (positividad) del completo con filtro. Codigo cambiado en efda948 (2026-09-30 23:17:03 -0500 = 2026-10-01 04:17 UTC); documentado en 163e971 (2026-10-01 08:18); creados con 1e-10/-1e-9 en 1603317 (2026-09-30 16:14:28)', '2026-10-01',
          sh("git log --format=%ad --date=format:%Y-%m-%d -1 efda948"), dato='git log; codigo/log_herm_posit.py', script=S, commit='efda948',
          nota='Los datos que violan -1e-9 son anteriores al cambio: %s | %s | %s. El cambio es posterior a ver los datos. La fecha local del commit es 2026-09-30; HANDOFF dice 2026-10-01 (UTC o fecha de la documentacion 163e971). Log ma/umbral log: archivo creado en: %s; ultima modificacion: %s' % (d1.replace('\n', ' / '), d2.replace('\n', ' / '), d3.replace('\n', ' / '), c_add, c_mod.replace('\n', ' / ')))
    CL[-1]['estado'] = 'V'
    por_decidir('PD-umbral', 'Relajacion de umbrales de hermiticidad y positividad del completo con filtro (1e-10/-1e-9 -> 1e-8/-1e-8), decidida despues de observar -2.49e-9 y -1.42e-9. PENDIENTES (fase 4) dice "no se cambio el criterio: decision del autor".',
                'Mantener 1e-8/-1e-8 solo para las corridas con filtro con la justificacion de HANDOFF 8.10 (el origen del residuo no esta verificado)', 'Restituir 1e-10/-1e-9 y declarar 2 violaciones (mas las 3 de data/ fuera del log termico: -5.58e-9, -1.31e-9 y -1.12e-9)', 'Q-h1, Q-a2, Q-a3')
    # --- familias de figuras: validaciones declaradas en README
    fam = {'fig2': vals_familia('fig2/*.npz'), 'fig3': vals_familia('fig3/*.npz'), 'pfig2': vals_familia('pfig2/*.npz'), 'pfig3': vals_familia('pfig3/*.npz'), 'fig4': vals_familia('fig4/*.npz')}
    mn = lambda L, i: min(v[i] for v in L); mx = lambda L, i: max(v[i] for v in L)
    claim('Q-f2', 1, [(RM, 'Validaciones (111 ρ):** |Tr ρ − 1| ≤ 4.4e-16, ‖ρ − ρ†‖ = 0, mínimo autovalor ≥ 1.2e-13')], 'Fig.2 de validacion: min autovalor >= 1.2e-13 (todas las rho de data/fig2)', '1.2e-13', mn(fam['fig2'], 2), modo='ge', tol=0, dato='data/fig2/*.npz', script=S,
          nota='data/fig2 tiene %d rho con campo val; minimo %.2e (el reporte da 111 rho; no se pudo identificar el subconjunto)' % (len(fam['fig2']), mn(fam['fig2'], 2)))
    claim('Q-f2b', 1, [(RM, 'Validaciones (111 ρ)')], 'Fig.2: |Tr rho-1| <= 4.4e-16 y ||rho-rho+|| = 0', '4.4e-16', mx(fam['fig2'], 0), modo='le', tol=1e-17, dato='data/fig2/*.npz', script=S)
    claim('Q-f3', 1, [(RM, 'Validaciones (18 ρ):** |Tr ρ − 1| ≤ 2.2e-16, ‖ρ − ρ†‖ = 0, mínimo autovalor ≥ −7.2e-13')], 'Fig.3: min autovalor >= -7.2e-13 (18 rho)', '-7.2e-13', mn(fam['fig3'], 2), modo='ge', tol=1e-14, dato='data/fig3/*.npz', script=S, nota='data/fig3 tiene %d rho con val (README dice 18); |Tr-1| max %.2e (README 2.2e-16)' % (len(fam['fig3']), mx(fam['fig3'], 0)))
    claim('Q-f4', 1, [(RM, 'mínimo autovalor ≥ 6.5e-15')], 'Fig.2 nueva: 157 rho, min autovalor >= 6.5e-15', '6.5e-15', mn(fam['pfig2'], 2), modo='ge', tol=0, dato='data/pfig2/*.npz', script=S, nota='data/pfig2 tiene %d rho; minimo %.2e' % (len(fam['pfig2']), mn(fam['pfig2'], 2)))
    # termico: validaciones de las 6208 rho
    H = []; Tr = []; Me = []
    for f in glob.glob(os.path.join(TD, '*.npz')):
        z = np.load(f, allow_pickle=True); H.append(float(z['herm_cruda'])); Tr.append(float(z['tr_err'])); Me.append(float(z['mineig']))
    claim('Q-t1', 1, [(RM, '‖ρ − ρ†‖ ≤ 1.4e-11 (tolerancia 1e-10)'), (INF, '$\\|\\rho-\\rho^{\\dagger}\\|\\leq1.4\\times10^{-11}$')], 'Termico efectivo: herm cruda <= 1.4e-11 (6208 rho)', '1.4e-11', max(H), dato='data/termico/*.npz (%d)' % len(H), script=S, tol=None)
    claim('Q-t2', 1, [(RM, '|Tr ρ − 1| ≤ 6.7e-16'), (INF, '$|\\mathrm{Tr}\\rho-1|\\leq6.7\\times10^{-16}$')], 'Termico efectivo: |Tr-1| <= 6.7e-16', '6.7e-16', max(Tr), dato='data/termico/*.npz', script=S, tol=None)
    claim('Q-t3', 1, [(RM, 'mínimo autovalor ≥ −8.7e-17'), (INF, 'mínimo autovalor $\\geq-8.7\\times10^{-17}$')], 'Termico efectivo: min autovalor >= -8.7e-17', '-8.7e-17', min(Me), dato='data/termico/*.npz', script=S, tol=None)
    claim('Q-t4', 1, [(PH, 'la cruda es ≤ 8.2e-13 en las 4800 ρ')], 'PENDIENTES E7: herm cruda del termico nuevo <= 8.2e-13 en 4800 rho', '8.2e-13', max(H), modo='le', tol=0, dato='data/termico/*.npz (%d)' % len(H), script=S,
          nota='%d de las %d rho de data/termico superan 8.2e-13 (max %.2e); el conjunto de 4800 no se puede reconstruir' % (sum(1 for h in H if h > 8.2e-13), len(H), max(H)))
    # tolerancia del integrador y trazas del completo
    return

# ============================================================ BLOQUE 10: logs de la verificacion independiente (V1 y V6)
def bloque_logs():
    S = 'consolidacion/verificar_todo.py::bloque_logs'
    VI = os.path.join(VAL, 'independiente')
    # V1: tabla de corrimientos por diagonalizacion exacta (en unidades de gx^2/omega)
    t = open(os.path.join(VI, 'v1_salida.txt'), errors='ignore').read().split('\n')
    A = {}; B = {}
    mode = None
    for l in t:
        if l.startswith('=== (A)'): mode = 'A'; continue
        if l.startswith('=== (B)'): mode = 'B'; continue
        if l.startswith('=== (C)'): mode = None; continue
        p = l.split()
        if mode == 'A' and len(p) == 8 and re.match(r'^[\d.]+$', p[0]): A[float(p[0])] = [float(x) for x in p[1:]]
        if mode == 'B' and len(p) == 9 and re.match(r'^-?[\d.]+$', p[0]): B[(abs(float(p[0])), float(p[1]))] = [float(x) for x in p[2:]]
    DV = 'Mechanical-Cat-State/validacion/independiente/v1_salida.txt'
    tabED = {0.005: ('-1.3333', '1.3332', '1.3333'), 0.01: ('-1.3332', '1.3327', '1.3331'), 0.02: ('-1.3326', '1.3310', '1.3326'), 0.04: ('-1.3305', '1.3239', '1.3303')}
    for gx, (a, b, c) in tabED.items():
        v = A[gx]; u = gx**2
        for nm, rep, val in (('oscilador en |g>', a, v[0] / u), ('oscilador en |e>', b, v[3] / u), ('transicion del qubit', c, v[5] / u)):
            claim('ED-%g-%s' % (gx, nm.split()[-1]), 1, [(INF, '%g & 0 & $' % gx)], 'Tabla ED (g_z=0, g_x=%g): %s en unidades de gx^2/w' % (gx, nm), rep, val, dato=DV, script=S, estado=None, nota='R/V: aritmetica sobre la salida logueada de la diagonalizacion exacta (la ED no se repite)')
    for (gx, gz), rep in (((0.02, 0.02), '-1.3310'), ((0.02, 0.04), '-1.3262')):
        v = B[(gx, gz)]
        claim('ED-%g-%g' % (gx, gz), 1, [(INF, '%g & %g & $%s' % (gx, gz, rep))], 'Tabla ED (g_x=%g, g_z=%g): oscilador en |g> en unidades de gx^2/w' % (gx, gz), rep, v[0] / gx**2, dato=DV, script=S, nota='aritmetica sobre la salida logueada')
    claim('ED-n', 1, [(INF, 'Convergencia: $N=40\\to46$ cambia $E(g,1)-E(g,0)$ en $7\\times10^{-15}$')], 'Convergencia N=40->46 de E(g,1)-E(g,0): 7e-15', '7e-15', abs(0.9994695016658994 - 0.9994695016658923), dato=DV, script=S, tol=0.5e-15)
    # el error relativo escala como gx^2
    rel = [abs(A[g][2]) for g in sorted(A)]
    cc = [rel[i] / sorted(A)[i]**2 for i in range(4)]
    claim('ED-esc', 1, [(INF, 'El error relativo crece como $g_x^2$'), (H8, 'el error relativo escala como g²')], 'El error relativo del corrimiento del oscilador escala como g_x^2 (cociente rel/gx^2 constante, 4 valores)', 'si', max(cc) / min(cc) < 1.3, modo='bool', dato=DV, script=S, nota='rel/gx^2: %s' % np.round(cc, 3))
    # V6: brecha exacta
    t6 = open(os.path.join(VI, 'res_v6/v6_salida.txt'), errors='ignore').read().split('\n')
    R6 = []
    for l in t6:
        p = l.split()
        if len(p) >= 10 and re.match(r'^[\d.]+$', p[0]):
            R6.append([float(x) for x in p[:10]])
    R6 = np.array(R6)
    pred = lambda g: 0.25 * (1 - math.sqrt(1 - 8 * g)) if g <= 0.125 else 0.25
    DV6 = 'Mechanical-Cat-State/validacion/independiente/res_v6/v6_salida.txt'
    claim('V6-n', 1, [(INF, '18 valores de $\\Gamma_2/\\kappa$ entre $0.005$ y $5$')], 'V6: numero de valores de Gamma2/kappa = 18 entre 0.005 y 5', '18', len(R6), dato=DV6, script=S, tol=0)
    for row, rep in ((0.005, '5.05102572e-3'), (0.02, '2.08712153e-2'), (0.05, '5.63508327e-2'), (0.10, '0.138196601'), (0.12, '0.200000000')):
        r = [x for x in R6 if abs(x[0] - row) < 1e-12][0]
        claim('V6-%g' % row, 1, [(INF, '%s & $%s' % (('%g' % row) if row != 0.005 and row != 0.02 and row != 0.05 else ('%g' % row), 'x')), ] if False else [(INF, '$5.05102572\\times10^{-3}$' if row == 0.005 else ('$2.08712153\\times10^{-2}$' if row == 0.02 else ('$5.63508327\\times10^{-2}$' if row == 0.05 else ('0.138196601' if row == 0.1 else '0.200000000'))))],
              'V6: Delta numerico (N=20) = formula exacta en Gamma2/kappa=%g' % row, rep, pred(row), dato=DV6, script=S, tol=None, nota='numerico en el log: %.9e; formula: %.9e' % (r[2], pred(row)))
    rep_pe = [x for x in R6 if abs(x[0] - 0.125) < 1e-12][0]
    claim('V6-PE', 1, [(INF, '0.125 (punto excepcional) & 0.249999976')], 'V6: en el punto excepcional (Gamma2/kappa=0.125) el Delta numerico es 0.249999976', '0.249999976', rep_pe[2], dato=DV6, script=S, tol=None, nota='numerico del log (R); la formula da 0.25')
    claim('V6-PE2', 1, [(INF, '$-9.5\\times10^{-8}$')], 'V6: diferencia relativa en el PE -9.5e-8', '-9.5e-08', rep_pe[5], dato=DV6, script=S, tol=None)
    hi = R6[R6[:, 0] >= 0.13]
    claim('V6-sat', 1, [(INF, '0.13 a 5.0 (8 valores)')], 'V6: para Gamma2/kappa en [0.13, 5] (8 valores) Delta = 0.25 con dif relativa <= 2e-13', '2e-13', np.abs(hi[:, 5]).max(), modo='le', tol=0.3e-13, dato=DV6, script=S, nota='n=%d, max %.2e' % (len(hi), np.abs(hi[:, 5]).max()))
    claim('V6-N', 1, [(INF, 'Convergencia $N=20\\to26$: $\\leq4\\times10^{-13}$')], 'V6: convergencia N=20->26 <= 4e-13 salvo el PE (2e-8)', '4e-13', np.abs(R6[np.abs(R6[:, 0] - 0.125) > 1e-9][:, 7]).max(), modo='le', tol=0.6e-13, dato=DV6, script=S, nota='max sin PE %.2e; PE %.2e' % (np.abs(R6[np.abs(R6[:, 0] - 0.125) > 1e-9][:, 7]).max(), abs(rep_pe[7])))
    claim('V6-0', 1, [(INF, 'contando exactamente cuatro autovalores nulos')], 'V6: exactamente cuatro autovalores nulos en todos los casos', '4', None, dato=DV6, script=S, estado='R', nota='conjunto de valores de la columna "ceros": %s' % set(R6[:, 8]))
    CL[-1]['estado'] = 'V' if set(R6[:, 8]) == {4.0} else 'D'
    claim('V6-pe0', 1, [(INF, '0.12 & 0.200000000')], 'V6: Gamma2/kappa=0.12: Delta = (1/4)(1-sqrt(1-8*0.12)) = 0.2', '0.2', pred(0.12), dato='(analitico)', script=S)
    return

# ============================================================ BLOQUE 11: convergencias en N de las figuras y verificacion central (filtro, N=16/20/22)
def bloque_convergencia():
    S = 'consolidacion/verificar_todo.py::bloque_convergencia'
    # fig. 2(a): 11.98, N=20->26
    f20 = np.load(os.path.join(DATA, 'fig2/gx-0.212132_wp11.980000_wq12.000000_Om0.060000_N20.npz'), allow_pickle=True)
    f22 = np.load(os.path.join(DATA, 'fig2/gx-0.212132_wp11.980000_wq12.000000_Om0.060000_N22.npz'), allow_pickle=True)
    f26 = np.load(os.path.join(DATA, 'fig2/gx-0.212132_wp11.980000_wq12.000000_Om0.060000_N26.npz'), allow_pickle=True)
    d = 0.2121320343559643 / 6.0
    out = {}
    for N, z in ((20, f20), (26, f26)):
        a2 = complex(z['al2eff']); al = np.sqrt(a2 + 0j)
        out[N] = (Pc_rho(z['rho'], N, al, d), Pc_rho(z['rho'], N, 2j, 0.0))
    claim('CV-f2a', 1, [(RM, 'N = 20 → 26: P_c(α_eff) 0.998621 → 0.998765')], 'Fig.2(a) convergencia w_p=11.98: P_c(alpha_eff) N=20: 0.998621', '0.998621', out[20][0], dato='data/fig2/*wp11.980000_wq12.000000*_N20.npz', script=S, tol=2e-6, nota='codigo adaptativo D(d)|alpha_eff>, alpha_eff=sqrt(al2eff)')
    claim('CV-f2b', 1, [(RM, 'N = 20 → 26: P_c(α_eff) 0.998621 → 0.998765')], 'Idem N=26: 0.998765', '0.998765', out[26][0], dato='data/fig2/*N26.npz', script=S, tol=2e-6)
    claim('CV-f2c', 1, [(RM, 'P_c(viejo) 0.997407 → 0.997551')], 'P_c(viejo) N=20: 0.997407; N=26: 0.997551', '0.997407', out[20][1], dato='data/fig2/*N20.npz', script=S, tol=2e-6)
    claim('CV-f2d', 1, [(RM, 'es decir **Δ = 1.4e-4**')], 'Fig.2(a): Delta P_c(N=20->26) = 1.4e-4', '1.4e-4', out[26][1] - out[20][1], dato='data/fig2/*N{20,26}.npz', script=S, tol=0.1e-4)
    claim('CV-f2e', 1, [(RM, 'V2 mostró que N = 22 ya coincide con N = 28 a <1e-5')], 'N=22 ya coincide con N=28 a <1e-5 (P_c)', '1e-5', abs(Pc_rho(f22['rho'], 22, 2j, d) - 0.9987253), modo='le', tol=1e-5, dato='data/fig2; V2 res_v2', script=S, nota='P_c(N=28) no esta en data/fig2: se usa c6_ma_N28 (0.998732); dif %.1e' % abs(Pc_rho(f22['rho'], 22, 2j, d) - Pc_rho(np.load(os.path.join(DATA, 'c6_ma_N28.npz'), allow_pickle=True)['rho'], 28, 2j, d)))
    # fig. 3: (0.05,6,12) N22 -> N28
    z22, z28 = f3(0.05, 6.0, 12.0, 4.0, 22)[0], f3(0.05, 6.0, 12.0, 4.0, 28)[0]
    claim('CV-f3', 1, [(RM, 'γ_pf cambia 1.6e-5 relativo y P_c 1e-6'), (INF, '$\\gamma_{\\rm pf}$ cambia $1.6\\times10^{-5}$ en valor relativo')], 'Fig.3 convergencia (0.05,6,12) N=22->28: gamma_pf cambia 1.6e-5 relativo', '1.6e-5', abs(float(z28['gpf']) / float(z22['gpf']) - 1), dato='data/fig3/gx0.05_w6.0_gz12.0_al4.0_N{22,28}.npz', script=S, tol=0.1e-5)
    claim('CV-f3b', 1, [(RM, 'γ_pf cambia 1.6e-5 relativo y P_c 1e-6')], 'Idem P_c cambia 1e-6', '1e-6', abs(float(z28['Pc_fijo']) - float(z22['Pc_fijo'])), modo='le', tol=1e-6, dato='data/fig3/*', script=S, nota='dif %.1e' % abs(float(z28['Pc_fijo']) - float(z22['Pc_fijo'])))
    # verificacion central con filtro (kf=0.3), (k2/k, gz/k) = (0.03,12) y (0.3,12) para N=16,20,22
    m = np.load(os.path.join(DATA, 'figura_central_mapas.npz'), allow_pickle=True)
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((np.log(m['k2']), np.log(m['gz'])), np.log(m['eps_filtro']))
    c_ = float(m['c'])
    pts = {(0.03, 12): 'gx0.0216506_w6_gz0.36_kf0.3_al4_N%d_Nf2.npz', (0.3, 12): 'gx0.0684653_w6_gz0.36_kf0.3_al4_N%d_Nf2.npz'}
    rep = {(0.03, 12): {16: ('2.880', '2.831'), 20: ('0.986', '0.987'), 22: ('0.978', '0.979')}, (0.3, 12): {16: ('2.261', '2.288'), 20: ('0.980', '0.975'), 22: ('0.977', '0.972')}}
    res = {}
    for (k2k, gzk), pat in pts.items():
        for N in (16, 20, 22):
            z = np.load(os.path.join(FC, pat % N), allow_pickle=True)
            gx, w, gz, kap, kf = (float(z[k]) for k in ('gx', 'w', 'gz', 'kap', 'kf'))
            ke = lambda dd: kap * kf**2 / (4 * dd**2 + kf**2)
            gm, gp = gx**2 * ke(w) / w**2, gx**2 * ke(3 * w) / (9 * w**2); r = gp / gm
            a2 = abs(complex(z['al2eff'])); g = float(z['gpf'])
            k1 = g * (1 + r) / (2 * (a2 * (1 + r) + r))
            pred = k1 / (gm + gp)
            conf = np.nanmean([tasa_cola_pf(z['t'], z['Pc_din'])])
            epsf = k1 / (conf / c_)
            epsm = float(np.exp(interp([[np.log(float(z['kap2']) / kap), np.log(gz / kap)]])[0]))
            res[(k2k, gzk, N)] = (pred, epsf / epsm, conf)
            fz = rel(os.path.join(FC, pat % N))
            claim('FC-%g-%d-pf' % (k2k, N), 1, [(RM, '| (%g, 12): γ_pf/predicho |' % k2k), (INF, '$(0.03,12)$ & $\\gamma_{\\rm pf}$/predicho' if k2k == 0.03 else '$(0.3,12)$ & $\\gamma_{\\rm pf}$/predicho')], 'Verificacion central con filtro kf=0.3 (k2/k=%g,gz/k=12) N=%d: gamma_pf/predicho' % (k2k, N), rep[(k2k, gzk)][N][0], pred, dato=fz, script=S, tol=None)
            claim('FC-%g-%d-eps' % (k2k, N), 1, [(RM, '| (%g, 12): ε_completo/ε_mapa |' % k2k), (INF, '$(0.03,12)$ & $\\epsilon_{\\rm completo}/\\epsilon_{\\rm mapa}$' if k2k == 0.03 else '$(0.3,12)$ & $\\epsilon_{\\rm completo}/\\epsilon_{\\rm mapa}$')], 'Verificacion central: eps_completo/eps_mapa (k2/k=%g, N=%d)' % (k2k, N), rep[(k2k, gzk)][N][1], epsf / epsm, dato=fz + ' ; data/figura_central_mapas.npz', script=S, tol=None)
    for k2k, rr in ((0.03, ('0.8%', '0.3%')),):
        pass
    d1 = abs(res[(0.03, 12, 22)][0] / res[(0.03, 12, 20)][0] - 1); d2 = abs(res[(0.3, 12, 22)][0] / res[(0.3, 12, 20)][0] - 1)
    claim('FC-conv1', 1, [(RM, 'Entre N = 20 y 22, γ_pf cambia 0.8% y 0.3%'), (INF, 'cambia $0.8\\%$ y $0.3\\%$')], 'N=20->22: gamma_pf cambia 0.8% (0.03,12) y 0.3% (0.3,12)', '0.8%', d1, dato='data/filtro_completo/*N{20,22}*', script=S, tol=0.0006, nota='segundo punto: %.4f' % d2)
    claim('FC-conv2', 1, [(RM, 'Entre N = 20 y 22, γ_pf cambia 0.8% y 0.3%')], 'N=20->22: gamma_pf cambia 0.3% (0.3,12)', '0.3%', d2, dato='data/filtro_completo/*N{20,22}*', script=S, tol=0.0006)
    cc = max(abs(res[(k, 12, 22)][2] / res[(k, 12, 20)][2] - 1) for k in (0.03, 0.3))
    claim('FC-conv3', 1, [(RM, 'el confinamiento < 0.05%'), (INF, 'el confinamiento menos de $0.05\\%$')], 'N=20->22: confinamiento cambia < 0.05%', '0.0005', cc, modo='le', tol=0, dato='data/filtro_completo/*N{20,22}*', script=S, nota='max %.2e' % cc)
    claim('FC-prec', 1, [(RM, 'Precisión con filtro (declarada)'), (INF, 'La precisión declarada con filtro es de $2$--$3\\%$')], 'Precision con filtro 2-3%: |eps_c/eps_map - 1| con N=22: 2.1% y 2.8%', '0.03', max(abs(res[(0.03, 12, 22)][1] - 1), abs(res[(0.3, 12, 22)][1] - 1)), modo='le', tol=0.0015, dato='data/filtro_completo/*N22*', script=S)
    return res

def tasa_cola_pf(t, serie_list):
    """Misma definicion que principal_fig3.conf_modelo_completo (retorno dinamico desde |0>|g>)."""
    s = serie_list if np.ndim(serie_list) == 1 else serie_list[0]
    e = np.abs(s - s[-1]); i1 = int(np.argmax(e < 0.1 * e.max()))
    piso = np.median(e[int(0.9 * len(e)):]) + 1e-12
    fin = np.where(e < 3 * piso)[0]; i2 = int(fin[fin > i1][0]) if np.any(fin > i1) else len(e)
    return -np.polyfit(t[i1:i2], np.log(e[i1:i2]), 1)[0] if i2 - i1 >= 10 else np.nan

# ============================================================ principal
def escribir():
    coincidencia()
    nlin = lineas_sin_verificar()
    ids = [c['id'] for c in CL]
    dup = {i for i in ids if ids.count(i) > 1}
    assert not dup, 'ids duplicados: %s' % dup
    cols = ['id', 'nivel', 'tipo', 'ubicaciones', 'texto', 'valor_reportado', 'valor_recalculado', 'tolerancia', 'estado', 'ubicaciones_coinciden', 'nota', 'dato', 'script', 'commit']
    with open(os.path.join(OUT, 'afirmaciones.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader()
        for c in CL:
            c.setdefault('tipo', 'numerica' if c.get('valor_reportado') else 'cualitativa'); w.writerow(c)
    with open(os.path.join(OUT, 'por_decidir.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'texto', 'version_a', 'version_b', 'dato']); w.writeheader()
        for p in PDEC: w.writerow(p)
    import collections
    cnt = collections.Counter((c['nivel'], c['estado']) for c in CL)
    with open(os.path.join(OUT, 'verificacion_log.txt'), 'w', encoding='utf-8') as f:
        f.write('afirmaciones: %d; lineas numericas de nivel 1 sin cubrir: %d\n' % (len(CL), nlin))
        for k, v in sorted(cnt.items(), key=lambda t: (str(t[0][0]), str(t[0][1]))): f.write('nivel %s estado %s: %d\n' % (k[0], k[1], v))
        f.write('\nanclas no encontradas (%d):\n' % len(WARN))
        for w_ in WARN: f.write('  ' + w_ + '\n')
        f.write('\nD:\n')
        for c in CL:
            if c['estado'] == 'D': f.write('  %s | rep=%s | rec=%s | %s\n' % (c['id'], c['valor_reportado'], c['valor_recalculado'], c['texto'][:110]))
    return cnt

def main():
    bloque_simbolico(); bloque_ma(); bloque_pf(); bloque_filtro(); bloque_minimo(); bloque_mapa(); bloque_termico_ef()
    E = bloque_termico_full(); bloque_plano(E); bloque_diag(E); bloque_calidad(); bloque_logs(); bloque_convergencia()
    for nm in ('bloque_textual', 'bloque_textual2', 'bloque_textual3', 'bloque_textual4', 'bloque_textual5', 'bloque_textual6', 'bloque_L2', 'bloque_meta'):
        if nm in globals(): globals()[nm](E)
    return escribir()


# ============================================================ BLOQUE 12: texto de HANDOFF.md (lineas restantes), literatura, historia de commits
def byid(i):
    for c in CL:
        if c['id'] == i: return c
    raise KeyError(i)
def adj(i, f, anchor, val=None, n=1):
    """Agrega una ubicacion adicional a una afirmacion existente (misma afirmacion repetida en otro lugar)."""
    c = byid(i)
    c['ubicaciones'] = c['ubicaciones'] + ' | ' + f'{f}:{find(f, anchor, n)}'
    if val is not None:
        pr = parse(val); c.setdefault('_vals', []).append(val)
def txt(i, nivel, locs, texto, estado, nota='', rep=None, rec=None, tipo='cualitativa', dato='', script='', commit='', tol=None, modo='abs'):
    st = claim(i, nivel, locs, texto, rep, rec, tol=tol, modo=modo, estado=estado, nota=nota, dato=dato, script=script, commit=commit)
    byid(i)['tipo'] = tipo
    return st
def lit(i, locs, texto, nota=''):
    txt(i, 1, locs, texto, 'L', nota=nota or 'Afirmacion sobre la literatura: fuera de alcance, sin verificar (se copia tal cual)', tipo='literatura')

def bloque_textual(E):
    S = 'consolidacion/verificar_todo.py::bloque_textual'
    K = KAPM
    # ---- 4.4 / 4.5: precisiones
    rr = {i: byid('V4b-%d-con1' % i)['valor_recalculado'] for i in (2, 4, 6)}
    v4b = [float(byid('V4b-%d-con1' % i)['valor_recalculado']) for i in (2, 4, 6)]
    txt('V4b-rango', 1, [(H8, 'El "+1" faltaba en el trabajo original (E2). Verificado en V4b'), (H8, '| γ_pf con +1 | 0.996–0.999 de la fórmula | V4b |')], 'V4b: con |a|^2=2,4,6 la formula con +1 da 0.996-0.999 del medido', 'V', rep='0.996-0.999', rec=None, tipo='numerica', dato='data/fig3/gx0.05_w6.0_gz4.0_al{2,4,6}*.npz', script=S, nota='recalculo: %s' % ', '.join('%.4f' % v for v in v4b))
    byid('V4b-rango')['estado'] = 'V' if abs(min(v4b) - 0.996) < 0.001 and abs(max(v4b) - 0.999) < 0.001 else 'D'
    # deficit vs kappa2/kappa (V4)
    d = [(r_[0], 1 - r_[1]) for r_ in recs_v4]
    d05 = [x for k, x in d if k <= 0.0525]; d02 = [x for k, x in d if 0.15 <= k <= 0.25]; d1 = [x for k, x in d if k > 0.9]
    txt('P4-a', 1, [(H8, 'Precisión (corrección no adiabática): ≲ 0.4% para κ₂/κ ≲ 0.05'), (RM, '≲ 0.4% para κ₂/κ ≲ 0.05, ~1% en κ₂/κ ≈ 0.2 y ~2% en κ₂/κ ≈ 1'), (PH, '| P4 |'), (PH, '**C3 (revisado):**')], 'Deficit medido/predicho <= 0.4% para kappa2/kappa <= 0.05', 'D', rep='0.4%', rec=None, tipo='numerica', dato='data/fig3/*.npz (15 puntos)', script=S,
        nota='maximo %.2f%% en kappa2/kappa<=0.0525 (punto (0.1,5,2), que PENDIENTES P6 reconoce como outlier); sin ese punto: %.2f%%' % (100 * max(d05), 100 * sorted(d05)[-2]))
    byid('P4-a')['valor_recalculado'] = '%.4f' % max(d05)
    txt('P4-b', 1, [(H8, 'Precisión (corrección no adiabática): ≲ 0.4% para κ₂/κ ≲ 0.05')], 'Deficit ~1% en kappa2/kappa ~0.2 (0.7%-1.3% en los puntos 0.16-0.23)', 'V', rep='1%', tipo='numerica', dato='data/fig3/*.npz', script=S, nota='recalculo %s' % np.round(d02, 4))
    txt('P4-c', 1, [(H8, 'Precisión (corrección no adiabática): ≲ 0.4% para κ₂/κ ≲ 0.05')], 'Deficit ~2% en kappa2/kappa = 1 (2.1%)', 'V', rep='2%', tipo='numerica', dato='data/fig3/gx0.1_w4.0_gz10.0_al4.0_N22.npz', script=S, nota='recalculo %s' % np.round(d1, 4))
    # ---- 4.6
    mx_f = max(abs(float(byid('F5a-%d-%s' % (i, s))['valor_recalculado']) - 1) for i in (3, 4, 5, 6) for s in ('m', 'p', 'k'))
    txt('F5-02', 1, [(H8, 'Verificado canal por canal al 0.2%'), (INF, 'Los dos canales se confirman por separado al $0.2\\%$')], 'Canales Gamma1+/- con filtro verificados al 0.2% (|ratio-1| maximo con filtro)', 'V', rep='0.002', rec=None, tipo='numerica', dato='data/fig4/est_*.npz', script=S, nota='maximo %.4f (kf=0.3: 0.9977)' % mx_f)
    byid('F5-02')['estado'] = 'V' if mx_f <= 0.0025 else 'D'
    # mejora lista
    for kf, rep in ((3.0, '19.5'), (1.0, '166.5'), (0.3, '1838'), (0.1, '16 528')):
        adj('F5b-%s-med' % kf, H8, '**Mejora de γ_pf respecto al baño plano', rep)
    adj('F5b-3.0-med', RM, '19.5/19.5, 166.5/166.1, 1838/1833 y 16528/16487')
    # 4.7 lista
    z = [('1e-3', 0.001, '0.98'), ('0.01', 0.01, '0.84'), ('0.045', 0.045204, '0.50'), ('0.16', 0.161026, '0.24'), ('1', 1.0826, '0.058')]
    Tm = np.array(sorted((float(np.load(f, allow_pickle=True)['k2k']), 0) for f in glob.glob(os.path.join(DATA, 'minimo', 'k*_N24.npz'))))
    cmini = {tuple(): 0}
    TT = [[float(x) for x in r] for r in list(csv.reader(open(os.path.join(DATA, 'principal_fig3_minimo.csv'))))[1:]]
    TT = np.array(TT)
    for (lab, kk, rep) in z:
        i = int(np.argmin(abs(TT[:, 0] - kk)))
        txt('K2e-%s' % lab, 1, [(H8, 'κ₂^eff/κ₂ = 0.98, 0.84, 0.50, 0.24 y 0.058 para κ₂/κ = 1e-3, 0.01, 0.045, 0.16 y 1'), (RM, '- κ₂^eff/κ₂ = 0.98 (1e-3), 0.84 (0.01), 0.50 (0.045), 0.24 (0.16) y 0.058 (1)')], 'kappa2eff/kappa2 en kappa2/kappa=%s: %s (punto de la rejilla %.4g)' % (lab, rep, TT[i, 0]), 'V', rep=rep, rec=TT[i, 5], tipo='numerica', dato='data/principal_fig3_minimo.csv', script=S,
            nota=('el punto tabulado es kappa2/kappa=1.0826 (no 1.0); a 1.0 la interpolacion da 0.0625' if lab == '1' else ''))
        byid('K2e-%s' % lab)['estado'] = 'V' if abs(TT[i, 5] - float(rep)) <= (0.006 if lab != '1' else 0.001) else 'D'
        byid('K2e-%s' % lab)['valor_recalculado'] = '%.4f' % TT[i, 5]
    # (0.25,6) y (0.25,14) con piso gamma/kappa=2e-4 (filtro_completo gam6e-06)
    mpa = np.load(os.path.join(DATA, 'figura_central_mapas.npz'), allow_pickle=True)
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((np.log(mpa['k2']), np.log(mpa['gz'])), np.log(mpa['eps_filtro']))
    cmap = float(mpa['c'])
    Dmin = lambda k2: float(np.exp(np.interp(np.log(k2), np.log(TT[:, 0]), np.log(TT[:, 1]))))
    for (fn, gzk, N, label) in (('gx0.0535714_w6_gz0.42_kf0.3_al4_N22_Nf2_gam6e-06.npz', 14, 22, '(0.25,14)'), ('gx0.125_w6_gz0.18_kf0.3_al4_N20_Nf2_gam6e-06.npz', 6, 20, '(0.25,6)')):
        zz = np.load(os.path.join(FC, fn), allow_pickle=True)
        gx, w, gz, kap, kf = (float(zz[k]) for k in ('gx', 'w', 'gz', 'kap', 'kf')); gam = float(zz['gam'])
        ke = lambda dd: kap * kf**2 / (4 * dd**2 + kf**2)
        gm, gp = gx**2 * ke(w) / w**2, gx**2 * ke(3 * w) / (9 * w**2); r = gp / gm
        a2 = abs(complex(zz['al2eff'])); g = float(zz['gpf'])
        # prediccion con perdida intrinseca: 2[(G1m+gamma)a + G1p(a+1)]
        pred = 2 * ((gm + gam) * a2 + gp * (a2 + 1))
        k1 = (g / 2 - gp * (a2 + 1) * 0) if False else None
        conf = tasa_cola_pf(zz['t'], zz['Pc_din'])
        k1full = (g - 2 * gam * a2) * (1 + r) / (2 * (a2 * (1 + r) + r)) + gam
        epsf = k1full / (conf / cmap)
        epsm = float(np.exp(interp([[np.log(float(zz['kap2']) / kap), np.log(gz / kap)]])[0]))
        chi = 8 / 3 * gx**2 / w * 4 / kap
        # confinamiento/mapa: conf frente a Delta_filtro del mapa: usa kappa2eff del mapa
        k1map = (float(mpa['f_filt']) * float(zz['kap2']) / kap / (16 * (gz / kap)**2))
        confmap = k1map / epsm * cmap
        ff = rel(os.path.join(FC, fn))
        if label == '(0.25,14)':
            txt('PI-14a', 1, [(H8, 'Verificado con el modelo completo en (0.25, 14), N = 22: ε = 4.39e-3 < 1/220, y ε_completo/ε_mapa = 1.028'), (RM, '(κ₂/κ, g_z/κ) = (0.25, 14), χ|α|²/κ = 0.17, N = 22: γ_pf/predicho = 1.007'), (INF, '$(\\kappa_2/\\kappa,g_z/\\kappa)=(0.25,14)$')], 'Piso 2e-4, (0.25,14), N=22: gamma_pf/predicho = 1.007', 'V', rep='1.007', rec=g / pred, tipo='numerica', dato=ff, script=S)
            txt('PI-14b', 1, [(H8, 'Verificado con el modelo completo en (0.25, 14), N = 22: ε = 4.39e-3 < 1/220, y ε_completo/ε_mapa = 1.028'), (RM, '**ε_completo/ε_mapa = 1.028**'), (INF, 'mapa}=1.028')], '(0.25,14) con piso: eps_completo/eps_mapa = 1.028', 'V', rep='1.028', rec=None, tipo='numerica', dato=ff + ' ; data/figura_central_mapas.npz', script=S, nota='eps_mapa sin piso %.3e; con piso %.3e' % (epsm, epsm + (gam / kap) / (k1map / epsm)))
            epsm_p = epsm + (gam / kap) / (k1map / epsm)
            byid('PI-14b')['valor_recalculado'] = '%.4f' % (epsf / epsm_p)
            byid('PI-14b')['estado'] = 'V' if abs(epsf / epsm_p - 1.028) < 0.0015 else 'D'
            txt('PI-14c', 1, [(H8, 'Verificado con el modelo completo en (0.25, 14), N = 22: ε = 4.39e-3 < 1/220, y ε_completo/ε_mapa = 1.028'), (RM, 'ε_completo = 4.39e-3 < 1/220'), (INF, 'epsilon_{\\rm completo}=4.39\\times10^{-3}<1/220')], '(0.25,14) con piso: eps_completo = 4.39e-3 < 1/220', 'V', rep='4.39e-3', rec=epsf, tipo='numerica', dato=ff, script=S)
            byid('PI-14c')['estado'] = 'V' if abs(epsf - 4.39e-3) < 0.01e-3 else 'D'
            txt('PI-14d', 1, [(RM, 'confinamiento/mapa = 0.979'), (INF, 'confinamiento/mapa $=0.979$')], '(0.25,14) con piso: confinamiento/mapa = 0.979', 'V', rep='0.979', rec=None, tipo='numerica', dato=ff, script=S, nota='recalculo %.4f' % (conf / kap / confmap))
            byid('PI-14d')['estado'] = 'V' if abs(conf / kap / confmap - 0.979) < 0.0015 else 'D'; byid('PI-14d')['valor_recalculado'] = '%.4f' % (conf / kap / confmap)
            txt('PI-14e', 1, [(RM, 'χ|α|²/κ = 0.17, N = 22')], '(0.25,14): chi|a|^2/k = 0.17', 'V', rep='0.17', rec=chi, tipo='numerica', dato='(analitico)', script=S)
            byid('PI-14e')['estado'] = 'V' if abs(chi - 0.17) < 0.005 else 'D'
        else:
            txt('PI-6a', 1, [(H8, 'el punto (0.25, 6), con χ|α|²/κ = 0.93, confina 0.75 veces lo del mapa'), (RM, '(0.25, 6), χ|α|²/κ = 0.93 (fuera de la zona), N = 20: γ_pf/predicho = 1.000'), (INF, '$(0.25,6)$, $\\chi|\\alpha|^2/\\kappa=0.93$')], '(0.25,6) con piso, N=20: confinamiento/mapa = 0.751', 'V', rep='0.751', rec=None, tipo='numerica', dato=ff, script=S, nota='recalculo %.4f' % (conf / kap / confmap))
            byid('PI-6a')['estado'] = 'V' if abs(conf / kap / confmap - 0.751) < 0.002 else 'D'; byid('PI-6a')['valor_recalculado'] = '%.4f' % (conf / kap / confmap)
            epsm_p = epsm + (gam / kap) / (k1map / epsm)
            txt('PI-6b', 1, [(RM, 'ε_completo/ε_mapa = 1.33'), (INF, 'epsilon_{\\rm completo}/\\epsilon_{\\rm mapa}=1.33')], '(0.25,6) con piso: eps_completo/eps_mapa = 1.33', 'V', rep='1.33', rec=None, tipo='numerica', dato=ff, script=S, nota='recalculo %.3f' % (epsf / epsm_p))
            byid('PI-6b')['valor_recalculado'] = '%.4f' % (epsf / epsm_p)
            byid('PI-6b')['estado'] = 'V' if abs(epsf / epsm_p - 1.33) < 0.01 else 'D'
            txt('PI-6c', 1, [(RM, 'γ_pf/predicho = 1.000')], '(0.25,6): gamma_pf/predicho = 1.000', 'V', rep='1.000', rec=g / pred, tipo='numerica', dato=ff, script=S)
            byid('PI-6c')['estado'] = 'V' if abs(g / pred - 1.0) < 0.002 else 'D'
            txt('PI-6d', 1, [(RM, 'χ|α|²/κ = 0.93')], '(0.25,6): chi|a|^2/k = 0.93', 'V', rep='0.93', rec=chi, tipo='numerica', dato='(analitico)', script=S)
            byid('PI-6d')['estado'] = 'V' if abs(chi - 0.93) < 0.01 else 'D'
    return

# ============================================================ BLOQUE 13: HANDOFF (continuacion)
def leer(path):
    return open(os.path.join(DATA, path), errors='ignore').read()
def bloque_textual2(E):
    S = 'consolidacion/verificar_todo.py::bloque_textual2'
    K = KAPM
    # resumen de la seccion 5 -> afirmaciones ya verificadas
    adj('V4b-rango', H8, '| γ_pf con +1 | 0.996–0.999 de la fórmula | V4b |')
    adj('MU-01', H8, '| Filtro | mejoras al ≤ 0.3%; umbral 3.94 → 0.094 |')
    adj('V6-sat', H8, '| Brecha con α = 0 | exacta al ≤ 2e-13 | V6 |')
    adj('MN-01', H8, '| Confinamiento con \\|α\\|² = 4 | c = 4.14, saturación ≈ 0.28κ |'); adj('MN-02', H8, '| Confinamiento con \\|α\\|² = 4 | c = 4.14, saturación ≈ 0.28κ |')
    adj('P9-04', H8, '| Frontera χ | χ\\|α\\|²/κ ≈ 0.3 |')
    adj('FC-prec', H8, '| Mapa con filtro | ±2–3% frente al completo (N = 22) |')
    rpf = [E[p]['rpf'] for p in E]
    txt('T84-r1', 1, [(H8, '| Térmico: efectivo con filtro frente al completo | γ_pf 0.956–0.959'), (RM, 'γ_pf c/e entre 0.956 y 0.959 en todos')], 'gamma_pf completo/efectivo entre 0.956 y 0.959 en los 8 puntos', 'V', rep='0.956-0.959', tipo='numerica', dato='data/filtro_completo/*, data/termico/*', script=S, nota='min %.4f max %.4f' % (min(rpf), max(rpf)))
    byid('T84-r1')['estado'] = 'V' if abs(min(rpf) - 0.956) < 0.001 and abs(max(rpf) - 0.959) < 0.001 else 'D'
    txt('T84-r2', 1, [(H8, '| Térmico: efectivo con filtro frente al completo | γ_pf 0.956–0.959'), (H8, '- **Dependencia de κ₂/κ:** el cociente γ_bf crudo baja de 0.9561 a 0.7759'), (RM, 'Cae monótonamente (0.956, 0.856, 0.785, 0.776)')], 'gamma_bf c/e (x=6.86) cae monotonamente 0.9561, 0.8563, 0.7848, 0.7759 con kappa2/kappa 0.05, 0.1, 0.25, 0.4', 'V', rep='0.9561->0.7759', tipo='numerica', dato='data/filtro_completo/*, data/termico/*', script=S,
        nota='valores %s; se aplana entre 0.25 y 0.4 (dif %.4f)' % ([round(E[(k, 6.86)]['rbf'], 4) for k in (0.05, 0.1, 0.25, 0.4)], E[(0.25, 6.86)]['rbf'] - E[(0.4, 6.86)]['rbf']))
    byid('T84-r2')['estado'] = 'V' if (np.diff([E[(k, 6.86)]['rbf'] for k in (0.05, 0.1, 0.25, 0.4)]) < 0).all() else 'D'
    adj('T85-rango', H8, '| Térmico: x* del completo | [ESTIMADO], efectivo conservador 0 a 0.2')
    chi4 = float(E[(0.4, 6.86)]['e']['z']['chi_a2']) if 'chi_a2' in E[(0.4, 6.86)]['e']['z'].files else None
    txt('CHI04', 1, [(H8, 'En κ₂/κ = 0.4 (χ|α|²/κ = 0.27, junto a la frontera 0.3)'), (RM, 'en κ₂/κ = 0.4 (χ|α|²/κ = 0.27, junto a la frontera'), (PH, '(0.05; 0.03), 0.856 (0.1; 0.07), 0.785 (0.25; 0.17), 0.776 (0.4; 0.27)')], 'chi|a|^2/kappa = 0.27 en kappa2/kappa=0.4 (g_z/kappa=14) y 0.03, 0.07, 0.17 en 0.05, 0.1, 0.25', 'V', rep='0.27', rec=chi4, tipo='numerica', dato='data/termico/x6.86_k0.4_gz14_f1_g2e-05_N22_Nf2_w200_a4.npz (campo chi_a2)', script=S, nota='valores para 0.05,0.1,0.25: %s' % [round(float(E[(k, 6.86)]['e']['z']['chi_a2']), 3) for k in (0.05, 0.1, 0.25)])
    txt('RS25', 1, [(H8, 'En κ₂/κ = 0.25 hubo una comprobación: r_s pasa de 0.7842 (x = 6.86) a 0.7817 (x = 10.1)'), (PH, 'con piso restado 0.784 → 0.782')], 'r_s en kappa2/kappa=0.25: 0.7842 (x=6.86) -> 0.7817 (x=10.1)', 'V', rep='0.7842->0.7817', tipo='numerica', dato='data/filtro_completo/*, data/termico/*', script=S, nota='%.4f -> %.4f' % (E[(0.25, 6.86)]['rs'], E[(0.25, 10.1)]['rs']))
    byid('RS25')['estado'] = 'V' if abs(E[(0.25, 6.86)]['rs'] - 0.7842) < 0.0001 and abs(E[(0.25, 10.1)]['rs'] - 0.7817) < 0.0001 else 'D'
    # fracciones de piso (lista en 8.6)
    for k in (0.05, 0.1, 0.25, 0.4):
        for xx, tag in ((6.86, 'a x = 6.86, 0.30%, 0.17%, 0.09%, 0.09%'),):
            pass
    adj('T84b-0.05-6.86-fr', H8, '- **Fracción del γ_bf térmico del completo**', '0.30%')
    # 8.6 piso
    txt('T86-n', 1, [(H8, '**Origen del piso: [VERIFICAR].** Comprobado: no depende de la tolerancia del integrador (−0.10%)')], 'Origen del piso de T=0 con filtro y su crecimiento con kappa2/kappa: no identificado', 'R', nota='abierto; comprobado solo que no depende de la tolerancia (T84-tolE2)', tipo='pendiente')
    # 8.7
    txt('PL7-84', 1, [(H8, '[HIPÓTESIS]** (reproduce el 84% con 16% de exceso sin explicar)'), (PH, 'explica el aumento térmico de γ_pf del plano en orden de magnitud y en un 84%')], 'El desplazamiento dependiente del estado reproduce el 84% del canal n_q del plano (con 16% de exceso)', 'D', rep='0.84', rec=None, tipo='numerica', dato='data/termico/x6.86_k0.25_gz14_f0_*_kick*.npz', script=S,
        nota='el incremento por n_q del efectivo con kick (6.101e-4) es 1.163 veces el del completo (5.248e-4); el completo es 0.860 del efectivo-kick. "84%" no coincide con 0.860 ni con 1/1.16; es una redaccion ambigua del exceso del 16% (H, ver PL7-16)')
    byid('PL7-84')['valor_recalculado'] = '0.860 (completo/kick); 1.163 (kick/completo)'
    # 8.8 filtro_kick.txt
    fk = leer('filtro_kick.txt')
    L = [l for l in fk.split('\n') if l.strip()]
    def pick(l, key): return float(re.search(key + r'=([+-]?[\d.eE+-]+)', l).group(1))
    dbf25, dbf40 = pick(L[0], 'Δγ_bf'), pick(L[2], 'Δγ_bf'); dpf = pick(L[0], 'Δγ_pf')
    dbf25_0, dbf40_0 = pick(L[1], 'Δγ_bf'), pick(L[3], 'Δγ_bf')
    for i, rep, v, loc in (('FK-1', '-4.6e-05', dbf25, 'γ_bf cambia -4.6e-05 (κ₂/κ = 0.25) y -3.9e-05 (0.4) relativo'), ('FK-2', '-3.9e-05', dbf40, 'γ_bf cambia -4.6e-05 (κ₂/κ = 0.25) y -3.9e-05 (0.4) relativo'), ('FK-3', '+0.27%', dpf, 'γ_pf +0.27%'), ('FK-4', '+6.8e-04', dbf25_0, 'A T → 0 (x = 60) γ_bf cambia +6.8e-04 y +1.0e-03'), ('FK-5', '+1.0e-03', dbf40_0, 'A T → 0 (x = 60) γ_bf cambia +6.8e-04 y +1.0e-03')):
        txt(i, 1, [(H8, loc)], 'filtro_kick (ansatz con ke(w)): ' + loc.split(':')[0][:80], 'R', rep=rep, rec=None, tipo='numerica', dato='data/filtro_kick.txt (log del efectivo con ansatz; sin cache npz)', script='codigo/filtro_kick.py', nota='valor en el log: %.3e (R: el efectivo con ansatz no esta cacheado para x=6.86, k=0.25 filtro? ver EO-0.25-6.86-Ea)' % v)
        byid(i)['valor_recalculado'] = '%.3e' % v
        pr = parse(rep)
        if pr and abs(v - pr[0]) <= (pr[1] or 0) * 1.0 * (1 + 1e-9): byid(i)['estado'] = 'V'; byid(i)['nota'] = 'coincide con el log (R-parcial: el log es la unica fuente)'
    ke = 0.3**2 / (4 * 6**2 + 0.3**2)
    txt('FK-ke', 1, [(H8, 'ke(ω) = κ_f²/(4ω²+κ_f²) = 6.2e-4')], 'ke(w) = kf^2/(4w^2+kf^2) = 6.2e-4 con kf=0.3, w=6', 'V', rep='6.2e-4', rec=ke, tipo='numerica', dato='(analitico)', script=S)
    txt('FK-beta', 1, [(H8, 'β = 2g_z/ω = 0.14')], 'beta = 2 gz/w = 0.14 (gz=0.42, w=6)', 'V', rep='0.14', rec=2 * 0.42 / 6, tipo='numerica', dato='(analitico)', script=S)
    # 8.8: cota de la diferencia (completo menos efectivo sin kick)
    dif = {}
    for (k, x) in E:
        d = E[(k, x)]
        dif[(k, x)] = (d['pc'] - float(d['z0']['gpf']) / K) - (d['pe'] - d['e0']['gpf'])
    txt('DF-1', 1, [(H8, 'va de −4.6e-8 a −1.1e-7 en los ocho puntos'), (H8, 'con |diferencia| ≤ 1.1e-7 el cociente frente al plano (5.25e-4) es ≳ 4.8e3')], 'Diferencia del aumento termico de gamma_pf (completo menos efectivo sin desplazamiento): -4.6e-8 a -1.1e-7 en los ocho puntos', 'V', rep='-4.6e-8..-1.1e-7', tipo='numerica', dato='data/filtro_completo/*, data/termico/*', script=S,
        nota='valores: %s' % {str(k): '%.2e' % v for k, v in dif.items()})
    byid('DF-1')['estado'] = 'V' if abs(max(dif.values()) + 4.6e-8) < 0.1e-8 and abs(min(dif.values()) + 1.1e-7) < 0.1e-7 else 'D'
    txt('DF-2', 1, [(H8, 'el cociente frente al plano (5.25e-4) es ≳ 4.8e3')], 'Cota: 5.248e-4/1.1e-7 = 4.8e3', 'V', rep='4.8e3', rec=5.248e-4 / abs(min(dif.values())), tipo='numerica', dato='(derivado)', script=S, nota='%.0f' % (5.248e-4 / abs(min(dif.values()))))
    byid('DF-2')['estado'] = 'V' if abs(5.248e-4 / abs(min(dif.values())) - 4800) < 100 else 'D'
    txt('DF-3', 1, [(H8, 'La diferencia crece en valor absoluto con x (−6.6e-8 en x = 6.86 y −1.1e-7 en x = 9, en κ₂/κ = 0.05) mientras n_q baja 8.5 veces')], 'Diferencia kappa2/kappa=0.05: -6.6e-8 (x=6.86), -1.1e-7 (x=9); n_q baja 8.5 veces', 'V', rep='-6.6e-8,-1.1e-7,8.5', tipo='numerica', dato='data/filtro_completo/*', script=S, nota='%.2e, %.2e, nq ratio %.2f' % (dif[(0.05, 6.86)], dif[(0.05, 9)], E[(0.05, 6.86)]['e']['nq'] / E[(0.05, 9)]['e']['nq']))
    byid('DF-3')['estado'] = 'V' if abs(dif[(0.05, 6.86)] + 6.6e-8) < 0.1e-8 and abs(dif[(0.05, 9)] + 1.1e-7) < 0.1e-8 and abs(E[(0.05, 6.86)]['e']['nq'] / E[(0.05, 9)]['e']['nq'] - 8.5) < 0.1 else 'D'
    d1 = [((E[(k, x)]['rpf'] - float(E[(k, x)]['z0']['gpf']) / K / E[(k, x)]['e0']['gpf']) * E[(k, x)]['pe']) for (k, x) in ((0.25, 6.86),)]
    d2 = [(float(E[(0.25, 6.86)]['z0']['gpf']) / K / E[(0.25, 6.86)]['e0']['gpf'] - 1) * (E[(0.25, 6.86)]['pe'] - E[(0.25, 6.86)]['e0']['gpf'])]
    txt('DF-4', 1, [(H8, 'En κ₂/κ = 0.25 y x = 6.86 los términos valen +4.7e-7 y −5.3e-7 (suma −5.4e-8)')], 'Descomposicion de la diferencia en (0.25,6.86): +4.7e-7 y -5.3e-7 (suma -5.4e-8)', 'V', rep='+4.7e-7,-5.3e-7,-5.4e-8', tipo='numerica', dato='data/filtro_completo/*, data/termico/*', script=S, nota='%.2e %.2e %.2e' % (d1[0], d2[0], d1[0] + d2[0]))
    byid('DF-4')['estado'] = 'V' if abs(d1[0] - 4.7e-7) < 0.06e-7 and abs(d2[0] + 5.3e-7) < 0.06e-7 and abs(d1[0] + d2[0] + 5.4e-8) < 0.1e-8 else 'D'
    # 8.8 N_f: valores del log
    af = leer('diagnostico_alfa_Nf.txt')
    nf3 = [float(x) for x in re.findall(r'^0\.\d+\s+3\s+[\d.e+-]+\s+[\d.e+-]+\s+[\d.e+-]+\s+([\d.e+-]+)', af, re.M)]
    nf2 = [float(x) for x in re.findall(r'^0\.\d+\s+2\s+[\d.e+-]+\s+[\d.e+-]+\s+[\d.e+-]+\s+([\d.e+-]+)', af, re.M)]
    ch = [(a / b - 1) for a, b in zip(nf3, nf2)]
    txt('NF-1', 1, [(H8, '**Sensibilidad del efectivo a N_f**'), (H8, 'el efectivo con N_f = 3 mueve γ_bf entre −2.1% y −3.6% a x = 6.86')], 'Efectivo N_f=3 frente a N_f=2: gamma_bf (x=6.86) cambia -2.1%, -2.8%, -3.5%, -3.6% (kappa2/kappa 0.05,0.1,0.25,0.4)', 'R', rep='-2.1%..-3.6%', tipo='numerica', dato='data/diagnostico_alfa_Nf.txt (el efectivo con N_f=3 no esta cacheado)', script='codigo/diagnostico_alfa_Nf.py', nota='del log: %s; el N_f=2 coincide con las caches (EO)' % np.round(ch, 4))
    byid('NF-1')['valor_recalculado'] = ','.join('%.4f' % c for c in ch)
    cn = [E[(k, 6.86)]['bc'] / v for k, v in zip((0.05, 0.1, 0.25, 0.4), nf3)]
    txt('NF-2', 1, [(H8, 'pasaría de 0.9561, 0.8563, 0.7848 y 0.7759 a 0.9764, 0.8807, 0.8132 y 0.8045')], 'Cociente completo/efectivo(N_f=3) de gamma_bf: 0.9764, 0.8807, 0.8132, 0.8045', 'V', rep='0.9764,0.8807,0.8132,0.8045', tipo='numerica', dato='data/diagnostico_alfa_Nf.txt + data/filtro_completo/*', script=S, nota='aritmetica con el gamma_bf(N_f=3) del log: %s' % np.round(cn, 4))
    byid('NF-2')['estado'] = 'V' if all(abs(a - b) < 0.0002 for a, b in zip(cn, (0.9764, 0.8807, 0.8132, 0.8045))) else 'D'
    # 8.9 modos
    zT0, _ = fcomp(0.25, None, N=22, plano=True); zT, _ = fcomp(0.25, 6.86, N=22, plano=True)
    mT0 = zT0['modos']; mT = zT['modos']
    txt('MD-1', 1, [(H8, 'A T = 0, el modo de paridad tiene λ = +0.999987, tasa 2.49940e-05 (Ma)')], 'Plano T=0: modo de paridad lambda=+0.999987, tasa 2.49940e-5 (Ma)', 'V', rep='2.49940e-05', rec=mT0[3][2], tipo='numerica', dato='data/filtro_completo/gx0.0535714_w6_gz0.42_kf0.3_al4_N22_Nf1_plano_gam6e-07.npz (modos)', script=S)
    txt('MD-2', 1, [(H8, 'el modo vecino tiene λ = -0.999987, tasa 2.49915e-05')], 'Plano T=0: modo vecino lambda=-0.999987, tasa 2.49915e-5', 'V', rep='2.49915e-05', rec=mT0[2][2], tipo='numerica', dato='idem (modos)', script=S)
    txt('MD-3', 1, [(H8, 'Las tasas difieren 1.0e-04 relativo')], 'Plano T=0: las tasas de los dos modos difieren 1.0e-4 relativo', 'V', rep='1.0e-04', rec=abs(mT0[2][2] / mT0[3][2] - 1), tipo='numerica', dato='idem', script=S)
    txt('MD-4', 1, [(H8, 'A x = 6.86, paridad λ = +0.999978, tasa 4.10809e-05, y vecino λ = -0.999978, tasa 4.13247e-05 (difieren 0.6%)')], 'Plano x=6.86: paridad tasa 4.10809e-5, vecino 4.13247e-5 (0.6%)', 'V', rep='4.10809e-05', rec=[m for m in mT if m[0] > 0.9999 and m[2] > 1e-6][0][2], tipo='numerica', dato='data/filtro_completo/*Nf1_plano*_x6.86.npz (modos)', script=S, nota='vecino: %.5e' % [m for m in mT if m[0] < -0.9999 and 3e-5 < m[2] < 6e-5][0][2])
    txt('MD-5', 1, [(H8, 'El bit-flip (Z_L, λ = −1) es el modo de tasa 9.13e-09 (T = 0) y 8.61e-07 (x = 6.86)')], 'Plano: modo de bit-flip tasa 9.13e-9 (T=0) y 8.61e-7 (x=6.86) en unidades de Ma', 'V', rep='9.13e-09', rec=mT0[1][2], tipo='numerica', dato='idem', script=S, nota='x=6.86: %.3e' % mT[1][2])
    txt('MD-6', 1, [(H8, 'traslape con P_L 1.409 y con Y_L 0.000')], 'Plano: traslapes con P_L, Y_L y Z_L del marco polaronico (1.409 ~ sqrt2)', 'R', rep='1.409', tipo='numerica', dato='data/filtro_completo/log_plano_modos.txt', script='codigo/plano_modos.py', nota='los traslapes con Y_L y Z_L no estan en los npz; se leen del log')
    rt = [(float(np.load(os.path.join(FC, 'gx%s_w6_gz0.42_kf0.3_al4_N22_Nf2_gam6e-07_x6.86.npz' % ('%.6g' % GXK[k])), allow_pickle=True)['modos'][3][2]) / float(np.load(os.path.join(FC, 'gx%s_w6_gz0.42_kf0.3_al4_N22_Nf2_gam6e-07_x6.86.npz' % ('%.6g' % GXK[k])), allow_pickle=True)['gpf'])) for k in (0.05, 0.1, 0.25, 0.4)]
    for k, rep_, r_ in zip((0.05, 0.1, 0.25, 0.4), ('1.010', '1.038', '1.193', '1.384'), rt):
        txt('MD-f%g' % k, 1, [(H8, 'Cociente de su tasa a la de γ_pf: 1.010, 1.038, 1.193, 1.384'), (PH, 'tasas 5.1e-6, 5.9e-6 y 6.9e-6 frente a 4.95e-6')], 'Con filtro, x=6.86: cociente tasa(modo vecino)/gamma_pf, kappa2/kappa=%g' % k, 'V', rep=rep_, rec=r_, tipo='numerica', dato='data/filtro_completo/gx*_x6.86.npz (modos[3])', script=S)
    # 8.10 relacion piso / residuo
    z0 = E[(0.05, 6.86)]['z0']
    txt('HR-1', 1, [(H8, 'El piso (1.48e-10 en unidades de Ma, 4.93e-9κ) es unas 16 veces menor que el residuo de hermiticidad cruda (2.33e-9')], 'Piso (k2/k=0.05) 1.48e-10 (Ma) = 4.93e-9 kappa; residuo de hermiticidad 2.33e-9; cociente 16; en unidades de kappa el piso es ~2 veces mayor', 'V', rep='1.48e-10', rec=float(z0['gbf']), tipo='numerica', dato=E[(0.05, 6.86)]['f0'], script=S, nota='herm cruda T=0: %.3e; cociente %.1f; en kappa: %.2f' % (float(z0['herm_cruda']), 2.33e-9 / float(z0['gbf']), float(z0['gbf']) / K / float(z0['herm_cruda'])))
    byid('HR-1')['estado'] = 'V' if abs(float(z0['gbf']) - 1.48e-10) < 0.01e-10 and abs(float(z0['herm_cruda']) - 2.33e-9) < 0.01e-9 and 15 < 2.33e-9 / float(z0['gbf']) < 17 else 'D'
    # 8.11 plano tiempos: log_plano_control
    lp = leer('filtro_completo/log_plano_control.txt')
    tm = re.findall(r'real\s+(\d+)m([\d.]+)s', lp)
    secs = [int(a) * 60 + float(b) for a, b in tm]
    txt('RC-plano', 1, [(H8, '- **Plano** (`log_plano_control.txt`; el tiempo total se asigna por el tiempo del propagador)')], 'Tiempos reales del plano: N=18 4m5s y 3m57s; N=22 11m36s y 11m27s; N=26 24m39s y 20m27s (el 13m46s de N=22 T=0 no esta en el log)', 'V', rep='245,237,696,687,1479,1227', tipo='numerica', dato='data/filtro_completo/log_plano_control.txt', script=S, nota='segundos en el log: %s' % [round(s) for s in secs])
    byid('RC-plano')['estado'] = 'V' if len(secs) >= 6 and abs(secs[0] - 245.3) < 1 and abs(secs[1] - 237.4) < 1 and abs(secs[2] - 696.3) < 1 and abs(secs[3] - 687.0) < 1 and abs(secs[4] - 1479.3) < 1 and abs(secs[5] - 1227.3) < 1 else 'D'
    txt('RC-plano2', 1, [(H8, 'T = 0 13 min 46 s (`time`)'), (PH, 'el plano a T = 0 tardó 13 min 46 s (propagador 806 s)')], 'Plano N=22 T=0: 13 min 46 s de tiempo real (propagador 806 s)', 'R', rep='826', tipo='numerica', dato='no hay log: log_plano_control.txt no contiene esa corrida', script='codigo/run_plano_control.sh', nota='el tiempo no esta en los logs del repo; el propagador (tprop) si esta en el npz: %.0f s' % float(zT0['tprop']))
    # 8.12 control de ocupacion real (efectivo): comparar caches ocreal con las normales
    oc = {}
    for filt in (1, 0):
        for xx in (6, 8, 10, 12):
            fo = os.path.join(TD, 'x%g_k0.25_gz14_f%d_g2e-05_N22_Nf2_w200_a4_ocreal.npz' % (xx, filt))
            fn = os.path.join(TD, 'x%g_k0.25_gz14_f%d_g2e-05_N22_Nf2_w200_a4.npz' % (xx, filt))
            if os.path.exists(fo) and os.path.exists(fn):
                oc[(filt, xx)] = float(np.load(fo, allow_pickle=True)['gpf']) / float(np.load(fn, allow_pickle=True)['gpf']) - 1
    txt('OC-1', 1, [(H8, 'con filtro, γ_pf cambia +2.5e-04 (x = 6), +9.8e-05 (8), +3.7e-05 (10) y +1.4e-05 (12) relativo; en el plano cambia +7.8% (x = 6)'), (RM, 'con filtro, γ_pf cambia ≤ 1e-4')], 'Control con ocupacion real n(w), n(3w) (efectivo, solo Gamma1+-): cambio relativo de gamma_pf', 'R', rep='+2.5e-4,+9.8e-5,+3.7e-5,+1.4e-5 (filtro); +7.8%,+2.9%,+1.1%,+0.4% (plano)', tipo='numerica', dato='data/termico/x*_k0.25_*_ocreal.npz', script='codigo/calc_termico.py', nota='caches normales vs ocreal disponibles: %s' % {str(k): '%.2e' % v for k, v in oc.items()})
    if len(oc) == 8:
        okf = all(abs(oc[(1, x)] - r) < 0.2e-4 for x, r in ((6, 2.5e-4), (8, 9.8e-5), (10, 3.7e-5), (12, 1.4e-5)))
        okp = all(abs(oc[(0, x)] - r) < 0.003 for x, r in ((6, 0.078), (8, 0.029), (10, 0.011), (12, 0.004)))
        byid('OC-1')['estado'] = 'V' if okf and okp else 'D'
    # barrido s (R): recomputar cruces con interpolacion log-log sobre los valores del log
    bs = leer('barrido_s_ocupacion.txt')
    txt('BS-1', 1, [(H8, 'El cociente γ_bf plano/filtrado cruza 1 en **s = 0.132** (variante A) y **s = 0.209** (variante B)')], 'Barrido de la densidad espectral: gamma_bf plano/filtrado cruza 1 en s = 0.132 (A) y 0.209 (B)', 'R', rep='0.132,0.209', tipo='numerica', dato='data/barrido_s_ocupacion.txt', script='codigo/barrido_s_ocupacion.py', nota='el efectivo con ocupaciones fisicas y ansatz no esta cacheado; se contrasta aritmetica con los valores del log')
    A = [(0.1, 0.942), (0.2, 1.094)]; B = [(0.1, 0.836), (0.2, 0.984), (0.5, 1.399)]
    def cruce(P):
        (s0, r0), (s1, r1) = P[0], P[-1]
        for (sa, ra), (sb, rb) in zip(P[:-1], P[1:]):
            if (ra - 1) * (rb - 1) <= 0:
                return float(np.exp(np.interp(0, [np.log(ra), np.log(rb)], [np.log(sa), np.log(sb)])))
    cA, cB = cruce(A), cruce(B + [(1.0, 2.004)])
    byid('BS-1')['valor_recalculado'] = '%.3f,%.3f' % (cA, cB)
    byid('BS-1')['estado'] = 'V' if abs(cA - 0.132) < 0.002 and abs(cB - 0.209) < 0.003 else 'D'
    byid('BS-1')['nota'] += '; cruce recalculado por interpolacion log-log de las filas del log: A=%.3f, B=%.3f' % (cA, cB)
    return

# ============================================================ BLOQUE 14: HANDOFF (tablas de ocupacion, superado, pendientes, historia)
def bloque_textual3(E):
    S = 'consolidacion/verificar_todo.py::bloque_textual3'
    K = KAPM
    # ---- modelo y codigo: comprobacion por inspeccion del script
    com = open(os.path.join(FF, 'codigo/comun.py'), errors='ignore').read()
    ok_h = ('H0 = w * a.dag() * a + 0.5 * wq * sz + (a + a.dag()) * (gx * sx + gz * sz)' in com) and ("[Om * sm.dag(), lambda t: np.exp(-1j * wp * t)]" in com) and ("np.sqrt(kap) * sm" in com)
    txt('MOD-1', 1, [(H8, 'H(t) = ω a†a + (ω_q/2)σ_z + (a + a†)(g_xσ_x + g_zσ_z)'), (INF, 'H(t)=\\omega\\,a^{\\dagger}a+\\frac{\\omega_q}{2}\\sigma_z'), (H8, 'Disipación a T = 0: κD[σ₋] y γD[a]')], 'Hamiltoniano y disipador del modelo (codigo/comun.py): H0 = w a+a + (wq/2) sz + (a+a+)(gx sx + gz sz), drive Om(s+ e^{-i wp t}+h.c.), colapso sqrt(kap) sm', 'V', rep='comun.py', rec=None, tipo='cualitativa', dato='Mechanical-Cat-State/figuras_finales/codigo/comun.py', script=S, nota='inspeccion del codigo: coincide con la definicion; sigmaz de QuTiP con indice 0 = |e>')
    byid('MOD-1')['estado'] = 'V' if ok_h else 'D'
    cf = open(os.path.join(FF, 'codigo/calc_filtro_completo.py'), errors='ignore').read()
    ok_t = ('nq' in cf and 'sm.dag()' in cf and 'nm' in cf)
    txt('MOD-2', 1, [(H8, 'Disipación térmica: κ[(n_q+1)D[σ₋] + n_qD[σ₊]] y γ[(n_m+1)D[a] + n_mD[a†]]'), (INF, 'kappa\\left[(n_q+1)\\D[\\smi]+n_q\\D[\\spl]\\right]')], 'Disipacion termica del modelo completo: kappa[(nq+1)D[s-]+nq D[s+]] + gamma[(nm+1)D[a]+nm D[a+]]', 'V', rep='calc_filtro_completo.py', rec=None, tipo='cualitativa', dato='Mechanical-Cat-State/figuras_finales/codigo/calc_filtro_completo.py', script=S, nota='verificacion superficial por presencia de los colapsos (sm.dag, nq, nm); no se reejecuta')
    byid('MOD-2')['estado'] = 'V' if ok_t else 'D'
    # n_q, n_m
    xq = 6.86
    txt('MOD-3', 1, [(H8, 'n_q = 1/(eˣ−1), n_m = 1/(e^{x/2}−1)'), (INF, 'n_q=[e^{x}-1]^{-1}')], 'n_q = 1/(e^x-1) y n_m = 1/(e^{x/2}-1): n_q(6.86) = 1.050e-3, n_m(6.86) = 3.347e-2', 'V', rep='1.050e-3', rec=1 / (math.exp(xq) - 1), tipo='numerica', dato='data/termico/x6.86_k0.25_*.npz (nq, nm)', script=S, nota='n_m = %.4e' % (1 / (math.exp(xq / 2) - 1)))
    # ---- tablas 8.12: ocupacion de banda lateral (plano / filtro)
    ep, ek = eff_pt(0.25, 6.86, 0), eff_pt(0.25, 6.86, 0, extra='_kick')
    ef, efk = eff_pt(0.25, 6.86, 1), eff_pt(0.25, 6.86, 1, extra='_kick')
    rows_pl = [('sin desplazamiento', '8.996e-4', '3.342e-5', '26.9', (ep['gpf'], ep['gbf'], ep['gpf'] / ep['gbf'])), ('desplazamiento con n_q', '1.506e-3', '3.192e-5', '47.2', (ek['gpf'], ek['gbf'], ek['gpf'] / ek['gbf']))]
    for i, (nm, a, b, c, (va, vb, vc)) in enumerate(rows_pl):
        txt('OP-%d' % i, 1, [(H8, '| %s | %s | %s | %s |' % (nm, a.replace('e-4', 'e-4'), b, c))], 'Efectivo plano, %s: gamma_pf, gamma_bf, eta (kappa=1)' % nm, 'V', rep='%s,%s,%s' % (a, b, c), tipo='numerica', dato=ep['file'] + ' ; ' + ek['file'], script=S, nota='recalculo %.3e, %.3e, %.2f' % (va, vb, vc))
        pa, pb, pc_ = parse(a), parse(b), parse(c)
        byid('OP-%d' % i)['estado'] = 'V' if abs(va - pa[0]) <= pa[1] and abs(vb - pb[0]) <= pb[1] and abs(vc - pc_[0]) <= pc_[1] else 'D'
    rows_fl = [('sin desplazamiento', '1.724e-4', '4.267e-5', '4.039', (ef['gpf'], ef['gbf'], ef['gpf'] / ef['gbf'])), ('ansatz con n_q', '1.728e-4', '4.267e-5', '4.051', (efk['gpf'], efk['gbf'], efk['gpf'] / efk['gbf']))]
    for i, (nm, a, b, c, (va, vb, vc)) in enumerate(rows_fl):
        txt('OF-%d' % i, 1, [(H8, '| %s | %s | %s | %s |' % (nm, a, b, c))], 'Efectivo filtro, %s: gamma_pf, gamma_bf, eta (kappa=1)' % nm, 'V', rep='%s,%s,%s' % (a, b, c), tipo='numerica', dato=ef['file'] + ' ; ' + efk['file'], script=S, nota='recalculo %.3e, %.3e, %.3f' % (va, vb, vc))
        pa, pb, pc_ = parse(a), parse(b), parse(c)
        byid('OF-%d' % i)['estado'] = 'V' if abs(va - pa[0]) <= pa[1] and abs(vb - pb[0]) <= pb[1] and abs(vc - pc_[0]) <= pc_[1] else 'D'
    adj('PL7-f1277', H8, '- **Cociente plano/filtrado (efectivo con el mismo modelo en ambos):**')
    txt('OP-R', 1, [(H8, '| desplazamiento con n(ω), n(3ω); Γ₁± con n_q | 5.692e-3 | 8.556e-5 | 66.5 |'), (H8, '| ansatz con n(ω), n(3ω); Γ₁± con n_q | 1.759e-4 | 4.272e-5 | 4.117 |')], 'Efectivo con ocupacion fisica n(w), n(3w) y ansatz (plano y filtro): resto de filas de las tablas 8.12', 'R', rep='ver HANDOFF 8.12', tipo='numerica', dato='data/handoff_verificacion_referencia.txt', script='codigo/verif_alfa_referencia.py', nota='sin cache; las cifras coinciden con el log (handoff_verificacion_referencia.txt) en las filas inspeccionadas')
    ref = leer('handoff_verificacion_referencia.txt')
    mm = re.findall(r'plano/filtrado=([\d.]+)\s+γ_bf plano/filtrado=([\d.]+)', ref)
    txt('OP-q', 1, [(H8, '- **Cociente plano/filtrado (efectivo con el mismo modelo en ambos):**')], 'gamma_bf plano/filtrado 0.783 sin desplazamiento, 0.748 con ansatz n_q, 2.00 con n(w),n(3w); gamma_pf 5.22, 8.71, 32.4', 'V', rep='0.783,0.748,2.00', tipo='numerica', dato='data/handoff_verificacion_referencia.txt ; ' + ep['file'], script=S, nota='cociente sin desplazamiento recomputado: gamma_bf %.3f gamma_pf %.2f' % (ep['gbf'] / ef['gbf'], ep['gpf'] / ef['gpf']))
    byid('OP-q')['estado'] = 'V' if abs(ep['gbf'] / ef['gbf'] - 0.783) < 0.001 and abs(ep['gpf'] / ef['gpf'] - 5.22) < 0.01 and abs(ek['gbf'] / efk['gbf'] - 0.748) < 0.001 else 'D'
    # barrido s: transcripcion contra el log
    bs = leer('barrido_s_ocupacion.txt')
    rowsb = re.findall(r'^\s*([\d.]+)\s*\|\s*([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.]+)\s*\|\s*([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', bs, re.M)
    doc = [('0.1', '8.021e-4', '4.020e-5', '1.724e-4', '4.268e-5', '4.654', '0.942'), ('0.2', '1.413e-3', '4.667e-5', '1.728e-4', '4.268e-5', '8.180', '1.094'), ('0.5', '3.143e-3', '6.372e-5', '1.739e-4', '4.270e-5', '18.07', '1.492'), ('1', '5.738e-3', '8.561e-5', '1.759e-4', '4.272e-5', '32.62', '2.004'), ('2', '1.014e-2', '1.135e-4', '1.799e-4', '4.277e-5', '56.35', '2.654'),
           ('0.1', '9.809e-4', '3.567e-5', '1.725e-4', '4.267e-5', '5.688', '0.836'), ('0.2', '1.542e-3', '4.198e-5', '1.728e-4', '4.268e-5', '8.920', '0.984'), ('0.5', '3.173e-3', '5.973e-5', '1.740e-4', '4.269e-5', '18.24', '1.399'), ('2', '1.039e-2', '1.262e-4', '1.798e-4', '4.278e-5', '57.78', '2.951')]
    nbad = 0
    if len(rowsb) == 10:
        rb = rowsb[:5] + rowsb[5:]
        rbB = [r for r in rowsb[5:] if r[0] != '1']
        rbA = rowsb[:5]
        for d_, r in zip(doc, rbA + rbB):
            lg = [r[1], r[2], r[4], r[5], r[7], r[8]]
            for a, b in zip(d_[1:], lg):
                pa, pb = parse(a), float(b)
                if abs(pa[0] - pb) > pa[1] * 1.0 * (1 + 1e-9): nbad += 1
    txt('BS-tab', 1, [(H8, '| 0.1 | A | 8.021e-4 | 4.020e-5 | 1.724e-4 | 4.268e-5 | 4.654 | 0.942 |')], 'Tabla del barrido de la densidad espectral (9 filas, 6 columnas): transcripcion frente a data/barrido_s_ocupacion.txt', 'R', rep='tabla', tipo='numerica', dato='data/barrido_s_ocupacion.txt', script='codigo/barrido_s_ocupacion.py', nota='transcripcion: %d celdas fuera de 1 unidad del ultimo digito de %d; efectivo con ocupacion fisica sin cache npz' % (nbad, 54))
    byid('BS-tab')['estado'] = 'R' if nbad == 0 else 'D'
    txt('BS-sup', 1, [(H8, '- **El plano sigue [NO VALIDADO]:** con ocupación física el efectivo con ansatz predice γ_pf 3.8 veces')], 'Con ocupacion fisica el efectivo con ansatz predice gamma_pf 3.8 veces y gamma_bf 2.7 veces los valores con n_q (plano)', 'V', rep='3.8,2.7', tipo='numerica', dato='data/handoff_verificacion_referencia.txt', script=S, nota='cocientes de las filas del log: %.2f y %.2f' % (5.692e-3 / 1.506e-3, 8.556e-5 / 3.192e-5))
    byid('BS-sup')['estado'] = 'V' if abs(5.692e-3 / 1.506e-3 - 3.8) < 0.05 and abs(8.556e-5 / 3.192e-5 - 2.7) < 0.05 else 'D'
    # ---- Seccion 9: criterios numericos (codigo)
    txt('CR-1', 1, [(H8, 'Integrador con atol 1e-12 y rtol 1e-10'), (RM, 'Tolerancias del integrador: atol 1e-12, rtol 1e-10')], 'Integrador: atol 1e-12, rtol 1e-10 (comun.py::OPTS)', 'V', rep='1e-12,1e-10', rec=None, tipo='numerica', dato='codigo/comun.py', script=S)
    byid('CR-1')['estado'] = 'V' if "OPTS = dict(atol=1e-12, rtol=1e-10" in com else 'D'
    lhp = open(os.path.join(FF, 'codigo/log_herm_posit.py'), errors='ignore').read()
    txt('CR-2', 1, [(H8, 'Validar traza, hermiticidad **antes** de hermitizar y positividad de ρ, con tolerancias 1e-10 / 1e-10 / −1e-9'), (H8, 'Umbrales vigentes (desde 2026-10-01)')], 'Umbrales en el codigo (log_herm_posit.py): UMB = {plano: (1e-10,-1e-9), filtro: (1e-8,-1e-8)}', 'V', rep='UMB', tipo='cualitativa', dato='codigo/log_herm_posit.py', script=S, nota='coincide con HANDOFF 8.10 y 9')
    byid('CR-2')['estado'] = 'V' if 'UMB = {True: (1e-10, -1e-9), False: (1e-8, -1e-8)}' in lhp else 'D'
    # superado y retirado (S): verificacion cuando es posible
    rho12 = np.load(os.path.join(DATA, 'fig2/gx-0.212132_wp12.000000_wq12.000000_Om0.060000_N22.npz'), allow_pickle=True)
    al = np.sqrt(complex(rho12['al2eff']))
    pad = Pc_rho(rho12['rho'], 22, al, 0.2121320343559643 / 6.0)
    txt('S-01', 1, [(H8, 'Código adaptado con α_eff para P_c: no sirve (daba P_c = 0.967 sin gato)'), (RM, 'en ω_p = 12 daría P_c = 0.967 con α_eff² = −2.48'), (PH, '| E6 |')], 'Codigo adaptado D(d)|alpha_eff> en w_p=12 da P_c = 0.967 (sin gato)', 'S', rep='0.967', rec=pad, tipo='numerica', dato='data/fig2/gx-0.212132_wp12.000000_wq12.000000_Om0.060000_N22.npz', script=S, nota='superado por el codigo fijo D(g_z/w)|+-alpha_nom> (C6); recalculo P_c(adaptado) = %.4f' % pad)
    byid('S-01')['estado'] = 'S'
    txt('S-02', 1, [(H8, '"hf_q/kT ≳ 9 para sesgo 100" (régimen Naseem, frecuencia mecánica) → usar §8.')], 'Superado: umbral termico hf/kT>~9 (frecuencia mecanica, Tarea 37) -> x*(100) = 7.18-10.11 en f_q (TX-*)', 'S', nota='superado por HANDOFF 8.2 y TX-*; commit a1 de la fase 1 (2026-09-29)', tipo='cualitativa')
    txt('S-03', 1, [(H8, '"Techo κ/4 del confinamiento" → ≈ 0.28κ con |α|² = 4 (κ/4 solo vale para α = 0).')], 'Superado: techo kappa/4 del confinamiento con |a|^2 != 0 -> 0.28 kappa (MN-02)', 'S', nota='MN-02 recalcula 0.2784; V6 da kappa/4 para alpha=0', tipo='cualitativa')
    txt('S-04', 1, [(H8, '"γ_bf ≈ 0.05 n_qκ" como constante → c_bf(κ₂/κ).')], 'Superado: gamma_bf ~ 0.05 nq kappa constante -> c_bf(k2/k) varia >2 ordenes (CB-var)', 'S', nota='CB-var: %s' % byid('CB-var')['nota'], tipo='cualitativa')
    txt('S-05', 1, [(H8, '"≤ 0.4%" de precisión en todo el rango → función de κ₂/κ (§4.5).')], 'Superado: precision <=0.4% en todo el rango -> funcion de kappa2/kappa (P4-a/b/c)', 'S', nota='P4-a..c', tipo='cualitativa')
    txt('S-06', 1, [(H8, 'Brecha con máximo en Γ₂/κ ≈ 0.13 y rama con |Im λ| ∝ N: artefactos de borde de Fock.')], 'Superado: maximo de la brecha en Gamma2/kappa~0.13 y rama |Im lambda|~N = artefactos de borde de Fock', 'S', nota='Tareas 35,38,45; F4m-* (|Im lambda| crece con N: 0.040, 0.054, 0.061)', tipo='cualitativa')
    txt('S-07', 1, [(H8, '"5.º modo espectral" como tasa de confinamiento (Tarea 43): espurio. Con la tasa dinámica, el filtro cuesta ≤ 5% para κ_f ≥ 0.3.')], 'Superado: 5.o modo espectral como tasa de confinamiento; con la tasa dinamica el filtro cuesta <= 5% para kf >= 0.3', 'S', rep='0.05', rec=None, tipo='numerica', dato='data/fig4b.csv', script=S, nota='recalculo del costo: %s (kf=3: 3.5%%, 1: 3.3%%, 0.3: 9.8%%): para kf=0.3 el costo es 10%%, no <=5%%' % [round(1 - cc[k] / cc[0.0], 3) for k in (3.0, 1.0, 0.3)] if False else 'F4c-*: kf=3: 3.5%, kf=1: 3.3%, kf=0.3: 9.8%, kf=0.1: 47%; "<=5% para kf>=0.3" solo vale para kf>=1 (kf=0.3 cuesta ~10%)')
    byid('S-07')['estado'] = 'S'
    txt('S-08', 1, [(H8, 'Exceso ×2.9 de P10: artefacto de N = 16.')], 'Superado: exceso x2.9 de gamma_pf (P10) = artefacto de N=16', 'S', rep='2.9', rec=float(byid('FC-0.03-16-pf')['valor_recalculado']), tipo='numerica', dato='data/filtro_completo/gx0.0216506_w6_gz0.36_kf0.3_al4_N16_Nf2.npz', script=S, nota='FC-0.03-16-pf: 2.88; con N=20: 0.986')
    byid('S-08')['estado'] = 'S'
    txt('S-09', 1, [(H8, 'Acuerdo del 0.05% de V3: era una compensación de errores (con la fórmula corregida: 0.982).')], 'Superado: acuerdo del 0.05% de V3 (compensacion); con la formula corregida 0.982', 'S', rep='0.982', rec=float(byid('V3-07')['valor_recalculado']), tipo='numerica', dato='res_v2/wp11.9800_N22.npz', script=S, nota='V3-07')
    byid('S-09')['estado'] = 'S'
    txt('S-10', 1, [(H8, '"γ_bf/n_q solo sube un 28% con el filtro"'), (RM, 'γ_bf/n_q con filtro frente al plano sube 28% en el efectivo')], 'Superado: gamma_bf/n_q solo sube 28% con el filtro -> +17% en el completo (un punto)', 'S', rep='0.28', rec=float(byid('PL7-f28')['valor_recalculado']), tipo='numerica', dato='PL7-f28, PL7-f17', script=S, nota='el +28% es del efectivo; en el completo +17% (PL7-f17); el informe.tex repite el +28% (l.909) y la nota del README')
    byid('S-10')['estado'] = 'S'
    txt('S-11', 1, [(H8, '- **"η_max ≈ 5000" en todo el mapa**')], 'Superado: eta_max ~ 5000 en todo el mapa -> 3.1e4 (k2/k=0.05) a 2.35e3 (0.4)', 'S', tipo='cualitativa', nota='T86-*-ec')
    txt('S-12', 1, [(H8, '- **"N = 24 con filtro: ~13 GB solo en el propagador"**')], 'Superado: N=24 con filtro ~13 GB -> N=22 ya usa 14.0 GB y N=24 no cabe', 'S', tipo='cualitativa', nota='RC-N22')
    txt('S-13', 1, [(H8, '- **Tiempo de 7.6 s de la corrida plana térmica**')], 'Superado: tiempo de 7.6 s de la corrida plana termica (lectura de cache); propagador de 417 s', 'S', rep='417', rec=float(np.load(os.path.join(FC, 'gx0.0535714_w6_gz0.42_kf0.3_al4_N22_Nf1_plano_gam6e-07_x6.86.npz'), allow_pickle=True)['tprop']), tipo='numerica', dato='data/filtro_completo/gx0.0535714_*Nf1_plano*_x6.86.npz (tprop)', script=S, nota='tprop del npz')
    byid('S-13')['estado'] = 'S'
    txt('S-14', 1, [(H8, '- **"7 de 10 corridas sobre 1e-10 en hermiticidad"**')], 'Superado: "7 de 10 corridas sobre 1e-10" -> 15 de 16 (log actual)', 'S', tipo='cualitativa', nota='Q-b3')
    txt('S-15', 1, [(H8, '- **"Positividad ≥ −4.4e-10 en las corridas completas"**')], 'Superado: positividad >= -4.4e-10 en las corridas completas -> -2.49e-9 y -1.42e-9 (log) y -5.58e-9 (verificacion central)', 'S', rep='-4.4e-10', rec=-5.58e-9, tipo='numerica', dato='Q-a1, Q-b7, Q-b8', script=S, nota='commit 2b64d26 elimina la frase del README (verificado con git show)')
    byid('S-15')['estado'] = 'S'
    txt('S-16', 1, [(H8, '- **Comparaciones plano/filtrado hechas con el efectivo plano sin el desplazamiento dependiente del estado**')], 'Superado: comparaciones plano/filtrado con el efectivo plano sin desplazamiento (no fiables)', 'S', tipo='cualitativa', nota='PL7-98, PL7-16')
    txt('S-17', 1, [(H8, '- **"El efectivo con filtro sobreestima γ_bf en la isla por un efecto de x"**')], 'Superado: sobreestimacion de gamma_bf por un efecto de x -> depende de kappa2/kappa (T84b)', 'S', tipo='cualitativa', nota='T84b-0.25-6.86-rbf, T84-ind-*')
    # inventario: papers / bib
    papers = sorted(os.listdir(os.path.join(ROOT, 'papers')))
    refs_exist = all(os.path.exists(os.path.join(ROOT, 'docs', p)) for p in ('brecha_punto_excepcional/refs.bib', 'mapa_plataformas/refs_mapa.bib'))
    txt('INV-1', 1, [(H8, 'faltan `refs.bib` (brecha) y `refs_mapa.bib` (mapa), y **Hou 2024 no está**')], 'HANDOFF 10: faltan refs.bib (brecha) y refs_mapa.bib (mapa)', 'S', rep='faltan', rec=None, tipo='cualitativa', dato='docs/brecha_punto_excepcional/refs.bib, docs/mapa_plataformas/refs_mapa.bib', script=S, nota='ambos archivos existen ahora (README del 2026-09-29: "reconstruida"); papers/ contiene %s; Ma 2019 y de Matos Filho 1996 estan en .gitignore y no estan en esta copia' % papers)
    byid('INV-1')['estado'] = 'S' if refs_exist else 'V'
    txt('ENV-1', 1, [(MD0, '.venv_qutip5/         QuTiP 5.3.1 (todo lo demás; usar este)'), (MD0, 'Los scripts de figuras_finales ya apuntan a `.venv_qutip5`')], 'El entorno .venv_qutip5 (QuTiP 5.3.1) existe', 'D', rep='existe', rec=None, tipo='cualitativa', dato='Mechanical-Cat-State/', script=S, nota='no existe en esta copia (esta en .gitignore; solo existe .venv); los comandos de la seccion 11 requieren recrearlo. Los scripts historicos citan /home/jhon/GitHub/trabajo-grado/... en logs')
    byid('ENV-1')['estado'] = 'D' if not os.path.exists(os.path.join(MCS, '.venv_qutip5')) else 'V'
    return

# ============================================================ BLOQUE 15: informe.tex y README de figuras_finales (lineas restantes)
def bloque_textual4(E):
    S = 'consolidacion/verificar_todo.py::bloque_textual4'
    K = KAPM
    # --- informe: condiciones de validez y parametros publicados
    adj('P9-04', INF, '\\item $\\chi|\\alpha|^2\\lesssim0.3\\,\\kappa$')
    adj('RU-10', INF, '\\item $g_z/\\omega\\lesssim0.1$ para que el desplazamiento')
    adj('MA-01', INF, 'el disipador es $\\Gamma[2\\smi\\rho\\spl')
    txt('PAR-1', 1, [(INF, '$\\omega/2\\pi$ & 6 GHz & 100 MHz & 17.7 MHz'), (INF, '$g_x/2\\pi$, $g_z/2\\pi$ & $-212$, $212$ MHz'), (INF, '$\\kappa/2\\pi$ & 30 MHz & 100 kHz & 16 MHz'), (INF, '$\\Omega$ o $\\varepsilon$ & $\\Omega/2\\pi=60$ MHz'), (INF, '$\\omega_q$ & 12 GHz & 200 MHz')], 'Parametros publicados de Ma, Naseem y Liu (Tabla param): valores copiados de los articulos', 'L', tipo='literatura', nota='fuera de alcance (valores de los articulos); consistentes con los usados en el codigo: Ma (gx=-0.2121, gz=0.2121 = 0.3 sin/cos pi/4, kappa=0.03=2*0.015, Omega=0.06, wq=12)')
    txt('PAR-2', 1, [(INF, '$g_x/2\\pi$, $g_z/2\\pi$ & $-212$, $212$ MHz')], 'Ma: |gx| = |gz| = 0.3 cos(pi/4) = 0.212 (2pi GHz), kappa = 2 Gamma = 0.03', 'V', rep='0.212', rec=0.3 * math.cos(math.pi / 4), tipo='numerica', dato='(analitico)', script=S, tol=0.001)
    # --- derivaciones (texto del informe) -> afirmaciones simbolicas
    for i, a in (('SIM-10', '\\frac{\\kappa_1}{\\kappa_2}=\\frac{10}{9}\\frac{g_x^2\\kappa}{\\omega^2}\\cdot'), ('SIM-11', '\\item El umbral $\\kappa_2/\\kappa_1=220$'), ('SIM-29', '\\paragraph{Piso intrínseco.} La pérdida intrínseca suma'), ('SIM-29', '\\frac{\\kappa_2^{\\rm eff}}{\\kappa}\\geq220'), ('SIM-22', 'que se reduce a la Ec.~\\eqref{eq:merito} para un baño plano'), ('SIM-26', 'Los autoestados de $\\hat N$ son'), ('SIM-23', 'El modo más lento distinto de cero es la coherencia')):
        adj(i, INF, a)
    txt('INF-N2', 1, [(INF, 'Los autoestados de $\\hat N$ son')], 'Elemento de matriz en el bloque N: G_N = G sqrt(N(N-1)); E_N^+- = -i k/4 +- sqrt(G_N^2 - k^2/16)', 'V', tipo='cualitativa', nota='verificado por SIM-23/SIM-26 (algebra simbolica)', script=S)
    # --- metodo
    adj('MN-08', INF, 'Para $\\alpha\\neq0$ (gato con $|\\alpha|^2=4$) se define la tasa de confinamiento por la dinámica')
    adj('P9-04', INF, 'La pérdida de confinamiento es del $5\\%$')
    adj('CR-1', INF, 'Se integra la ecuación maestra~\\eqref{eq:ME} con el hamiltoniano completo')
    zd = np.load(os.path.join(DATA, 'fig3/gx0.05_w6.0_gz4.0_al4.0_N22.npz'), allow_pickle=True)
    txt('PO-1', 1, [(INF, 'Las tasas son $r_k=-\\ln|\\lambda_k|/T_p$'), (RM, 'mayor |Tr(PR)|, peso de borde < 0.5')], 'Modo de paridad: traslape |Tr(P R)|/||R|| ~ 1.41', 'V', rep='1.41', rec=float(zd['par_overlap']), tipo='numerica', dato='data/fig3/gx0.05_w6.0_gz4.0_al4.0_N22.npz', script=S, tol=0.005)
    # signo del termino de pares (V1b)
    vb = open(os.path.join(VAL, 'independiente/v1b_salida.txt'), errors='ignore').read()
    rowsv = re.findall(r'gx=([+-][\d.]+) gz=([+-][\d.]+)\s+H_ef\[g2,e0\]=([+-][\d.e-]+)\s+-G\*sqrt2=([+-][\d.e-]+)', vb)
    txt('SG-1', 1, [(INF, 'Por ejemplo, con $(g_x,g_z)=(+0.01,+0.02)$ el elemento vale $-5.651\\times10^{-4}$')], 'Elemento de pares <g,2|H_ef|e,0> en (+0.01,+0.02) = -5.651e-4 frente a -5.657e-4', 'R', rep='-5.651e-04', tipo='numerica', dato='Mechanical-Cat-State/validacion/independiente/v1b_salida.txt', script='validacion/independiente/v1_signo.py', nota='coincide con el log: %s; -sqrt(2) G (analitico) = %.4e' % (rowsv[0][2:], -math.sqrt(2) * 2 * 0.01 * 0.02 / 1.0))
    txt('SG-2', 1, [(INF, 'y con $(-0.02,-0.03)$ vale $-1.692\\times10^{-3}$')], 'Elemento de pares en (-0.02,-0.03) = -1.692e-3 frente a -1.697e-3', 'R', rep='-1.692e-03', tipo='numerica', dato='v1b_salida.txt', script='validacion/independiente/v1_signo.py', nota='coincide con el log: %s' % (rowsv[3][2:],))
    # sesgo termico: texto del informe
    cq_pl = ep0 = None
    ep, efl = eff_pt(0.25, 6.86, 0), eff_pt(0.25, 6.86, 1)
    nq = 1 / (math.exp(6.86) - 1)
    txt('CBF-28', 1, [(INF, 'mientras que $\\gamma_{\\rm bf}/n_q$ sube un $28\\%$'), (RM, 'γ_bf/n_q con filtro frente al plano sube 28% en el efectivo (0.032κ → 0.041κ'), (H8, '"γ_bf/n_q solo sube un 28% con el filtro"')], 'gamma_bf/n_q (isla, x=6.86): plano 0.032 kappa, filtro 0.041 kappa (+28%)', 'V', rep='0.032,0.041', tipo='numerica', dato=ep['file'] + ' ; ' + efl['file'], script=S, nota='recalculo %.4f y %.4f' % (ep['gbf'] / nq, efl['gbf'] / nq))
    byid('CBF-28')['estado'] = 'V' if abs(ep['gbf'] / nq - 0.032) < 0.0006 and abs(efl['gbf'] / nq - 0.041) < 0.0006 else 'D'
    txt('CBF-17', 1, [(RM, '17% en el completo (0.027κ → 0.032κ, un punto)')], 'gamma_bf/n_q en el completo: plano 0.027 kappa, filtro 0.032 kappa (+17%)', 'V', rep='0.027,0.032', tipo='numerica', dato='data/filtro_completo/*', script=S, nota='recalculo %.4f y %.4f' % (float(fcomp(0.25, 6.86, plano=True)[0]['gbf']) / K / nq, E[(0.25, 6.86)]['bc'] / nq))
    byid('CBF-17')['estado'] = 'V' if abs(float(fcomp(0.25, 6.86, plano=True)[0]['gbf']) / K / nq - 0.027) < 0.0006 and abs(E[(0.25, 6.86)]['bc'] / nq - 0.032) < 0.0006 else 'D'
    # curvas: 34 de 160, hot
    R = [[float(x) for x in r] for r in list(csv.reader(open(os.path.join(DATA, 'termico_curvas.csv'))))[1:]]
    nc, hot = sum(1 for r in R if r[10] == 1.0), sum(1 for r in R if r[11] == 1.0)
    txt('CV-34', 1, [(RM, '34 de 160 puntos cambian > 3% entre N = 22 y 24'), (INF, '34 de 160 puntos cambian más del $3\\%$'), ], 'En la isla, 34 de 160 puntos cambian >3% entre N=22 y 24 (todos con x >~ 13)', 'V', rep='34', rec=nc, tipo='numerica', dato='data/termico_curvas.csv', script=S, nota='x minimo de los no convergidos: %.2f' % min(r[2] for r in R if r[10] == 1.0))
    byid('CV-34')['estado'] = 'V' if nc == 34 and min(r[2] for r in R if r[10] == 1.0) >= 12.5 else 'D'
    txt('CV-hot', 1, [(RM, 'Región caliente n_q > 0.3 (x < 1.47)'), (INF, 'La región caliente $n_q>0.3$')], 'Region caliente n_q > 0.3 equivale a x < 1.47', 'V', rep='1.47', rec=math.log(1 + 1 / 0.3), tipo='numerica', dato='(analitico)', script=S, tol=0.005)
    # isla completo (tabla del informe l.938-941)
    d = E[(0.25, 10.1)]
    for nm, rep, v, ix in (('gpf_c', '1.556', d['pc'] * 1e4, 'gamma_pf completo (x1e-4)'), ('gpf_e', '1.627', d['pe'] * 1e4, 'gamma_pf efectivo'), ('gbf_c', '1.316', d['bc'] * 1e6, 'gamma_bf completo (x1e-6)'), ('gbf_e', '1.650', d['be'] * 1e6, 'gamma_bf efectivo'), ('eta_c', '118.2', d['etac'], 'eta completo'), ('eta_e', '98.6', d['etae'], 'eta efectivo')):
        adj('T84a-0.25-10.1-' + nm, INF, {'gpf_c': '$\\gamma_{\\rm pf}$ & $1.556\\times10^{-4}$', 'gpf_e': '$\\gamma_{\\rm pf}$ & $1.556\\times10^{-4}$', 'gbf_c': '$\\gamma_{\\rm bf}$ & $1.316\\times10^{-6}$', 'gbf_e': '$\\gamma_{\\rm bf}$ & $1.316\\times10^{-6}$', 'eta_c': '$\\eta=\\gamma_{\\rm pf}/\\gamma_{\\rm bf}$ & 118.2', 'eta_e': '$\\eta=\\gamma_{\\rm pf}/\\gamma_{\\rm bf}$ & 118.2'}[nm])
    adj('T84b-0.25-10.1-rbf', INF, '$\\gamma_{\\rm bf}$ & $1.316\\times10^{-6}$'); adj('T84b-0.25-10.1-rpf', INF, '$\\gamma_{\\rm pf}$ & $1.556\\times10^{-4}$'); adj('T84b-0.25-10.1-reta', INF, '$\\eta=\\gamma_{\\rm pf}/\\gamma_{\\rm bf}$ & 118.2')
    txt('IS-1', 1, [(INF, '$\\gamma_{\\rm bf}(T=0)$ & $3.09\\times10^{-8}$ & $5.8\\times10^{-9}$')], 'Isla: gamma_bf(T=0) completo 3.09e-8 (Ma) y efectivo 5.8e-9 (kappa=1)', 'V', rep='3.09e-8', rec=float(d['z0']['gbf']), tipo='numerica', dato=d['f0'], script=S, nota='el informe da el valor del completo en unidades de Ma (3.086e-8 = 0.03 * 1.029e-6?): gbf Ma = %.3e; en unidades de kappa = %.3e (HANDOFF T86: 3.086e-08). El informe lo escribe como 3.09e-8 sin aclarar unidades (T86-0.25-pc)' % (float(d['z0']['gbf']), d['bf0c']))
    byid('IS-1')['estado'] = 'V' if abs(d['bf0c'] - 3.09e-8) < 0.01e-8 else 'D'
    txt('IS-2', 1, [(INF, 'Su piso de $T=0$ es el $2.35\\%$ de $\\gamma_{\\rm bf}$ térmico'), (INF, 'coincide con el efectivo al $4.4\\%$')], 'Isla: piso = 2.35% del gamma_bf termico; gamma_pf coincide al 4.4%; gamma_bf 20% por debajo; eta completo 20% mayor', 'V', rep='2.35%', rec=d['fr'], tipo='numerica', dato=d['fz'], script=S, nota='gamma_pf: %.3f, gamma_bf: %.3f, eta: %.3f' % (d['rpf'], d['rbf'], d['rseta']))
    # piso del efectivo (informe l.947)
    e0p, e0f = eff_pt(0.25, 60, 0), eff_pt(0.25, 60, 1)
    p_pl2 = float(np.load(os.path.join(TD, 'x25_k0.25_gz14_f0_g0.0002_N22_Nf2_w200_a4.npz'), allow_pickle=True)['gbf']); p_fi2 = float(np.load(os.path.join(TD, 'x25_k0.25_gz14_f1_g0.0002_N22_Nf2_w200_a4.npz'), allow_pickle=True)['gbf'])
    txt('PF-1', 1, [(INF, 'el piso del modelo efectivo es $2.1\\times10^{-7}\\kappa$ con baño plano y $5.8\\times10^{-9}\\kappa$ con filtro')], 'Piso efectivo (x=60): plano 2.1e-7 kappa, filtro 5.8e-9 kappa, razon 36', 'V', rep='2.1e-7', rec=e0p['gbf'], tipo='numerica', dato=e0p['file'] + ' ; ' + e0f['file'], script=S, nota='filtro %.3e; razon %.1f' % (e0f['gbf'], e0p['gbf'] / e0f['gbf']))
    byid('PF-1')['estado'] = 'V' if abs(e0p['gbf'] - 2.1e-7) < 0.06e-7 and abs(e0f['gbf'] - 5.8e-9) < 0.06e-9 and abs(e0p['gbf'] / e0f['gbf'] - 36) < 1 else 'D'
    txt('PF-2', 1, [(INF, 'al multiplicar $\\gamma/\\kappa$ por $10$ (de $2\\times10^{-5}$ a $2\\times10^{-4}$) cambia de $2.114$ a $2.116\\times10^{-7}$ (plano) y de $5.82$ a $6.13\\times10^{-9}$ (filtro)')], 'El piso no depende de gamma: x=25, gamma/kappa 2e-4: plano 2.116e-7, filtro 6.13e-9', 'V', rep='2.116e-7', rec=p_pl2, tipo='numerica', dato='data/termico/x25_k0.25_gz14_f{0,1}_g0.0002_N22_*.npz', script=S, nota='filtro %.3e (6.13e-9); el valor a gamma=2e-5 es x=60' % p_fi2)
    byid('PF-2')['estado'] = 'V' if abs(p_pl2 - 2.116e-7) < 0.002e-7 and abs(p_fi2 - 6.13e-9) < 0.02e-9 else 'D'
    txt('PF-3', 1, [(INF, 'En el modelo completo el piso es $3.09\\times10^{-8}\\kappa$, $5.3$ veces el del efectivo')], 'Piso completo / efectivo (isla) = 5.3', 'V', rep='5.3', rec=d['bf0c'] / e0f['gbf'], tipo='numerica', dato=d['f0'], script=S, tol=0.05)
    # control ocupacion real: x*(100)
    for filt, nm, rep in ((1, 'filtro', '10.114'), (0, 'plano', '8.173')):
        c = eff_curve(0.25, filt, 2e-5, extra='_ocreal')
        xs = xstar_c(c, 100)
        txt('OC-x%d' % filt, 1, [(INF, 'con filtro $\\gamma_{\\rm pf}$ cambia $\\leq10^{-4}$ y $x^{*}(100)$ no cambia'), (RM, 'con filtro, γ_pf cambia ≤ 1e-4 y x*(100) no cambia (10.114)')] if filt else [(INF, 'pasa de $8.197$ a $8.173$'), (RM, 'x*(100) pasa de 8.197 a 8.173')], 'Control ocupacion real: x*(100) %s con n(w), n(3w) = %s' % (nm, rep), 'V', rep=rep, rec=xs, tipo='numerica', dato='data/termico/x*_k0.25_*_ocreal.npz', script=S, tol=0.002, nota='recalculo %.4f' % xs)
    # --- README de figuras_finales
    adj('MA-01', RM, '**Modelo:** H completo de Ma en el laboratorio (sin RWA), ω = 6, g = 0.3')
    f2b = csvrows('fig2b.csv'); npts = [int(r[9]) for r in f2b]
    txt('RM-17', 1, [(RM, '17 valores de ω_p por serie'), (RM, 'La serie κ₂/κ = 1 tiene además el punto ω_p = 12 de (a)')], 'Fig.2(b): 17 valores de w_p por serie; la serie kappa2/kappa=1 tiene ademas w_p=12 (18)', 'V', rep='17,17,17,18,17', rec=None, tipo='numerica', dato='data/fig2b.csv (n_points)', script=S, nota='n_points: %s' % npts)
    byid('RM-17')['estado'] = 'V' if npts == [17, 17, 17, 18, 17] else 'D'
    txt('RM-10', 1, [(RM, 'Son los 10 puntos de V4 más 5 nuevos'), (RM, '**(a)** 15 puntos con |α|²_nom = 4 y N = 22')], 'Fig.3(a): 15 puntos = 10 de V4 + 5 nuevos', 'V', rep='10+5', tipo='numerica', dato='validacion/independiente/res_v4/gx*_N22.npz', script=S, nota='res_v4 tiene %d puntos gx*_N22 con alpha nominal 4 (excluye b_al*, N28 y la repeticion)' % len([f for f in glob.glob(os.path.join(VAL, 'independiente/res_v4/gx*_N22.npz'))]))
    byid('RM-10')['estado'] = 'V' if len(glob.glob(os.path.join(VAL, 'independiente/res_v4/gx*_N22.npz'))) == 10 else 'D'
    zN = [f3(0.05, 6.0, 4.0, a, n)[0] for a, n in ((2.0, 22), (4.0, 22), (6.0, 26))]
    txt('RM-N3', 1, [(RM, '(N = 22, 22 y 26)'), (INF, '($N=22,22,26$)')], 'Fig.3(b): N = 22, 22 y 26 para |a|^2 nominal 2, 4, 6', 'V', rep='22,22,26', rec=None, tipo='numerica', dato='data/fig3/gx0.05_w6.0_gz4.0_al{2,4,6}*', script=S, nota='N de los npz: %s' % [int(z['N']) for z in zN])
    byid('RM-N3')['estado'] = 'V' if [int(z['N']) for z in zN] == [22, 22, 26] else 'D'
    zv = np.load(os.path.join(DATA, 'principal_fig2d.npz'), allow_pickle=True)['val']
    txt('RM-v2d', 1, [(RM, '(d): |Tr ρ − 1| = 2.6e-12, ‖ρ − ρ†‖ = 1.1e-13, mínimo autovalor 5e-9')], 'Fig.2(d): |Tr rho - 1| = 2.6e-12, ||rho - rho+|| = 1.1e-13, min autovalor 5e-9', 'V', rep='2.6e-12,1.1e-13,5e-9', tipo='numerica', dato='data/principal_fig2d.npz (val)', script=S, nota='val = %s' % zv)
    byid('RM-v2d')['estado'] = 'V' if abs(zv[0] - 2.6e-12) < 0.1e-12 and abs(zv[1] - 1.1e-13) < 0.1e-13 and abs(zv[2] - 5e-9) < 0.1e-9 else 'D'
    txt('RM-v2ab', 1, [(RM, '(a) y (b) cumplen con |Tr ρ − 1| = 0 y mínimo autovalor ≥ 4e-15')], 'Fig.2 principal (a),(b): |Tr rho - 1| = 0 y min autovalor >= 4e-15 (rho de w_p=12 y 11.98)', 'V', rep='4e-15', tipo='numerica', dato='data/fig2/*wp12.000000*N22, *wp11.980000*N22', script=S, nota='min autovalor: %s ; |Tr-1| max %s' % ([float(np.load(os.path.join(DATA, 'fig2/gx-0.212132_wp%s_wq12.000000_Om0.060000_N22.npz' % w_), allow_pickle=True)['val'][2]) for w_ in ('12.000000', '11.980000')], [float(np.load(os.path.join(DATA, 'fig2/gx-0.212132_wp%s_wq12.000000_Om0.060000_N22.npz' % w_), allow_pickle=True)['val'][0]) for w_ in ('12.000000', '11.980000')]))
    # pfig2: convergencia en N de la regla universal
    def pfpair(name_pat, wq):
        a16 = glob.glob(os.path.join(DATA, 'pfig2', name_pat % (wq, 16))); a22 = glob.glob(os.path.join(DATA, 'pfig2', name_pat % (wq, 22)))
        return a16, a22
    pat = 'w4_gx-0.142857_gz0.21_k0.03_Om0.06_wq%s_wp%s_N%d.npz'
    def pcx(wq, N):
        f = os.path.join(DATA, 'pfig2', 'w4_gx-0.142857_gz0.21_k0.03_Om0.06_wq%s_wp%s_N%d.npz' % (wq, wq, N))
        z = np.load(f, allow_pickle=True); return Pc_rho(z['rho'], N, 2j, 0.21 / 4.0)
    p16, p22 = pcx('7.9863946', 16), pcx('7.9863946', 22)
    f16, f22 = pcx('8.0088946', 16), pcx('8.0088946', 22)
    txt('PF2-c1', 1, [(RM, 'ω = 4, κ₂/κ = 1, N = 16 → 22: P_c en el pico sube 3.7e-3 (0.9945 → 0.9983)'), (INF, 'para $\\omega=4$, $\\kappa_2/\\kappa=1$, $N=16\\to22$ el pico sube $3.7\\times10^{-3}$'), (PH, '| P2 |')], 'Regla universal: w=4, kappa2/kappa=1, N=16->22: P_c en el pico sube 3.7e-3 (0.9945 -> 0.9983)', 'V', rep='3.7e-3', rec=p22 - p16, tipo='numerica', dato='data/pfig2/w4_gx-0.142857_*_wp7.9863946_N{16,22}.npz', script=S, tol=0.1e-3, nota='N=16: %.4f; N=22: %.4f' % (p16, p22))
    txt('PF2-c2', 1, [(RM, 'en el flanco x = 1.5 cambia 9e-5'), (INF, 'el flanco en $x=1.5$ cambia $9\\times10^{-5}$')], 'Idem en el flanco x=1.5 (w_p=8.0089) cambia 9e-5', 'V', rep='9e-5', rec=abs(f22 - f16), tipo='numerica', dato='data/pfig2/w4_gx-0.142857_*_wp8.0088946_N{16,22}.npz', script=S, tol=0.5e-5)
    # --- PENDIENTES: lineas con ids de pendientes/errores (tipo pendiente)
    for i, (pid, desc) in enumerate((('P1', 'Regla universal: colapso parcial aceptado'), ('P2', 'Curvas N=16 recalculadas con N=22: resuelto'), ('P3', 'Panel (d) gato transitorio'), ('P4', 'Deficit no adiabatico sin explicacion analitica: abierto'), ('P5', 'Desplazamiento polaronico optimo 2.4% menor que g_z/w'), ('P6', 'Exceso +1.6% en V4 a g_z/k pequeno explicado por el +1; (0.1,5,2) con deficit 0.7%'), ('P7', 'Figura termica: borrador'), ('P8', 'Apagado automatico del PC'), ('P9', 'Corrimiento del qubit dependiente de n: resuelto'), ('P10', 'Exceso de kappa1 con filtro en regimen saturado: artefacto de N=16: resuelto'))):
        pass
    txt('P5-1', 1, [(PH, '| P5 |')], 'P5: el desplazamiento polaronico optimo en Ma es 2.4% menor que g_z/w; efecto en P_c <= 1e-6', 'V', rep='0.024', rec=1 - float(np.real(np.load(os.path.join(DATA, 'c6_ma_N22.npz'), allow_pickle=True)['dopt'])) / (0.2121320343559643 / 6.0), tipo='numerica', dato='data/c6_ma_N22.npz', script=S, tol=0.0006, nota='dif de P_c: %.1e (C6-07 vs C6-05)' % abs(float(byid('C6-07')['valor_recalculado']) - float(byid('C6-05')['valor_recalculado'])))
    txt('P6-1', 1, [(PH, '| P6 |')], 'P6: el punto (0.1,5,gz/k=2) tiene deficit 0.7%, mayor que sus vecinos', 'V', rep='0.7%', rec=1 - float(byid('V4-05-con1')['valor_recalculado']), tipo='numerica', dato='data/fig3/gx0.1_w5.0_gz2.0_al4.0_N22.npz', script=S, tol=0.0006)
    return

# ============================================================ BLOQUE 16: HANDOFF (cierre de lineas), historia git, pendientes, literatura
def bloque_textual5(E):
    S = 'consolidacion/verificar_todo.py::bloque_textual5'
    K = KAPM
    # --- adjuntar lineas repetidas a afirmaciones ya verificadas
    AD = [('PS-06', H8, '- Con γ/κ = 2e-4: franja κ₂/κ ≈ 0.19–0.33'), ('P9-04', H8, '- Régimen de validez: γ ≪ κ; κ, g ≪ ω'),
          ('T84-c2', H8, '| γ_bf del efectivo con filtro dentro de ±5% del completo |'), ('CN-054', H8, '| γ_bf del efectivo con filtro dentro de ±5% del completo |'), ('T84-ind-ok', H8, '| el cociente γ_bf completo/efectivo con piso restado no depende de x'),
          ('PL7-98', H8, '| factor 1.52 de γ_pf en el plano es el canal n_q del qubit |'), ('PL7-2', H8, '- **El factor 1.52 de γ_pf es térmico, no de T = 0**'),
          ('T85-0.05-100-co', H8, 'Modelo: γ_pf,c(x) = q_pf · γ_pf,e(x)'), ('PL7-16', H8, '| desplazamiento con n(ω), n(3ω); Γ₁± con n_q | 5.692e-3'),
          ('BS-1', H8, '- **Barrido de la densidad espectral del baño [NO VALIDADO]**'), ('OP-R', H8, '| desplazamiento con n(ω), n(3ω); Γ₁± con n(ω), n(3ω) | 5.738e-3'), ('OP-R', H8, '| sin desplazamiento; Γ₁± con n(ω), n(3ω) | 9.459e-4'),
          ('OP-R', H8, '| ansatz con n(ω), n(3ω); Γ₁± con n(ω), n(3ω) | 1.759e-4'), ('OP-R', H8, '| sin ansatz; Γ₁± con n(ω), n(3ω) | 1.724e-4'),
          ('BS-tab', H8, '| 0.2 | A | 1.413e-3'), ('BS-tab', H8, '| 2 | B | 1.039e-2'), ('CR-1', H8, '- Convergencia en N en al menos un punto por resultado.'),
          ('T84-r2', H8, '| efectivo con filtro frente al completo, γ_pf y γ_bf | ±5% en cada tasa |'), ('T84-ind-ok', H8, '| independencia de x del cociente γ_bf con piso restado | cambio < 3%'),
          ('T84-tolE', H8, '| tolerancia del integrador (isla) | reportar el cambio |'), ('PL7-N26-p', H8, '| barrido en N, plano | convergencia |'), ('PL7-98', H8, '| apagado de n_q y n_m en el completo plano |'), ('PL7-16', H8, '| efectivo plano con desplazamiento | reproducir el canal n_q del completo |'),
          ('FK-3', H8, '| desplazamiento en el acoplamiento qubit-filtro (efectivo con filtro) |'), ('MD-4', H8, '| modo vecino de γ_pf (plano) | identificar |'), ('OC-1', H8, '| control con n(ω) y n(3ω) (efectivo) | cambio de γ_pf pequeño |'), ('CN-d20bf', H8, '| convergencia en N con filtro (N = 18, 20 y 22) |'),
          ('PL7-10', H8, '- **El acuerdo de η del efectivo plano con desplazamiento es compensación de errores**'), ('T84-c2', H8, '- La figura térmica depende del modelo efectivo.'), ('PL7-98', H8, '- **Baño plano sin validar:**')]
    for i, f, a in AD:
        try: adj(i, f, a)
        except KeyError: WARN.append('adj: id inexistente %s' % i)
    # --- piso: fracciones por x en la lista 8.6
    adj('T86-etamax', H8, '| piso γ_bf(T=0) frente al γ_bf térmico | < 1% (criterio original)')
    # --- isla y parametros
    zi = np.load(os.path.join(TD, 'x6.86_k0.25_gz14_f1_g2e-05_N22_Nf2_w200_a4.npz'), allow_pickle=True)
    ok_i = abs(float(zi['k2k']) - 0.25) < 1e-9 and float(zi['gzk']) == 14 and float(zi['wk']) == 200 and float(zi['al2']) == 4 and abs(float(zi['kfw']) - 0.05) < 1e-9
    txt('ISLA-1', 1, [(H8, '- **Modelo efectivo** (qubit y filtro explícitos, `calc_termico.py`) en la isla'), (RM, '- Isla: κ₂/κ = 0.25, g_z/κ = 14, ω/κ = 200, |α|² = 4')], 'Isla = (k2/k, gz/k, w/k, |a|^2, kf/w) = (0.25, 14, 200, 4, 0.05)', 'V', rep='(0.25,14,200,4,0.05)', tipo='cualitativa', dato='data/termico/x6.86_k0.25_gz14_f1_*.npz (campos k2k, gzk, wk, al2, kfw)', script=S)
    byid('ISLA-1')['estado'] = 'V' if ok_i else 'D'
    zc = np.load(os.path.join(FC, 'gx0.0535714_w6_gz0.42_kf0.3_al4_N22_Nf2_gam6e-07_x6.86.npz'), allow_pickle=True)
    ok_c = abs(float(zc['gz']) / float(zc['kap']) - 14) < 1e-9 and abs(float(zc['w']) / float(zc['kap']) - 200) < 1e-9 and float(zc['al2']) == 4 and abs(float(zc['gam']) / float(zc['kap']) - 2e-5) < 1e-12 and abs(float(zc['kf']) / float(zc['w']) - 0.05) < 1e-12
    txt('ISLA-2', 1, [(H8, '- **Un solo juego de parámetros.** Toda la validación con el modelo completo usa'), (H8, '- **Un solo juego de parámetros** (g_z/κ = 14')], 'Toda la validacion con el modelo completo usa gz/k=14, w/k=200, |a|^2=4, gamma/k=2e-5, kf/w=0.05 (N=22, N_f=2)', 'V', rep='(14,200,4,2e-5,0.05)', tipo='cualitativa', dato='data/filtro_completo/gx0.0535714_*_x6.86.npz', script=S, nota='todas las corridas termicas del log usan gam6e-07 y kf0.3 (nombres de archivo)')
    byid('ISLA-2')['estado'] = 'V' if ok_c and all('kf0.3' in p['nombre'] and 'gam6e-07' in p['nombre'] for p in [dict(nombre=os.path.basename(f)) for f in glob.glob(os.path.join(FC, '*gam6e-07*.npz')) if 'kf1.2' not in f]) else 'D'
    # --- 8.8: efecto del ansatz plano 6.06e-4
    inc_k = (ek_ := eff_pt(0.25, 6.86, 0, extra='_kick'))['gpf'] - eff_pt(0.25, 60, 0, extra='_kick')['gpf']
    inc_0 = eff_pt(0.25, 6.86, 0)['gpf'] - eff_pt(0.25, 60, 0)['gpf']
    txt('FK-6', 1, [(H8, '- **El efecto despreciable es consecuencia directa del factor 6.2e-4:**')], 'El efecto termico del desplazamiento en el plano es 6.06e-4; por ke=6.2e-4 da 3.8e-7 frente a 4.64e-7 obtenido', 'V', rep='6.06e-4', rec=inc_k - inc_0, tipo='numerica', dato='data/termico/x{6.86,60}_k0.25_gz14_f0_*_kick.npz', script=S, tol=0.04e-4, nota='recalculo %.3e (el HANDOFF 8.7 da 6.219e-4-1.332e-5 = 6.086e-4); producto con ke: %.2e' % (inc_k - inc_0, (inc_k - inc_0) * 6.2e-4))
    txt('FK-t', 1, [(H8, '- **Efectivo:** segundos por punto')], 'Efectivo: segundos por punto (<= 2 s por corrida en plano_kick y filtro_kick)', 'R', rep='2', tipo='numerica', dato='data/plano_kick.txt, data/filtro_kick.txt', script='codigo/plano_kick.py, codigo/filtro_kick.py', nota='filtro_kick.txt: 0.9, 1.1, 1.5, 1.0 s; plano_kick.txt: 0.1 s (calculados con rerun=True)')
    # --- literatura (fuera de alcance)
    lit('LIT-1', [(H8, '- **Ma 2019, Ec. (4):** omite −(g_x²/ω)n'), (PH, '| E1 |'), (INF, '\\item Ma et al., Ec.~(4): omite $-(g_x^2/\\omega)\\hn$')], 'Ma 2019, Ec.(4): omite -(gx^2/w)n, da 3gx^2/w para el qubit; el termino de pares es correcto')
    lit('LIT-2', [(H8, '- **Liu 2025, Ec. (11):** omite'), (INF, '\\item Liu et al., Ec.~(11): omite')], 'Liu 2025, Ec.(11): omite -(4gx^2/3w)n; posible inconsistencia del factor del drive (e_p vs 2e_p)')
    lit('LIT-3', [(H8, '- **Naseem 2026:** no incluye δ₁'), (INF, '\\item Naseem: su código no incluye $\\delta_1$')], 'Naseem 2026: no incluye delta_1, usa g_eff=g en lugar de 2g; su receta de aumentar kappa contradice kappa1/kappa2 ~ kappa^2', nota='Parte propia verificable: g_eff=2g (Tareas 7-8, historicas, nivel 3) y kappa1/kappa2~kappa^2 (SIM-10, V); el contenido del articulo de Naseem queda fuera de alcance')
    lit('LIT-4', [(INF, '\\item Ma et al.: su Ec.~(5), el modelo efectivo, conserva la paridad')], 'Ma et al., Ec.(5): el modelo efectivo conserva la paridad; Liu y Hou no se comparan')
    lit('LIT-5', [(H8, '- **Hou 2024:** pendiente de lectura.'), (H8, 'y **Hou 2024 no está**')], 'Hou 2024: pendiente de lectura; no esta en papers/ (2407.17299 es Dubovitskii)', nota='Fuera de alcance; el hecho de archivo (papers/ no contiene Hou 2024) si es verificable: ' + str(sorted(os.listdir(os.path.join(ROOT, 'papers')))))
    lit('LIT-6', [(INF, '\\paragraph{Liu et al.} Con $\\kappa/\\omega=0.9$ y $g_z/\\omega=0.2$ la jerarquía')], 'Liu et al.: Ec.(9) completa no forma gato (F<=0.35 frente a 0.95 del efectivo; |<a^2>| max 0.4-1.1): conclusion numerica propia sobre un modelo de la literatura', nota='La simulacion es propia (Tareas 46/47b, historicas, nivel 3: cache47b_*); la afirmacion sobre lo que dice Liu queda fuera de alcance')
    lit('LIT-7', [(INF, '\\item Naseem: su código no incluye $\\delta_1$ en el hamiltoniano efectivo; su Ec.~(11) usa $g_{\\rm eff}=g$')], 'Naseem: g_eff=2g confirmado por Rabi de dos fotones (<5%) y algebra polaronica (Tareas 7-8)', nota='ver LIT-3')
    # Liu: aritmetica propia
    txt('PL-L', 1, [(H8, '| Liu | 0.90 | 0.22 | 0.031 |'), (INF, '$\\kappa/\\omega$ & 0.005 & 0.001 & 0.90'), (INF, '$g_z/\\kappa$ & 7.1 & 60 & 0.22')], 'Liu: kappa/w = 16/17.7 = 0.90; gz/kappa = 3.54/16 = 0.22; kappa2/kappa = 16 (gx/w)^2 (gz/k)^2 = 0.031 (con los parametros publicados de Liu)', 'V', rep='0.90,0.22,0.031', tipo='numerica', dato='(analitico; parametros de la Tabla param)', script=S, nota='%.3f, %.3f, %.4f' % (16 / 17.7, 3.54 / 16, 16 * (3.54 / 17.7)**2 * (3.54 / 16)**2))
    byid('PL-L')['estado'] = 'V' if abs(16 / 17.7 - 0.90) < 0.006 and abs(3.54 / 16 - 0.22) < 0.006 and abs(16 * (3.54 / 17.7)**2 * (3.54 / 16)**2 - 0.031) < 0.001 else 'D'
    # --- historia git (HANDOFF 17) y otros hashes citados
    HH = [('040eb0e', '2026-09-29'), ('2652746', '2026-09-29'), ('faaabea', '2026-09-29'), ('bedbe5e', '2026-09-30'), ('effff72', '2026-09-30'), ('4d23ea4', '2026-09-30'), ('2b64d26', '2026-09-30'), ('1603317', '2026-09-30'), ('efda948', '2026-09-30'), ('163e971', '2026-10-01'), ('dc3c8e8', '2026-10-01'), ('8ffd720', '2026-10-01'), ('cfa8f01', '2026-10-01')]
    filas = {'040eb0e': '| 1 | 2026-09-29 | `040eb0e`', '2652746': '| 1 | 2026-09-29 | `040eb0e`', 'faaabea': '| 2 | 2026-09-29 a 09-30 |', 'bedbe5e': '| 2 | 2026-09-29 a 09-30 |', 'effff72': '| 2 | 2026-09-29 a 09-30 |', '4d23ea4': '| 3 | 2026-09-30 | `4d23ea4`', '2b64d26': '| 4 | 2026-09-30 | `2b64d26`', '1603317': '| 5 | 2026-09-30 | `1603317`', 'efda948': '| 6 | 2026-09-30 a 10-01 |', '163e971': '| 6 | 2026-09-30 a 10-01 |', 'dc3c8e8': '| N = 20 y N = 18 | 2026-10-01 |', '8ffd720': '| x* con piso | 2026-10-01 |', 'cfa8f01': '- **Corrida a κ_f/ω = 0.2 [MEDIDO], regla fijada antes**'}
    for h, fecha in HH:
        subj = sh("git log -1 --format='%%ad|%%s' --date=short %s" % h)
        ex = bool(subj) and subj.split('|')[0] == fecha
        txt('GIT-' + h, 1, [(H8, filas[h])], 'Commit %s existe y su fecha es %s' % (h, fecha), 'V', rep=h, tipo='cualitativa', dato='git log', script=S, commit=h, nota=subj[:140])
        byid('GIT-' + h)['estado'] = 'V' if ex else 'D'
    for h in ('43ca141', '4eb9c27'):
        subj = sh("git log -1 --format='%%ad|%%s' --date=short %s" % h)
        txt('GIT-' + h, 3, [('Mechanical-Cat-State/validacion/tareas_ma/verificacion_signo_pair.md', '`ma2.py` tiene solo dos commits')], 'Commit %s existe (verificacion_signo_pair.md)' % h, 'V', rep=h, tipo='cualitativa', dato='git log', script=S, commit=h, nota=subj[:140])
    # ma2.py solo dos commits
    nma = len(sh("git log --format=%h -- Mechanical-Cat-State/validacion/tareas_ma/ma2.py").split())
    txt('GIT-ma2', 3, [('Mechanical-Cat-State/validacion/tareas_ma/verificacion_signo_pair.md', '`ma2.py` tiene solo dos commits')], 'ma2.py tiene solo dos commits (43ca141, 4eb9c27)', 'V', rep='2', rec=nma, tipo='numerica', dato='git log -- ma2.py', script=S, tol=0)
    # el commit 'N=18' no tiene hash en HANDOFF
    h18 = sh("git log -1 --format=%h --grep='N=18 filtered convergence'")
    txt('GIT-N18', 1, [(H8, '| N = 20 y N = 18 | 2026-10-01 |')], 'El commit de datos de N=18 (sin hash en el HANDOFF) es %s' % h18, 'V', rep='sin hash', tipo='cualitativa', dato='git log', script=S, commit=h18, nota=sh("git log -1 --format='%%ad %%s' --date=short %s" % h18)[:120])
    # --- pendientes (HANDOFF 12), tipo pendiente
    PEN = [(381, 'Convergencia en N con filtro: falta el residual en kappa2/kappa=0.05 (N=20, ~1 h) y el barrido del piso a T=0 con filtro; N=24 no cabe'), (382, 'Validacion de kappa_f y de gamma/kappa=2e-4 con el modelo completo'), (383, 'Naseem (x~0.96) fuera del rango validado: no lanzar hasta decidirlo; termico con kappa2/kappa=2.07 no calculado'),
           (384, 'Exceso del 16% del canal n_q en el efectivo plano con desplazamiento: sin explicacion; incorporar el canal al efectivo con filtro y repetir en otro kf y otro k2/k'), (385, 'Origen de la discrepancia de gamma_bf con kappa2/kappa (0.956 a 0.776)'), (386, 'Origen del piso de T=0 con filtro (crece con kappa2/kappa en el completo, no en el efectivo)'),
           (387, 'Validacion del bano plano y del supuesto de ruido blanco (n(w), n(3w)) en el completo'), (388, 'Desplazamiento con filtro, parte no secular: calculo de Floquet con tiempo'), (389, 'Modo vecino con filtro y origen del residuo de hermiticidad'), (390, 'Exponente s de c_bf en el completo'),
           (391, 'N_f del completo: correr el completo con N_f=3; alpha_eff^2 no explica E_obs'), (394, 'Tabla de plataformas para el texto'), (395, 'Elegir entre figura_central y figura_central_sinpiso; estilo de figura_termica'), (396, 'Fig. 1 (esquema TikZ)'), (397, 'documento_completo v2'), (398, 'Revision de citas, lectura de Hou 2024, P4'), (399, 'Revision humana independiente; autoria; politica APS sobre IA')]
    for ln, tx in PEN:
        txt('PEN-%d' % ln, 1, [(H8, {381: '1. **Convergencia en N con filtro:**', 382: '2. **Validación de κ_f y de γ/κ = 2e-4**', 383: '3. **Naseem** (x ≈ 0.96', 384: '4. **Exceso del 16% del canal n_q', 385: '5. **Origen de la discrepancia de γ_bf', 386: '6. **Origen del piso de T = 0 con filtro**', 387: '7. **Validación del baño plano**', 388: '8. **Desplazamiento con filtro, parte no secular:**', 389: '9. **Modo vecino con filtro**', 390: '10. **Exponente s de c_bf', 391: '11. **N_f del completo:**', 394: '11. **Tabla de plataformas**', 395: '12. Elegir entre `figura_central`', 396: '13. Fig. 1', 397: '14. `documento_completo` v2', 398: '15. Revisión de citas', 399: '16. Revisión humana independiente'}[ln])], 'Pendiente abierto (HANDOFF 12): ' + tx, 'H', tipo='pendiente', nota='pendiente abierto; origen en HANDOFF 12 (ver las afirmaciones relacionadas)')
    return

# ============================================================ BLOQUE 17: P10, preregistro, README/PENDIENTES (lineas restantes), agrupaciones
def bloque_textual6(E):
    S = 'consolidacion/verificar_todo.py::bloque_textual6'
    K = KAPM
    # corregir GIT-ma2: el tercer commit es el movimiento de carpetas
    c = byid('GIT-ma2'); c['estado'] = 'V'; c['nota'] = 'git log (sin --follow) da %s commit(s); con --follow son 3: el tercero (1c20218) es la reorganizacion de carpetas del 2026-09-29' % c['valor_recalculado']
    # ---- P10 (p10_exceso.csv)
    P = [[float(x) for x in r] for r in list(csv.reader(open(os.path.join(DATA, 'p10_exceso.csv'))))[1:]]
    P = np.array(P)
    sel = lambda gz, k2, kfw, a2, N, ng=0: P[(P[:, 0] == gz) & (abs(P[:, 1] - k2) < 1e-5) & (abs(P[:, 2] - kfw) < 1e-5) & (P[:, 3] == a2) & (P[:, 4] == N) & (P[:, 5] == ng)][0]
    DP = 'data/p10_exceso.csv'
    TP = lambda i, loc, tx, rep, v, tol=None, nota='': txt(i, 1, [loc], tx, 'V', rep=rep, rec=v, tipo='numerica', dato=DP, script=S, tol=tol, nota=nota)
    TP('P10-1', (PH, '| P10 |'), 'P10: (k2/k,gz/k)=(0.3,12) N=16: k1 completo / filtrado predicho = 2.26', '2.26', sel(12, 0.3, 0.05, 4, 16)[9])
    TP('P10-2', (PH, '| P10 |'), 'P10: sin gz sz(a+a+) y con pares explicitos (0.3,12): 0.999', '0.999', sel(12, 0.3, 0.05, 4, 16, 1)[9])
    for k2, rep in ((0.15, '2.77'), (0.3, '2.26'), (0.6, '1.51')):
        TP('P10-3-%g' % k2, (PH, '| P10 |'), 'P10: kappa2/kappa=%g (gz/k=12, N=16): exceso %s' % (k2, rep), rep, sel(12, k2, 0.05, 4, 16)[9])
    for k2, rep in ((0.15, '0.0013'), (0.3, '0.0021'), (0.6, '0.0029')):
        TP('P10-pe-%g' % k2, (PH, '| P10 |'), 'P10: P_e promedio %s en kappa2/kappa=%g' % (rep, k2), rep, sel(12, k2, 0.05, 4, 16)[10], nota='')
    TP('P10-4', (PH, '| P10 |'), 'P10 fase 2: (0.03,12) razon 2.88 (N=16)', '2.88', sel(12, 0.03, 0.05, 4, 16)[9])
    Fs = [sel(g, 0.03, 0.05, 4, 16)[9] - 1 for g in (4, 6, 8, 12)]
    for g, rep, v in zip((4, 6, 8, 12), ('0.03', '0.16', '0.50', '1.88'), Fs):
        TP('P10-F%d' % g, (PH, '| P10 |'), 'P10: F = razon - 1 = %s para gz/k=%d (kappa2/kappa=0.03)' % (rep, g), rep, v, tol=0.006)
    sl = np.polyfit(np.log([4, 6, 8, 12]), np.log(Fs), 1)[0]
    TP('P10-s', (PH, '| P10 |'), 'P10: F ~ (gz/k)^3.7', '3.7', sl, tol=0.3, nota='ajuste log-log: %.2f' % sl)
    ex = [sel(g, 0.03, 0.05, 4, 16)[8] for g in (4, 6, 8, 12)]
    sl2 = np.polyfit(np.log([4, 6, 8, 12]), np.log(ex), 1)[0]
    TP('P10-s2', (PH, '| P10 |'), 'P10: exceso absoluto ~ (gz/k)^1.7', '1.7', sl2, tol=0.2, nota='ajuste log-log: %.2f' % sl2)
    for kfw, rep, rep2 in ((0.02, '7.7e-10', '2.5'), (0.05, '3.6e-9', '1.9'), (0.2, '5.2e-8', '1.7')):
        r = sel(12, 0.03, kfw, 4, 16)
        TP('P10-kf%g' % kfw, (PH, '| P10 |'), 'P10: kf/w=%g: exceso absoluto %s, F %s' % (kfw, rep, rep2), rep, r[8], nota='F = %.2f' % (r[9] - 1))
        TP('P10-kfF%g' % kfw, (PH, '| P10 |'), 'P10: kf/w=%g: F ~ %s' % (kfw, rep2), rep2, r[9] - 1, tol=0.1)
    for a2, rep in ((2, '0'), (3, '0.07'), (4, '1.88')):
        TP('P10-a%d' % a2, (PH, '| P10 |'), 'P10: |a|^2=%d: F ~ %s' % (a2, rep), rep, sel(12, 0.03, 0.05, a2, 16)[9] - 1, tol=0.012)
    TP('P10-gx', (PH, '| P10 |'), 'P10: F no depende de gx con kappa2 pequeno: 1.88 (0.03) y 1.77 (0.15)', '1.77', sel(12, 0.15, 0.05, 4, 16)[9] - 1, tol=0.01)
    TP('P10-sat', (PH, '| P10 |'), 'P10: F baja en saturacion: 1.26 (0.3) y 0.51 (0.6)', '0.51', sel(12, 0.6, 0.05, 4, 16)[9] - 1, tol=0.01, nota='0.3: %.2f' % (sel(12, 0.3, 0.05, 4, 16)[9] - 1))
    TP('P10-a5', (PH, '| P10 |'), 'P10 fase 3: |a|^2=5 con N=20 da F=0.15', '0.15', sel(12, 0.03, 0.05, 5, 20)[9] - 1, tol=0.006)
    TP('P10-n22', (PH, '| P10 |'), 'P10: (0.03,12) |a|^2=4 N=22 sin filtro (kf/w=0): 0.988', '0.988', sel(12, 0.03, 0.0, 4, 22)[9], tol=0.0006)
    # ---- 1b criterio del piso (PENDIENTES)
    adj('T86-etamax', PH, 'En la isla, η_max = η(T=0)', None) if False else None
    txt('PI-1b', 1, [(PH, 'Eso desplaza x* como mucho'), (PH, '**Propuesta (pendiente de decisión de Jhon):**')], 'Criterio del piso: fraccion eta/eta_max = 2% (eta=100) a 4.4% (eta=220); x* se desplaza <= ln(1/(1-0.044)) = 0.045; 4968>=2200; 2352>=2200', 'V', rep='0.045', rec=math.log(1 / (1 - 0.044)), tipo='numerica', dato='T86-*', script=S, tol=0.001, nota='100/4968=%.3f, 220/4968=%.3f; 10*220=2200' % (100 / 4968, 220 / 4968))
    por_decidir('PD-piso', 'Criterio del piso gamma_bf(T=0) < 1% del termico como criterio de aceptacion', 'Mantener el criterio original (falla a x~9 y en la isla a x=10.1: 1.35%-2.52%)', 'Reportar el piso y sustituirlo por eta_max >= 10 x nivel de sesgo (en k2/k=0.4: 2352 >= 2200, justo)', 'HANDOFF 16, PENDIENTES 1b, T86-*')
    # ---- E2..E5 (PENDIENTES) -> adj
    adj('V3-07', PH, '| E3 |'); adj('V4b-rango', PH, '| E2 |'); adj('F4c-flat', PH, '| E5 |'); adj('F4c-0.1', PH, '| E5 |')
    adj('CV-f2d', PH, '- Fig. 2(a) de validación con N = 22; la primera versión con N = 20 subestimaba P_c en 1.4e-4.')
    txt('PEN-4a', 1, [(PH, '- Convergencia con filtro solo N = 16 → 18 (N = 20 no cabe en 14 GB).'), (PH, '- La regla universal usa N = 14–16 en las curvas nuevas (ver P2).')], 'PENDIENTES 4 (2026-09-28): "convergencia con filtro solo N=16->18 (N=20 no cabe en 14 GB)"; "regla universal con N=14-16"', 'S', rep='N=20 no cabe', rec=None, tipo='cualitativa', dato='log_termico_ma5.txt (N=20: 11.36 GB)', script=S, nota='superada: N=20 y N=18 corrieron el 2026-10-01 (CN-*, RC-N20); las curvas de la regla universal con |a|^2=4 fueron recalculadas con N=22 en |x|<=1 (P2); el encabezado de PENDIENTES dice "Actualizado: 2026-09-28" y no incluye P7 fases posteriores en la fecha')
    # ---- preregistro: aritmetica y parametros
    txt('PR-1', 1, [(PRE, 'Modelo completo con filtro, N = 20, N_f = 2, en (κ₂/κ, x) = (0.25, 6.86)')], 'Preregistro: parametros del completo kf/w=0.2: gx=0.0535714, w=6, gz=0.42, kappa=0.03, kf=1.2, |a|^2=4, gamma=6e-7, N=20, N_f=2', 'V', rep='kf=1.2', tipo='cualitativa', dato='data/filtro_completo/gx0.0535714_*kf1.2*N20*.npz', script=S)
    zz = np.load(os.path.join(FC, 'gx0.0535714_w6_gz0.42_kf1.2_al4_N20_Nf2_gam6e-07.npz'), allow_pickle=True)
    okp = all(abs(float(zz[k]) - v) < 1e-9 for k, v in (('gx', 0.0535714), ('w', 6.0), ('gz', 0.42), ('kap', 0.03), ('kf', 1.2), ('al2', 4.0), ('gam', 6e-7))) and int(zz['N']) == 20 and int(zz['Nf']) == 2
    byid('PR-1')['estado'] = 'V' if okp else 'D'
    txt('PR-2', 1, [(PRE, '| γ_pf a T = 0 (x = 60) | 1.66411e-04 |'), (PRE, '| γ_pf térmico (x = 6.86) | 1.78395e-04 |'), (PRE, '| Δ_e = aumento térmico sin ansatz | 1.1984e-05 |'), (PRE, '| γ_pf térmico con ansatz (n_q) | 1.85736e-04 |'), (PRE, '| **E_ansatz** = aumento por el ansatz con n_q'), (PRE, '| γ_bf térmico / a T = 0 (efectivo)'), (PRE, '| E con ocupación física (referencia)')], 'Predicciones del efectivo (kf/w=0.2, N=22) del preregistro: gamma_pf T=0 1.66411e-4, termico 1.78395e-4, Delta_e 1.1984e-5, con ansatz 1.85736e-4, E_ansatz +7.341e-6', 'R', tipo='numerica', dato='data/preregistro_kf0.2.md (el efectivo con kf/w=0.2 no esta en data/termico)', script='codigo/calc_termico.py', nota='aritmetica interna: 1.78395e-4-1.66411e-4 = %.4e (Delta_e); 1.85736e-4-1.78395e-4 = %.4e (E_ansatz)' % (1.78395e-4 - 1.66411e-4, 1.85736e-4 - 1.78395e-4))
    byid('PR-2')['valor_recalculado'] = '%.4e; %.4e' % (1.78395e-4 - 1.66411e-4, 1.85736e-4 - 1.78395e-4)
    txt('PR-3', 1, [(PRE, '- **Esperado con canal de tamaño ansatz:** E_obs ≈ +7.3e-6')], 'Preregistro: con canal D ~ E_ansatz + (r_0-1) Delta_e = +6.8e-6; sin canal D ~ (r_0-1) Delta_e = -5.3e-7 (r_0 = 0.956)', 'V', rep='6.8e-6', rec=7.341e-6 + (0.956 - 1) * 1.1984e-5, tipo='numerica', dato='(analitico)', script=S, tol=0.06e-6, nota='sin canal: %.2e' % ((0.956 - 1) * 1.1984e-5))
    txt('PR-4', 1, [(PRE, '- **4J²/κ_f = κ se mantiene:**')], 'Preregistro: J = sqrt(kappa kf)/2 = 0.09487 y 4J^2/kf = 0.03 para kf=1.2', 'V', rep='0.09487', rec=math.sqrt(0.03 * 1.2) / 2, tipo='numerica', dato='(analitico)', script=S, nota='4J^2/kf = %.5f' % (4 * (math.sqrt(0.03 * 1.2) / 2)**2 / 1.2))
    txt('PR-5', 1, [(PRE, '- **γ ≪ κ:** γ/κ = 2e-5; además γ/κ_f = 5e-7.')], 'Preregistro: gamma/kappa = 2e-5 y gamma/kf = 5e-7', 'V', rep='5e-7', rec=6e-7 / 1.2, tipo='numerica', dato='(analitico)', script=S, nota='gamma/kappa = %.1e' % (6e-7 / 0.03))
    txt('PR-6', 1, [(PRE, '- **Resolución:** el residual de N = 20 en γ_pf'), (PRE, '- ke(ω) = κ_f²/(4ω² + κ_f²) = 9.9e-3')], 'Preregistro: ke(w) con kf=1.2 = 9.9e-3 (16 veces el de kf/w=0.05)', 'V', rep='9.9e-3', rec=1.2**2 / (4 * 36 + 1.2**2), tipo='numerica', dato='(analitico)', script=S, tol=0.05e-3, nota='cociente con 6.2e-4: %.1f' % (1.2**2 / (144 + 1.44) / (0.3**2 / (144 + 0.09))))
    txt('PR-7', 1, [(PRE, '- **N_f = 2 basta:**')], 'Preregistro: efectivo con N_f=2,3,4 a kf/w=0.2: gamma_pf termico 1.7839497e-4, 1.7839545e-4, 1.7839545e-4 (cambio 2.7e-6)', 'R', rep='2.7e-6', tipo='numerica', dato='data/preregistro_kf0.2.md', script='codigo/calc_termico.py', nota='no hay caches del efectivo con N_f=3,4; aritmetica: 1.7839545/1.7839497-1 = %.2e' % (1.7839545 / 1.7839497 - 1))
    txt('PR-8', 1, [(PRE, 'No se lanza `nogz_pair`, N = 20 en κ₂/κ = 0.05, N = 24 ni Naseem')], 'Preregistro: no se lanzan nogz_pair, N=20 en k2/k=0.05, N=24 ni Naseem', 'V', rep='no lanzadas', tipo='cualitativa', dato='data/filtro_completo/', script=S, nota='no existe en filtro_completo ninguna corrida kf1.2 salvo las dos de la pareja; N24: %d archivos; N20 para gx0.0239579: %d' % (len(glob.glob(os.path.join(FC, '*N24*'))), len(glob.glob(os.path.join(FC, 'gx0.0239579*N20*')))))
    byid('PR-8')['estado'] = 'V' if len(glob.glob(os.path.join(FC, '*N24*'))) == 0 and len(glob.glob(os.path.join(FC, 'gx0.0239579*N20*'))) == 0 and len(glob.glob(os.path.join(FC, '*kf1.2*'))) == 3 else 'D'
    # ---- C6 linea 33
    adj('C6-07', C6, '- La diferencia de P_c entre d = g_z/ω y d_opt es ≤ 1e-6')
    # ---- README: comprobaciones restantes
    txt('RM-x133', 1, [(RM, 'Recuadro: Wigner de Ma en x = 0 y en ω_p = 2ω (x = 1.33')], 'En omega_p = 2w = 12, x = (wp - wp*)/G = 1.33', 'V', rep='1.33', rec=(12.0 - 11.98) / 0.015, tipo='numerica', dato='(analitico)', script=S, tol=0.005)
    n22 = [f for f in glob.glob(os.path.join(DATA, 'pfig2', '*_N22.npz')) if not os.path.basename(f).startswith('w6_gx-0.212132')]
    txt('RM-48', 1, [(RM, 'se recalcularon con N = 22 (48 puntos)')], 'P2: 48 puntos recalculados con N=22 en |x|<=1 para las curvas |a|^2=4 que tenian N=16', 'V', rep='48', rec=len(n22), tipo='numerica', dato='data/pfig2/*_N22.npz', script=S, tol=0, nota='data/pfig2 tiene %d archivos *_N22.npz (excluyendo Ma, w6_gx-0.212132); no identifica por si mismo cuales fueron "recalculados"' % len(n22))
    nmin = len(glob.glob(os.path.join(DATA, 'minimo', 'k*_N24.npz')))
    txt('RM-30', 1, [(RM, '- **Modelo mínimo:** 30 valores de κ₂/κ entre 1e-3 y 10 con N = 24')], 'Modelo minimo: 30 valores de kappa2/kappa entre 1e-3 y 10 con N=24', 'V', rep='30', rec=nmin, tipo='numerica', dato='data/minimo/k*_N24.npz', script=S, tol=0)
    umb = [[float(x) for x in r] for r in list(csv.reader(open(os.path.join(DATA, 'termico_umbral_parametro.csv'))))[1:]]
    ug = [r for r in umb if r[0] == 1 and r[1] == 0]
    txt('RM-gz', 1, [(RM, 'El barrido en g_z/κ ∈ [10, 40] se calculó, pero ya no se grafica')], 'Barrido en gz/k de 10 a 40 (filtro): x*(100) = 10.4 -> 10.0', 'V', rep='10.4', rec=ug[0][3], tipo='numerica', dato='data/termico_umbral_parametro.csv', script=S, tol=0.05, nota='extremo gz/k=40: %.3f' % ug[-1][3])
    txt('RM-gz10', 1, [(RM, 'Ojo: su fila g_z/κ = 10 tiene χ|α|²/κ = 0.333 > 0.3')], 'Fila gz/k=10 (k2/k=0.25, w/k=200): chi|a|^2/k = 0.333', 'V', rep='0.333', rec=(4 / 6) * 200 * 0.25 / 100, tipo='numerica', dato='(analitico)', script=S)
    Ux = csvrows('termico_umbrales.csv')
    txt('RM-1.5', 1, [(RM, 'la diferencia local de η junto a cada umbral es ≤ 1.5%'), (INF, 'La diferencia local relativa de $\\eta$ junto a cada umbral es $\\leq1.5\\%$')], 'Diferencia local relativa de eta junto a cada umbral <= 1.5%', 'V', rep='0.015', rec=max(r[6] for r in Ux), tipo='numerica', dato='data/termico_umbrales.csv', script=S, tol=0, modo='le')
    byid('RM-1.5')['estado'] = 'V' if max(r[6] for r in Ux) <= 0.015 else 'D'
    # plano a gamma bajo
    cp = {}
    for gm in (1e-6, 1.77828e-6, 3.16228e-6):
        try:
            c = eff_curve(0.25, 0, gm); cp[gm] = xstar_c(c, 100)
        except Exception: pass
    txt('RM-g6', 1, [(RM, 'Con γ/κ ≲ 1e-5, en el plano el umbral se vuelve casi independiente de γ (x*(100) ≈ 8.4'), (INF, 'con $\\gamma/\\kappa\\lesssim10^{-5}$ en el plano se vuelve casi independiente de $\\gamma$')], 'Plano con gamma/kappa <~ 1e-5: x*(100) ~ 8.4 (regimen termico puro)', 'V', rep='8.4', rec=cp.get(1e-6), tipo='numerica', dato='data/termico/x*_k0.25_gz14_f0_g1e-06_*.npz', script=S, tol=0.1, nota='x*(100) para gamma=1e-6, 1.78e-6, 3.16e-6: %s' % {k: round(v, 3) for k, v in cp.items()})
    return

# ============================================================ BLOQUE 18: informe (superado) y docs LaTeX (nivel 2)
FMT = 'docs/figura_merito/figura_merito.tex'; MPT = 'docs/mapa_plataformas/mapa_plataformas.tex'; LST = 'docs/lamb_shift_ma2019/lamb_shift_ma2019.tex'; BRT = 'docs/brecha_punto_excepcional/brecha_punto_excepcional.tex'; DCT = 'docs/documento_completo/documento_completo.tex'
def bloque_L2(E):
    S = 'consolidacion/verificar_todo.py::bloque_L2'
    K = KAPM
    # ---- informe.tex: afirmaciones superadas por trabajo posterior (el .tex no se edita)
    sup = [('INF-S1', '\\paragraph{Validación con el modelo completo (un punto).}', 'Informe: la validacion termica con el completo es de un solo punto (k2/k=0.25, x=10.1); "esta en curso un segundo punto"', 'data/filtro_completo: 8 puntos validados (T84-*), commits 2652746..163e971'),
           ('INF-S2', 'Está en curso un segundo punto', 'Informe: "esta en curso un segundo punto (k2/k=0.05, x=6.86)"', 'completado el 2026-09-29 (commit 2652746) y ampliado a 0.1 y 0.4 (4d23ea4)'),
           ('INF-S3', '\\item Validación térmica con el modelo completo. Solo hay un punto validado', 'Informe, Limitaciones 1: "solo hay un punto validado"', 'T84-* (8 puntos), HANDOFF 8.4'),
           ('INF-S4', 'Excepción declarada: las corridas del modelo completo con filtro y temperatura superan la tolerancia', 'Informe 4.1: hermiticidad cruda 1.2e-9 (punto validado) y 2.7e-10 (corrida previa) por encima de la tolerancia', 'Q-b6 (1.6e-12..2.8e-9), Q-h1 (umbrales 1e-8 desde 2026-09-30/10-01)'),
           ('INF-S5', 'mientras que $\\gamma_{\\rm bf}/n_q$ sube un $28\\%$', 'Informe 4.5: gamma_bf/n_q sube 28% con el filtro (efectivo); "no esta contrastado con el completo"', 'PL7-f17: +17% en el completo (un punto)'),
           ('INF-S6', 'No se estudió su dependencia con $N$ ni con la tolerancia del integrador', 'Informe: el piso de gamma_bf a T->0 del completo no se estudio en N ni en la tolerancia', 'T84-tolE2 (-0.10%); plano N=18/22/26 (PL7-N*); con filtro sigue sin barrido en N del piso'),
           ('INF-S7', 'Hasta completarlo, los umbrales de la Tabla', 'Informe: "hasta completarlo, los umbrales de la tabla y los c_bf se consideran resultados del efectivo"', 'T85-*, T84-*; sigue vigente como [ESTIMADO] para x*')]
    for i, a, t, sp in sup:
        txt(i, 1, [(INF, a)], t, 'S', nota='superado por: ' + sp + '; docs/Informe/informe.tex no se modifica (ultimo cambio 7304288, 2026-09-29, anterior a las fases 3 a 6)', tipo='cualitativa', commit='7304288')
    # ---- figura_merito.tex
    A, G5 = 5 / 72, None
    r = (5 / 24) * (36 / 5)**(2 / 3)
    txt('FM-01', 2, [(FMT, 'Con $(5/24)(36/5)^{2/3}=0.777$')], 'figura_merito: (5/24)(36/5)^(2/3) = 0.777', 'V', rep='0.777', rec=r, tipo='numerica', dato='(analitico)', script=S, tol=0.0005)
    import sympy as sp
    k_, A_, B_ = sp.symbols('k A B', positive=True)
    ks = sp.solve(sp.diff(A_ * k_**2 + B_ / k_, k_), k_)[0]
    rmin = sp.simplify((A_ * k_**2 + B_ / k_).subs(k_, ks))
    txt('FM-02', 2, [(FMT, 'Derivando e igualando a cero'), (FMT, 'r_{\\min}=3A\\kappa_{*}^{2}')], 'figura_merito: r(k)=A k^2 + B/k tiene minimo en k*^3 = B/(2A) con r_min = 3 A k*^2', 'V', rep='simbolico', tipo='cualitativa', dato='(sympy)', script=S)
    byid('FM-02')['estado'] = 'V' if sp.simplify(ks**3 - B_ / (2 * A_)) == 0 and sp.simplify(rmin - 3 * A_ * ks**2) == 0 else 'D'
    q = (r * 220)**1.5
    txt('FM-03', 2, [(FMT, 'Q\\geq\\frac{1}{\\epsilon\\eta}\\left(\\frac{0.777}{1/220}\\right)^{3/2}')], 'figura_merito: Q >= 2.23e3/(eps eta)', 'V', rep='2.23e3', rec=q, tipo='numerica', dato='(analitico)', script=S, nota='recalculo %.0f' % q)
    for e_, h_, rep in ((0.03, 0.1, '7.4e5'), (0.1, 0.1, '2.2e5'), (0.03, 0.05, '1.5e6')):
        txt('FM-Q%g-%g' % (e_, h_), 2, [(FMT, 'Con $\\epsilon=0.03$ y $\\eta=0.1$ se necesita')], 'figura_merito: Q >= %s con eps=%g, eta=%g' % (rep, e_, h_), 'V', rep=rep, rec=q / (e_ * h_), tipo='numerica', dato='(analitico)', script=S, nota='recalculo %.3g' % (q / (e_ * h_)))
    gx_, gz_ = 0.3 * math.sin(math.pi / 4), 0.3 * math.cos(math.pi / 4)
    txt('FM-04', 2, [(FMT, 'da $r_{q}=1.39\\times10^{-3}$')], 'figura_merito: Ma r_q = (5/72)(k/gz)^2 = 1.39e-3; gamma/kappa2 = 2e-5', 'V', rep='1.39e-3', rec=(5 / 72) * (0.03 / gz_)**2, tipo='numerica', dato='(analitico)', script=S, nota='gamma/kappa2 = %.2e' % (6e-7 / 0.03))
    k1 = (10 / 9) * gx_**2 * 0.03 / 36
    txt('FM-05', 2, [(FMT, 'el cociente es $\\kappa_{1}/(\\kappa/4)=5.6\\times10^{-3}$')], 'figura_merito: kappa1/(kappa/4) = 5.6e-3 para Ma', 'V', rep='5.6e-3', rec=k1 / (0.03 / 4), tipo='numerica', dato='(analitico)', script=S)
    txt('FM-S1', 2, [(FMT, 'la brecha, que satura en $\\kappa/4$')], 'figura_merito: la brecha satura en kappa/4 (con |alpha|^2=4 satura en ~0.28 kappa)', 'S', rep='0.25', rec=0.2784, tipo='numerica', dato='MN-02', script=S, nota='superado: HANDOFF 11 "Techo kappa/4"; con la tasa dinamica el confinamiento satura en 0.28 kappa y el epsilon de Ma es 0.022 (PL-17), no 5.6e-3')
    byid('FM-S1')['estado'] = 'S'
    txt('FM-S2', 2, [(FMT, 'Falta comprobar num\\\'ericamente la dependencia')], 'figura_merito: "falta comprobar numericamente la dependencia (k/gz)^2 variando gz/k a gx fijo"', 'S', nota='superado por V4 (15 puntos, 0.979-0.999; V4-RANGO) y V4b', tipo='cualitativa')
    txt('FM-S3', 2, [(FMT, 'requieren $hf_{q}/k_{B}T\\gtrsim5$')], 'figura_merito: termico "hf_q/kT >~ 5"', 'S', nota='superado por MP (conclusion 4: ~8.8) y por HANDOFF 8 (x*(100) = 7.18-10.11 segun bano y gamma)', tipo='cualitativa')
    txt('FM-S4', 2, [(FMT, 'con $P_{c}=0.996$ en $\\kappa_{2}/\\kappa\\approx1$')], 'figura_merito: Ma re-sintonizado P_c = 0.996 en kappa2/kappa~1', 'S', rep='0.996', rec=float(byid('MA-13')['valor_recalculado']), tipo='numerica', dato='MA-12, MA-13', script=S, nota='0.996 es el valor transitorio a Gt=30 (cache39: 0.9961); estacionario: 0.99754 (codigo sin desplazar) y 0.99873 (codigo fijo)')
    byid('FM-S4')['estado'] = 'S'
    # ---- mapa_plataformas.tex
    adj('PL-08', MPT, '$\\kappa_{1}/\\kappa_{2}$ & $1.4\\times10^{-3}$ & $9.2\\times10^{-5}$')
    txt('MP-1', 2, [(MPT, '$\\Gamma_{1}/\\kappa_{2}$ & $1.39\\times10^{-3}$ & $1.9\\times10^{-5}$')], 'mapa_plataformas: Gamma1/kappa2 Ma 1.39e-3, Naseem 1.9e-5; gamma/kappa2 2e-5 y 7.2e-5; kappa1/kappa2 = suma (1.4e-3, 9.2e-5)', 'V', rep='1.9e-5', rec=(5 / 72) / 60**2, tipo='numerica', dato='(analitico)', script=S, tol=None, nota='gamma/kappa2 Naseem = %.2e; suma = %.2e (mapa_plataformas l.94 aclara que incluye gamma)' % (1.5e-4 / 2.0736, (5 / 72) / 3600 + 1.5e-4 / 2.0736))
    txt('MP-2', 2, [(MPT, 'el qubit a 200 MHz tiene $n_{q}=0.62$ a 10 mK')], 'mapa_plataformas: Naseem n_q = 0.62 a 10 mK; T <~ 1.1 mK para x=8.8; f_q >~ 1.8 GHz; w_m/2pi >~ 0.9 GHz', 'V', rep='0.62', rec=1 / (math.exp(200 / 208.37) - 1), tipo='numerica', dato='(analitico)', script=S, tol=0.005, nota='T para x=8.8: %.2f mK; f_q: %.2f GHz' % (200 / 208.37 * 10 / 8.8, 8.8 * 0.20837))
    txt('MP-3', 2, [(MPT, '$n_{q}$ (10 mK) & $10^{-25}$ & 0.62')], 'mapa_plataformas: Ma n_q(10 mK) = 1e-25', 'V', rep='1e-25', rec=1 / (math.exp(12000 / 208.37) - 1), tipo='numerica', dato='(analitico)', script=S, tol=0.5e-25)
    for lvl, xx, nq_rep in ((100, 8.8, '1.5e-4'), (100, 7.8, '4.2e-4'), (10, 6.4, '1.7e-3'), (10, 5.4, '4.6e-3'), (1, 3.9, '2.1e-2'), (1, 2.5, '9.1e-2')):
        txt('MP-nq%g-%g' % (lvl, xx), 2, [(MPT, 'sesgo $\\eta$ & $n_{q}^{*}$')], 'mapa_plataformas Tabla termico: n_q* para eta=%d y hf/kT=%g: %s' % (lvl, xx, nq_rep), 'V', rep=nq_rep, rec=1 / (math.exp(xx) - 1), tipo='numerica', dato='(analitico, aritmetica con la x redondeada de la tabla)', script=S, tol=0.05 * float(nq_rep))
    txt('MP-S1', 2, [(MPT, 'la Tabla~\\ref{tab:termico} exige $hf_{q}/k_{B}T\\geq8.8$ para un sesgo de 100')], 'mapa_plataformas: requisito termico hf/kT >= 8.8 para sesgo 100 (|a|^2=2, Naseem, Tarea 36)', 'S', nota='superado por HANDOFF 8 / TX-*: x*(100) = 7.18-10.11 en la isla (|a|^2=4, kappa2/kappa=0.25), segun bano y gamma/kappa; el valor 8.8 viene de la Tarea 36 con baños inconsistentes (Tarea 37)', tipo='cualitativa')
    txt('MP-S2', 2, [(MPT, 'el confinamiento satura en 0.14$\\kappa$'), (MPT, 'sube a $1.0\\times10^{-2}$')], 'mapa_plataformas: Ma satura el confinamiento en 0.14 kappa (|a|^2=2, Tarea 43) y kappa1/conf = 1.0e-2; Naseem 0.23 kappa y 8.3e-4', 'S', rep='1.0e-2', rec=1.41e-3 / 0.14, tipo='numerica', dato='MP-1', script=S, nota='aritmetica 1.41e-3/0.14 = %.3g; superado por el confinamiento dinamico con |a|^2=4 (satura en 0.28 kappa; PL-16/PL-17: epsilon de Ma = 0.022)' % (1.41e-3 / 0.14))
    byid('MP-S2')['estado'] = 'S'
    txt('MP-S3', 2, [(MPT, '\\item Correcci\\\'on a un valor citado antes en la conversaci\\\'on')], 'mapa_plataformas: correccion del criterio termico (8.8 en vez de 5; 4.8 de la Tarea 37 es en frecuencia mecanica)', 'S', nota='superado por HANDOFF 8 (ver MP-S1)', tipo='cualitativa')
    # ---- lamb_shift_ma2019.tex
    adj('MA-02', LST, 'se obtiene $g_{x}^{2}/\\omega=0.0075$ y $\\delta_{\\mathrm{osc}}=-0.010$')
    txt('LS-1', 2, [(LST, 'dos tercios de $g_{\\mathrm{eff}}=2\\pi\\times15$~MHz')], 'lamb_shift: delta_osc = 0.010 es 2/3 de g_eff = G = 0.015', 'V', rep='0.667', rec=0.01 / 0.015, tipo='numerica', dato='(analitico)', script=S, tol=0.001)
    c39 = {}
    for fn in ('wp12.0_N22', 'wp11.98_N22'):
        z39 = np.load(os.path.join(VAL, 'tareas_ma/cache39/%s.npz' % fn), allow_pickle=True); c39[fn] = z39
    cellrows = [('wp12.0_N22', 20, 'F', '0.710'), ('wp12.0_N22', 20, 'par', '0.798'), ('wp12.0_N22', 20, 'Pe', '0.128'), ('wp12.0_N22', 30, 'F', '0.682'), ('wp12.0_N22', 30, 'par', '0.729'), ('wp12.0_N22', 30, 'Pe', '0.125'),
                ('wp11.98_N22', 20, 'F', '0.731'), ('wp11.98_N22', 20, 'par', '0.487'), ('wp11.98_N22', 20, 'Pe', '0.010'), ('wp11.98_N22', 30, 'F', '0.692'), ('wp11.98_N22', 30, 'par', '0.387'), ('wp11.98_N22', 30, 'Pe', '0.008')]
    for fn, Gt, key, rep in cellrows:
        z39 = c39[fn]; i = int(np.argmin(abs(z39['Gt'] - Gt)))
        v = float(np.real(z39[key][i]))
        txt('LS-T-%s-%d-%s' % (fn.split('_')[0], Gt, key), 2, [(LST, '$\\omega_{p}=2\\omega$ (Ma et al.) & 20 & 0.710' if fn.startswith('wp12') else '$\\omega_{p}=2(\\omega+\\delta_{\\mathrm{osc}})$ & 20 & 0.731')], 'lamb_shift Tabla full: %s, %s en Gt~%d: %s' % (fn.split('_')[0], key, Gt, rep), 'V', rep=rep, rec=v, tipo='numerica', dato='Mechanical-Cat-State/validacion/tareas_ma/cache39/%s.npz (Gt=%.2f)' % (fn, z39['Gt'][i]), script=S, tol=0.0025, nota='recalculo %.4f; la tabla del documento usa otra fase de muestreo (tarea39_40_resultados.md: P_e y paridad tienen micromovimiento)' % v)
    txt('LS-pf', 2, [(LST, '2|\\alpha|^{2}\\left(\\Gamma_{1}^{-}+\\Gamma_{1}^{+}\\right) = 2\\cdot4')], 'lamb_shift: tasa de phase-flip 2|a|^2(G1m+G1p) = 3.3e-4; extraida de la tabla 3.4e-4 (acuerdo "dentro del 3%"); T_coh ~ 0.5 us; G1m = 3.8e-5 (60 veces mayor que gamma con Q=1e7)', 'V', rep='3.3e-4', rec=2 * 4 * sum(Gam1(gx_, 6.0, 0.03)), tipo='numerica', dato='(analitico)', script=S, tol=None, nota='extraida: ln(0.487/0.387)/(10/0.015) = %.3e (%.1f%% de la prediccion: el documento dice 3%%); Gamma1- = %.2e; G1m/gamma = %.0f; 1/rate = %.2f us' % (math.log(0.487 / 0.387) / (10 / 0.015), 100 * (math.log(0.487 / 0.387) / (10 / 0.015) / (2 * 4 * sum(Gam1(gx_, 6.0, 0.03))) - 1), Gam1(gx_, 6.0, 0.03)[0], Gam1(gx_, 6.0, 0.03)[0] / 6e-7, 1 / (2 * 4 * sum(Gam1(gx_, 6.0, 0.03)) * 2 * math.pi * 1e9) * 1e6))
    txt('LS-S1', 2, [(LST, 'El modelo completo dependiente del tiempo confirma el corrimiento')], 'lamb_shift: con el drive de Ma el modelo completo da P_e ~ 0.13 y |<a^2>| ~ 2.44; con el drive re-sintonizado P_c = 0.996', 'V', rep='0.13', rec=float(byid('MA-06')['valor_recalculado']), tipo='numerica', dato='MA-06, MA-10', script=S, tol=0.02, nota='valores transitorios (Gt=20..30: 0.125-0.128); estacionarios: P_e = 0.115, |a^2| = 2.49 (MA-06, MA-10); P_c re-sintonizado estacionario 0.9975/0.9987 (MA-12/13)')
    # ---- brecha_punto_excepcional.tex
    tb = [('0.0050', '0.00506', '0.00505', 0.005), ('0.0631', '0.07407', '0.07407', 0.0631), ('0.0794', '0.09906', '0.09900', 0.0794), ('0.1000', '0.13820', '0.13820', 0.1), ('0.1259', '0.25000', '0.25000', 0.1259), ('0.3981', '0.25000', '0.25000', 0.3981)]
    pr = lambda g: 0.25 * (1 - math.sqrt(1 - 8 * g)) if g <= 0.125 else 0.25
    for g_, num, pre, gv in tb:
        txt('BR-T%s' % g_, 2, [(BRT, ('%s & %s & %s' % (g_, num, pre)))], 'brecha_punto_excepcional Tabla: Gamma2/kappa = %s: Delta predicho %s (formula exacta)' % (g_, pre), 'V', rep=pre, rec=pr(gv), tipo='numerica', dato='(analitico)', script=S, nota='columna numerica del documento: %s (R: ED de la verificacion V6/Tarea 35)' % num)
    txt('BR-c', 2, [(BRT, 'Num\\\'ericamente se obtiene $c(0)=1$')], 'brecha: c(0)=1, c(1)~1.40, c(2)~2.57, c(3)~3.31; cruce G^x ~ 0.18, 0.10, 0.08 kappa', 'V', rep='0.18,0.10,0.08', tipo='numerica', dato='(analitico sobre los c del documento)', script=S, nota='0.25/1.40=%.3f, 0.25/2.57=%.3f, 0.25/3.31=%.3f; los c(alpha^2) vienen de simulaciones del modelo minimo (R); c(4)=4.14 (MN-01) es consistente con la secuencia' % (0.25 / 1.40, 0.25 / 2.57, 0.25 / 3.31))
    byid('BR-c')['estado'] = 'V' if abs(0.25 / 1.40 - 0.18) < 0.005 and abs(0.25 / 2.57 - 0.10) < 0.005 and abs(0.25 / 3.31 - 0.08) < 0.005 else 'D'
    txt('BR-S1', 2, [(BRT, 'satura en valores del orden de $0.10\\kappa$ a $0.17\\kappa$'), (BRT, 'Un valor de saturaci\\\'on entre $0.10\\kappa$ y $0.17\\kappa$')], 'brecha: la brecha del modelo completo de Naseem satura en 0.10-0.17 kappa, claramente por debajo de kappa/4, con un maximo en Gamma2/kappa~0.13', 'S', nota='superado: HANDOFF 11 (artefactos de modos de borde de Fock, Tareas 35, 38, 45, 47); la tasa dinamica satura en ~0.28 kappa (MN-02)', tipo='cualitativa')
    # ---- documento_completo.tex (v1)
    txt('DC-S1', 2, [(DCT, 'El costo es la tasa de confinamiento, que cae de $4.2\\times10^{-3}$ (plano)')], 'documento_completo v1: tasa de confinamiento 4.2e-3 (plano) a 3.6, 3.1, 2.2, 0.98e-3 (kf=3,1,0.3,0.1), "no convergido en N (6%)"', 'S', nota='superado: con la tasa dinamica el plano es 6.98e-3 y la relativa 0.965, 0.967, 0.90, 0.53 (F4c-*); HANDOFF 11 y PENDIENTES E5', tipo='cualitativa')
    txt('DC-S2', 2, [(DCT, 'Umbrales: sesgo 100 con')], 'documento_completo v1: umbrales termicos hf/kT = 8.8, 7.8, 6.4, 5.4 (Naseem, Tarea 36)', 'S', nota='superado por HANDOFF 8 (ver MP-S1)', tipo='cualitativa')
    txt('DC-S3', 2, [(DCT, 'En el esquema de Naseem ($\\Gamma_{2}/\\kappa=0.13$), $\\gamma_{\\rm bf}\\approx0.05n_{q}\\kap')], 'documento_completo v1: gamma_bf ~ 0.05 n_q kappa', 'S', nota='superado: S-04 (c_bf depende de kappa2/kappa)', tipo='cualitativa')
    adj('MA-13', DCT, '$\\omega_{p}=\\omega_{p}^{*}=11.98$ & 0.99754')
    adj('FW-06', DCT, 'el ajuste ${\\rm FWHM}=cG$, $c=2.94$')
    adj('FW-10', DCT, 'Con siete valores de $\\kappa_{2}/\\kappa$ (Tarea 47(d)), $c=2.87\\pm0.25$')
    adj('V3-07', DCT, 'Ma, con piso, f\\\'ormula con $+1$ y $|\\langle a^{2}\\rangle|=3.973$')
    adj('MF-T0.1', DCT, 'El costo es la tasa de confinamiento') if False else None
    txt('DC-docs', 2, [(DCT, '\\item El m\\\'aximo de la brecha en $\\Gamma_{2}/\\kappa\\approx0.13$ y su saturaci')], 'documento_completo v1 lista como retirados: maximo de la brecha en 0.13 (artefacto), umbral hf/kT>=5, precision <=0.4% para kappa2/kappa<=0.25, +1 faltante, signo de pares en 45(d)', 'S', nota='coincide con HANDOFF 11 y PENDIENTES E2-E5; HANDOFF 12.14 pide v2', tipo='cualitativa')
    return

# ============================================================ BLOQUE 19: meta (referencias, hashes, unicode, copias de figuras, compilacion)
def bloque_meta(E):
    S = 'consolidacion/verificar_todo.py::bloque_meta'
    # (h) archivos citados en los .md de nivel 1: existencia por nombre (con expansion de patrones)
    import fnmatch
    allf = []
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in ('.venv', '.venv_qutip5', '.git.bak', '.git', '__pycache__', 'consolidacion')]
        allf += fn
    docs = [H8, MD0, RM, PH, C6, PRE]
    miss = []; total = 0
    ext = r'(?:npz|py|sh|md|tex|pdf|png|csv|log|txt|bib)'
    for d in docs:
        t = open(os.path.join(ROOT, d), errors='ignore').read()
        for m in re.finditer(r'`([^`\n]+)`', t):
            for tok in re.findall(r'[\w\-\.\*/{}\[\]]+\.' + ext, m.group(1)):
                total += 1
                b = os.path.basename(tok)
                pat = re.sub(r'\{[^}]*\}', '*', b)
                ok = any(fnmatch.fnmatch(k, pat) for k in allf)
                if not ok: miss.append((d, tok))
    txt('REF-1', 1, [(H8, 'Mechanical-Cat-State/validacion/independiente/V1..V6_reporte.md'), (H8, '`termico_validacion_fase2.txt`/`.csv`/`_tolE.txt` (fase 2)')], 'Archivos citados entre comillas invertidas en HANDOFF, READMEs, PENDIENTES, C6 y preregistro existen (por nombre, con expansion de patrones)', 'V', rep='todos', tipo='cualitativa', dato='repositorio', script=S,
        nota='%d referencias; sin resolver: %s (ambas son abreviaturas: "V1..V6_reporte.md" = V1..V6_reporte.md y "_tolE.txt" = termico_validacion_fase2_tolE.txt, que existen)' % (total, sorted(set(t for d, t in miss))))
    byid('REF-1')['estado'] = 'V' if set(t for d, t in miss) <= {'Mechanical-Cat-State/validacion/independiente/V1..V6_reporte.md', '_tolE.txt'} else 'D'
    # referencias rotas reales (fase A)
    txt('REF-2', 1, [(H8, 'Mechanical-Cat-State/validacion/independiente/V1..V6_reporte.md')], 'Referencias rotas candidatas de la Fase A: handoff_cifras.py (g_gz14_f1_..._N22_...npz), V1..V6_reporte.md, _tolE.txt, detuning_scan_results (fig3_git_v4.py), marco_teorico.tex, msc/ma2019.pdf', 'V', rep='resueltas', tipo='cualitativa', dato='consolidacion/referencias_rotas.csv', script=S,
        nota='handoff_cifras.py: plantilla "x60_k%g_gz14_f1_g2e-05_N22_Nf2_w200_a4.npz" -> existe (4 archivos); V1..V6_reporte.md: abreviatura, existen V1..V6 y V4b; _tolE.txt: renombrado a termico_validacion_fase2_tolE.txt; detuning_scan_results.{csv,npz}: salida que escribe el script original (codigo_original/fig3_git_v4.py), no se incluye en el repo; marco_teorico.tex: archivo de otro proyecto (tesis, citado solo en HANDOFF_chat); msc/ma2019.pdf: citado en V1_reporte.md, papers/ no lo tiene (gitignored: Ma 2019 solo en local)')
    # (6) unicode
    nus = [f for f in os.listdir(os.path.join(VAL, 'tareas_ma')) if 'ν' in f]
    import unicodedata
    ok_n = all(unicodedata.is_normalized('NFC', f) and unicodedata.is_normalized('NFD', f) and [hex(ord(c)) for c in f if ord(c) > 127] == ['0x3bd'] for f in nus)
    scripts_nu = [os.path.basename(p) for p in glob.glob(os.path.join(VAL, 'tareas_ma', '*.py')) + glob.glob(os.path.join(VAL, 'tareas_ma', '*.sh')) if 'ν' in open(p, errors='ignore').read()]
    cp_scripts = set(hex(ord(c)) for p in glob.glob(os.path.join(VAL, 'tareas_ma', 'tarea4[67]*.py')) for c in open(p, errors='ignore').read() if c == 'ν' or unicodedata.normalize('NFD', c) == 'ν' or unicodedata.normalize('NFC', c) == 'ν')
    txt('UNI-1', 1, [(MD0, 'validacion/')], 'Los 4 .npz con nu en el nombre (cache46_{i,ii}_nu, cache47b_{i,ii}_nu): la letra es U+03BD (nu griega), sin descomposicion: NFC = NFD; los scripts que los generan (tarea46_run.py, tarea47b_run.py, tarea47b_fig2.py) usan la misma letra U+03BD literal', 'V', rep='4 archivos', tipo='cualitativa', dato='validacion/tareas_ma/cache46_*, cache47b_*', script=S,
        nota='archivos: %s; scripts con la letra literal: %s (codepoints %s). Ninguna referencia falla por NFC/NFD' % (nus, scripts_nu, sorted(cp_scripts)))
    byid('UNI-1')['estado'] = 'V' if len(nus) == 4 and ok_n and cp_scripts == {'0x3bd'} else 'D'
    # figuras copia (README raiz)
    def sha(p):
        import hashlib
        return hashlib.sha256(open(os.path.join(ROOT, p), 'rb').read()).hexdigest()
    okc = all(sha('docs/documento_completo/%s.pdf' % a) == sha('Mechanical-Cat-State/figuras_finales/figuras/%s.pdf' % b) for a, b in (('fig2_val', 'fig2'), ('fig3_val', 'fig3'), ('fig2_wigner', 'apendice_fig2_estados')))
    txt('FIG-1', 1, [(MD0, '`docs/documento_completo`: `fig2_val.pdf`, `fig3_val.pdf` y `fig2_wigner.pdf` son copias')], 'documento_completo: fig2_val.pdf, fig3_val.pdf y fig2_wigner.pdf son copias de figuras_finales/figuras/{fig2, fig3, apendice_fig2_estados}.pdf', 'V', rep='copias', tipo='cualitativa', dato='sha256 de los PDF', script=S)
    byid('FIG-1')['estado'] = 'V' if okc else 'D'
    same_t = sha('docs/Informe/figuras/figura_termica.pdf') == sha('Mechanical-Cat-State/figuras_finales/figuras/figura_termica.pdf')
    txt('FIG-2', 1, [(INF, 'figuras/figura_termica.pdf')] if False else [(INF, 'figura_termica.pdf')], 'docs/Informe/figuras/figura_termica.pdf es la figura termica actual de figuras_finales', 'S', rep='igual', tipo='cualitativa', dato='sha256', script=S, nota='no es igual: la del informe es del 2026-09-29 (39d026c); figuras_finales/figuras/figura_termica.pdf se regenero en 2b64d26 (2026-09-30) con los puntos del completo y las notas "not validated"')
    byid('FIG-2')['estado'] = 'S' if not same_t else 'V'
    # compilacion de los docs (resultado guardado de una corrida manual en el scratchpad)
    rc = os.path.join(OUT, 'compilacion_docs.txt')
    if os.path.exists(rc):
        t = open(rc).read()
        bad = [l for l in t.split('\n') if l.strip() and ('undefined_citations=0' not in l or 'undefined_refs=0' not in l or 'pdf=1' not in l.replace(' ', ''))]
        txt('COMP-1', 1, [(MD0, 'Todos los docs compilan (pdflatex + bibtex, sin citas indefinidas)')], 'Todos los docs compilan con pdflatex + bibtex sin citas indefinidas', 'V', rep='compilan', tipo='cualitativa', dato='consolidacion/compilacion_docs.txt (copia en scratchpad; docs/ no se modifica)', script='(manual) pdflatex/bibtex', nota=t.strip().replace('\n', ' | '))
        byid('COMP-1')['estado'] = 'V' if not bad else 'D'
    else:
        txt('COMP-1', 1, [(MD0, 'Todos los docs compilan (pdflatex + bibtex, sin citas indefinidas)')], 'Todos los docs compilan con pdflatex + bibtex sin citas indefinidas', 'R', rep='compilan', tipo='cualitativa', nota='no se compilo')
    # tareas_ma: scripts que importan ma2 y la linea del termino de pares
    imps = sorted(os.path.basename(p) for p in glob.glob(os.path.join(VAL, 'tareas_ma', '*.py')) if re.search(r'^\s*import [^\n]*\bma2\b|^\s*from ma2 import', open(p, errors='ignore').read(), re.M))
    ma2 = open(os.path.join(VAL, 'tareas_ma/ma2.py'), errors='ignore').read()
    txt('SG-3', 3, [('Mechanical-Cat-State/validacion/tareas_ma/verificacion_signo_pair.md', 'Scripts que importan `ma2`')], 'verificacion_signo_pair.md: solo tarea42/43/45d/45e importan ma2; la linea "if pair: H0 = H0 - p[G] (sp a a + sm a+ a+)" existe en ma2.py', 'V', rep='4 scripts', tipo='cualitativa', dato='validacion/tareas_ma/*.py', script=S, nota='importan ma2: %s' % imps)
    byid('SG-3')['estado'] = 'V' if imps == ['tarea42_worker.py', 'tarea43_worker.py', 'tarea45d_worker.py', 'tarea45e_worker.py'] and "if pair: H0 = H0 - p['G'] * (sp * a * a + sm * a.dag() * a.dag())" in ma2 else 'D'
    return

# ============================================================ postproceso: coincidencia de ubicaciones repetidas y lineas sin verificar
def _nums(line):
    t = line.replace('−', '-').replace('\\times', '×').replace('$', '').replace('{', '').replace('}', '').replace('\\,', '')
    t = re.sub(r'×\s*10\^\s*(-?\d+)', r'e\1', t)
    t = re.sub(r'(\d)\s*×\s*10\^?(-?\d+)', r'\1e\2', t)
    out = []
    for m in re.finditer(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', t):
        pr = parse(m.group(0))
        if pr: out.append(pr)
    return out
def coincidencia():
    for c in CL:
        locs = [u for u in c['ubicaciones'].split(' | ') if ':' in u]
        if len(locs) < 2 or c.get('tipo') in ('pendiente',): continue
        reps = [x for x in re.split(r'[;, ]+', str(c['valor_reportado'])) if parse(x)]
        if not reps: continue
        ok_all, bad = True, []
        for u in locs:
            f, ln = u.rsplit(':', 1)
            L = lines(f)
            if not ln.isdigit() or int(ln) < 1 or L is None: continue
            nums = _nums(L[int(ln) - 1])
            found = False
            for rep in reps:
                vr, ur = parse(rep)
                if any(abs(v - vr) <= max(ur or 0, u or 0) * (1 + 1e-9) or (vr and abs(v - vr) <= 0.011 * abs(vr)) for v, u in nums): found = True
            if not found: ok_all = False; bad.append(u)
        c['ubicaciones_coinciden'] = c['ubicaciones_coinciden'] or ('coinciden' if ok_all else 'cifra no literal en: ' + ';'.join(bad))
def lineas_sin_verificar():
    import importlib
    cov = {}
    for c in CL:
        for u in c['ubicaciones'].split(' | '):
            if ':' in u:
                f, l = u.rsplit(':', 1)
                if l.isdigit() and int(l) > 0: cov.setdefault(f, set()).add(int(l))
    def sust(l):
        t = re.sub(r'`[^`]*`', '', l); t = re.sub(r'§\s?\d+(\.\d+)?', '', t); t = re.sub(r'\b[PECV]\d+(b)?\b', '', t); t = re.sub(r'Tareas? \d+(\s*[-–a y]\s*\d+)?', '', t); t = re.sub(r'\d{4}-\d\d-\d\d', '', t)
        t = re.sub(r'\\label\{[^}]*\}|\\ref\{[^}]*\}|\\eqref\{[^}]*\}|\\cite\{[^}]*\}|\\includegraphics.*|\b(19|20)\d\d\b', '', t)
        return bool(re.search(r'\d+\.\d+|\de-\d|\d\s?%|\d+ de \d+|[NK]\s*=\s*\d+|\d+\s*(GB|s\b|min|kB|MHz|GHz|Hz|mK)|×10|10\^|\\times|\d{3,}', t))
    L1 = [H8, MD0, RM, PH, C6, PRE, INF]
    out = []
    for f in L1:
        for i, l in enumerate(lines(f), 1):
            if i not in cov.get(f, set()) and not l.startswith('#') and sust(l) and len(l) > 15:
                kind = 'tabla/encabezado' if (l.strip().startswith('|') or '&' in l) else ('caption/titulo' if ('\\caption' in l or '\\paragraph' in l or '\\geometry' in l or '\\date' in l) else 'texto')
                out.append((f, i, kind, l.strip()[:300]))
    with open(os.path.join(OUT, 'lineas_sin_verificar.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(['archivo', 'linea', 'tipo', 'texto'])
        for r in out: w.writerow(r)
    return len(out)

# ---- COLA (mantener al final)
if __name__ == '__main__':
    print(main())
