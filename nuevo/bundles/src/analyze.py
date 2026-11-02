"""Tablas de comparacion analitico-exacto a partir de data/*.npz. Escribe data/analysis.txt (se cita en informe_WP1.md)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from bundles_core import *

D = os.path.join(os.path.dirname(__file__), '..', 'data')
out = []
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)
def st(rel):
    rel = np.abs(rel).ravel()
    return f'max {rel.max():.3f} medio {rel.mean():.3f} mediana {np.median(rel):.3f} | <5%: {np.mean(rel < .05):.2f} <10%: {np.mean(rel < .10):.2f}'

# ---------------- D2 ----------------
p('## D2: cerrada vs exacta (episodios), malla 20x20 Om in [0.02,0.3], kap in [3e-4,1e-2], gam=2e-4, gph=4e-4')
for n in (2, 3):
    for lam in (0.1, 0.14, 0.2):
        d = np.load(f'{D}/D2_n{n}_lam{lam}.npz'); Pe, Pc, Pcep, Je, Jc, xi = (d[k] for k in ('P_exact', 'P_closed', 'P_closed_ep', 'J_exact', 'J_closed', 'xi')); Pv1 = d['P_closed_v1']
        val = xi < 0.5   # regimen de tasas: Om_n/gamma_c(0) < 0.5
        p(f'n={n} lam={lam}  [toda la malla, {Pe.size} pts]')
        p('   P cerrada v2 (floor e^-l2)', st((Pc - Pe) / Pe)); p('   P cerrada v2 (floor 1/(2-e^-l2))', st((Pcep - Pe) / Pe)); p('   P cerrada v1 (scout, sin termino FC a mitad de cascada)', st((Pv1 - Pe) / Pe), '| 1-P', st(((1 - Pv1) - (1 - Pe)) / (1 - Pe)))
        p('   impureza 1-P (e^-l2) ', st(((1 - Pc) - (1 - Pe)) / (1 - Pe)), '| con floor ep:', st(((1 - Pcep) - (1 - Pe)) / (1 - Pe)))
        p('   flujo                ', st((Jc - Je) / Je))
        r = np.load(f'{D}/D2b_rate_n{n}_lam{lam}.npz'); Pr, Jr = r['P_rate'], r['J_rate']
        p('   [modelo de tasas lorentziano] P', st((Pr - Pe) / Pe), '| 1-P', st(((1 - Pr) - (1 - Pe)) / (1 - Pe)), '| J', st((Jr - Je) / Je))
        p('   sesgo medio con signo: closed-exact en P: %+.4f ; rate-exact: %+.4f' % ((Pc - Pe).mean(), (Pr - Pe).mean()))
        if val.sum() > 0:
            p(f'   [region de tasas xi<0.5: {val.sum()} pts]  P', st(((Pc - Pe) / Pe)[val]), '| 1-P', st((((1 - Pc) - (1 - Pe)) / (1 - Pe))[val]), '| J', st(((Jc - Je) / Je)[val]))
        p(f'   [region de tasas xi<0.5, modelo de tasas] P', st(((Pr - Pe) / Pe)[val]), '| 1-P', st((((1 - Pr) - (1 - Pe)) / (1 - Pe))[val]), '| J', st(((Jr - Je) / Je)[val]))
        w = (xi >= 0.5)
        if w.sum() > 0:
            p(f'   [Rabi coherente xi>=0.5: {w.sum()} pts] P', st(((Pc - Pe) / Pe)[w]), '| J', st(((Jc - Je) / Je)[w]))
        p('   P exacta rango', f'{Pe.min():.3f}-{Pe.max():.3f}', '| P cerrada rango', f'{Pc.min():.3f}-{Pc.max():.3f}')
# ---------------- D1 ----------------
p('\n## D1: posicion del pico de flujo (exacto) vs analitico')
d = np.load(f'{D}/D1_spectrum.npz'); m = json.loads(str(d['meta']))
for i, n in enumerate(d['pos_n']):
    sel = np.abs(d['dL'] - d['pos_dL'][i]) < 0.03; dl = d['dL'][sel]; J = d['J'][sel]
    j = np.argmax(J); gc = (n * m['kap'] + m['gam'] + m['gph']) / 2
    p(f'n={n}: pico exacto dL={dl[j]:+.5f} analitico {d["pos_dL"][i]:+.5f} (Bin-III+lam^2 {d["pos_dL_bin"][i]:+.5f}) dif={dl[j] - d["pos_dL"][i]:+.1e} = {(dl[j] - d["pos_dL"][i]) / gc:+.2f} gamma_c ; J_pico={J[j]:.3e} Pdom={d["Pdom"][sel][j]:.3f} (k={d["kdom"][sel][j]}) ; paso rejilla 5e-4')
    p(f'      lineshape lorentziano: J_pico={d["J_lorentz"][sel].max():.3e}')
# ---------------- D4 ----------------
p('\n## D4: maestra local vs vestida')
d = np.load(f'{D}/D4_local_vs_dressed.npz'); k = d['kap']
for n in (2, 3):
    dk = d[f'n{n}_local_kmean'] - d[f'n{n}_dressed_kmean']
    p(f'n={n}: exceso de fonones/ciclo (local-vestida) vs nube kap*lam^2/gam  [kap, medido, predicho, ratio]')
    for i in range(0, len(k), 3):
        p(f'   {k[i]:.4f} {dk[i]:.3f} {d[f"n{n}_cloud_pred"][i]:.3f} {dk[i] / d[f"n{n}_cloud_pred"][i]:.3f}   Pcyc dress {d[f"n{n}_dressed_cycle_P"][i]:.3f} loc {d[f"n{n}_local_cycle_P"][i]:.3f} pred {d[f"n{n}_Ploc_pred"][i]:.3f} | Pep dress {d[f"n{n}_dressed_episode_P"][i]:.3f} loc {d[f"n{n}_local_episode_P"][i]:.3f}')
    rel = (d[f'n{n}_Ploc_pred'] - d[f'n{n}_local_cycle_P']) / d[f'n{n}_local_cycle_P']
    p('   P_loc(ciclos) ~ P_dr e^{-cloud}: error rel ', st(rel))
# ---------------- D5 ----------------
p('\n## D5: cota FC')
d = np.load(f'{D}/D5_floor.npz')
for n in (2, 3):
    p(f'n={n}: lam | P_ep^max exacto (kap*, borde) | cerrada (kap*) | floor e^-l2 | floor ep | P_cyc^max exacto')
    for i, l in enumerate(d['lam']):
        p(f'   {l:.3f} | {d[f"n{n}_Pex"][i]:.4f} ({d[f"n{n}_kex"][i]:.4f},{d[f"n{n}_edge"][i]}) | {d[f"n{n}_Pcl"][i]:.4f} ({d[f"n{n}_kcl"][i]:.4f}) | {d["floor_exp"][i]:.4f} | {d["floor_ep"][i]:.4f} | {d[f"n{n}_Pcyc"][i]:.4f}')
    p('   error rel. impureza cerrada vs exacta', st(((1 - d[f'n{n}_Pcl']) - (1 - d[f'n{n}_Pex'])) / (1 - d[f'n{n}_Pex'])))
# ---------------- D3 ----------------
p('\n## D3: puntos del frente cerrado evaluados exactamente')
d = np.load(f'{D}/D3_pareto.npz')
for n in (2, 3):
    for lam in (0.1, 0.14, 0.2):
        key = f'n{n}_lam{lam}'
        p(key, 'Om,kap,Jcl,Pcl,Jex,Pex');
        for a in zip(d[f'{key}_sel_Om'], d[f'{key}_sel_kap'], d[f'{key}_sel_Jcl'], d[f'{key}_sel_Pcl'], d[f'{key}_sel_Jex'], d[f'{key}_sel_Pex']):
            p('   %.3f %.4f %.2e %.4f %.2e %.4f' % a)
# ---------------- C ----------------
if os.path.exists(f'{D}/C_mc.npz'):
    p('\n## C: MC (400 trayectorias, maestra vestida) vs exacta por episodios vs cerrada (v2, v1) vs prediccion operativa P_ep(1-q)(1-s)/(1+s)')
    d = np.load(f'{D}/C_mc.npz'); z = []
    for i in range(len(d['n'])):
        n, Om, kap, gam = int(d['n'][i]), d['Om'][i], d['kap'][i], d['gam'][i]
        Pv2 = P_closed(0.14, Om, kap, gam, 4e-4, n); Pv1 = P_closed(0.14, Om, kap, gam, 4e-4, n, fcmid=False)
        p(f'n={n} Om={Om} kap={kap} gam={gam:.0e} (kap/gam={kap / gam:.0f}) T={d["T"][i]:.1e} | P_ep={d["P_ep"][i]:.4f} P_cl(v2)={Pv2:.4f} P_cl(v1)={Pv1:.4f} | J mc {d["flux_mc"][i]:.3e}+-{d["flux_mc_err"][i]:.1e} ex {d["flux_exact"][i]:.3e} ({(d["flux_mc"][i] - d["flux_exact"][i]) / d["flux_mc_err"][i]:+.1f} sigma)')
        for j, tc in enumerate(('2/k', '5/k', '10/k')):
            zz = (d["P_pred"][i][j] - d["P_mc"][i][j]) / d["P_mc_err"][i][j]; z.append(zz)
            p(f'      tau_c={tc}: P_MC={d["P_mc"][i][j]:.4f}+-{d["P_mc_err"][i][j]:.4f} (nb={d["nb"][i][j]}) pred_op={d["P_pred"][i][j]:.4f} ({zz:+.1f} sigma) ; P_MC-P_ep={d["P_mc"][i][j] - d["P_ep"][i]:+.4f}')
    z = np.array(z); p(f'pred_op vs MC: |z| medio {np.mean(abs(z)):.1f}, max {abs(z).max():.1f}; diferencia absoluta media {np.mean(abs(d["P_pred"] - d["P_mc"])):.4f}, max {np.abs(d["P_pred"] - d["P_mc"]).max():.4f}')
    fz = (d['flux_mc'] - d['flux_exact']) / d['flux_mc_err']; p(f'flujo MC vs exacto: z medio {np.mean(abs(fz)):.2f}, max {abs(fz).max():.2f}; err rel medio {np.mean(abs(d["flux_mc"] / d["flux_exact"] - 1)):.3f}')
    # mejor tau_c por punto (el de mayor P_MC) y su diferencia con P_ep
    for i in range(len(d['n'])):
        j = int(np.argmax(d['P_mc'][i])); p(f'   n={int(d["n"][i])} Om={d["Om"][i]} kap={d["kap"][i]} gam={d["gam"][i]:.0e}: mejor tau_c={(2, 5, 10)[j]}/kap  P_MC={d["P_mc"][i][j]:.4f} vs P_ep {d["P_ep"][i]:.4f}  (dif {d["P_mc"][i][j] - d["P_ep"][i]:+.4f})')
if os.path.exists(f'{D}/C2_cycles_local.json'):
    p('\n## C2: ciclos MC vs exactos (vestida / local simulada en marco de laboratorio / local polaron) y agrupacion local')
    c = json.load(open(f'{D}/C2_cycles_local.json'))
    for r in c['rows']:
        mc, ex = np.array(r['cyc_mc'][:6]), np.array(r['cyc_exact'][:6]); err = np.sqrt(mc * (1 - mc) / r['ncyc'])
        p(f'{r["kind"]} kap={r["kap"]}: P(k=0..5) MC {np.round(mc, 4)} exacta {np.round(ex, 4)} ; max |z| {np.max(abs(mc - ex) / err):.1f} ; J mc/ex {r["flux_mc"]:.3e}/{r["flux_exact"]:.3e}')
        if r.get('P_mc') is not None and not (isinstance(r['P_mc'], float)): p(f'      clustering (2,5,10/kap): {np.round(r["P_mc"], 4)} ; P_ep exacta (clasificacion pre-salto) {r["P_ep_exact"]:.4f}')
# ---------------- D6 / D7 ----------------
d = np.load(f'{D}/D6_trajectories.npz')
p('\n## D6: trayectorias T=2e4')
for n in (1, 2, 3):
    cnt = [len(d[f'n{n}_phonon_t{k}']) for k in range(8)]; cph = [len(d[f'n{n}_photon_t{k}']) for k in range(8)]
    p(f'n={n}: eventos de fonon por trayectoria {cnt} (rep={int(d[f"n{n}_rep"])}), fotones {cph}; flujo esperado*T = {d[f"n{n}_flux_exact"] * 2e4:.2f}')
d = np.load(f'{D}/D7_dephasing.npz'); p('\n## D7: P_2(episodios) y J vs gph (gam=2e-4) y vs gam (gph=4e-4); exacto (cerrada)')
for i in range(0, len(d['gph']), 3): p(f'   gph={d["gph"][i]:.1e}: P={d["P_vs_gph"][i]:.4f} ({d["Pcl_vs_gph"][i]:.4f})  J={d["J_vs_gph"][i]:.3e} ({d["Jcl_vs_gph"][i]:.3e})')
for i in range(0, len(d['gam']), 3): p(f'   gam={d["gam"][i]:.1e}: P={d["P_vs_gam"][i]:.4f} ({d["Pcl_vs_gam"][i]:.4f})  J={d["J_vs_gam"][i]:.3e} ({d["Jcl_vs_gam"][i]:.3e})')
rel = (d['Pcl_map'] - d['P_map']) / d['P_map']; p('mapa 2D (gam,gph): error cerrada vs exacta en P', st(rel), '; impureza', st(((1 - d['Pcl_map']) - (1 - d['P_map'])) / (1 - d['P_map'])), '; flujo', st((d['Jcl_map'] - d['J_map']) / d['J_map']))
open(f'{D}/analysis.txt', 'w').write('\n'.join(out))
