import numpy as np, tarea46 as T
DS = 2.0; A2 = T.epsp * DS / 2 / abs(T.Gexch)      # alpha^2 = eps_p/g_eff con g_eff=|G_exch|
L = ["# Tarea 47(b) — Tarea 46 repetida con la convención correcta (coseno 2ε_p, deriva efectiva ε_p, α²=ε_p/g_eff=2.5)\n",
     f"Unidades 2π·MHz=1: ν={T.nu}, G={T.G}, g_x=g_z=√2G/4={T.gx:.4f}, ε_p={T.epsp}, Δ_m=ν/2={T.Dm}, γ={T.gam}, sin pérdida del magnón. N=20, deriva 2ε_p cos(ω_p t), integración directa (mesolve) en el marco de laboratorio; oscilador rotado a ω_p/2 para P_c y F.",
     f"Cat objetivo: |α|²=ε_p/g_eff (deriva efectiva ε_p = 2ε_p·½ por la RWA del coseno) con g_eff=|G_exch|=2g_xg_z/w={T.Gexch:.4f} ⇒ |α|²={A2:.3f}; fase de α = ½·arg⟨a²⟩ tardío (rotante). Re-sintonía: δ_osc=g_x²[ImS(w)+ImS(3w)]={T.delta_osc:.4f} (con κ²/4) ⇒ ω_p=2(ν/2+δ_osc)={2*(T.nu/2+T.delta_osc):.4f}.",
     "Disipación (ii): σ₋(bare)=((cosθ−1)/2)σ̃₊+((cosθ+1)/2)σ̃₋+(sinθ/2)σ̃_z (θ=π/4: coeficientes −0.1464, 0.8536, 0.3536).\n"]
gts = np.concatenate([[0], np.geomspace(0.5, 200, 60)]); wr = 2 * (T.nu / 2 + T.delta_osc); R = {}
for ver in ('i', 'ii'):
    for name, wp in (("ν", T.nu), ("re-sint.", wr)):
        ts, st = T.simulate(ver, wp, 20, gts, dscale=DS); r0 = T.analyze(ts, st, wp, 20, alpha=np.sqrt(A2))
        phase = np.angle(r0['a2'][-6:].mean()) / 2; alpha = np.sqrt(A2) * np.exp(1j * phase); r = T.analyze(ts, st, wp, 20, alpha=alpha); r['gts'] = gts; R[ver, name] = r
        np.savez(f"cache47b_{ver}_{name.replace('.', '').replace('-', '')}.npz", **{k: v for k, v in r.items()})
L += ["## Curvas (γt) — P_c, paridad, P_e (vestido), F (gato par), |⟨a²⟩|\n"]
sel = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
for (ver, name), r in R.items():
    L += [f"### Versión ({ver}) {'γD[σ̃₋] (la suya)' if ver=='i' else 'γD[σ₋] bare en base vestida (física)'}, ω_p={name}\n", "| γt | P_c | paridad | P_e | F | \\|⟨a²⟩\\| |", "|---|---|---|---|---|---|"]
    for i in sel: L.append(f"| {r['gts'][i]:.2f} | {r['Pc'][i]:.3f} | {r['par'][i]:+.3f} | {r['Pe'][i]:.3f} | {r['F'][i]:.3f} | {abs(r['a2'][i]):.3f} |")
    L.append(f"\nfase de α: {np.angle(r['alpha']):+.3f} rad; máx |⟨a²⟩|={abs(r['a2']).max():.3f} (analítico {A2:.3f}).\n")
# tasa de paridad
form = 2 * A2 * (T.G1m + T.G1p); form0 = 2 * A2 * (T.G1m0 + T.G1p0)
L += ["## Tasa de decaimiento de la paridad frente a 2|α|²(Γ₁₋+Γ₁₊)\n",
      f"Γ₁₋={T.G1m:.4f}, Γ₁₊={T.G1p:.4f} (con κ²/4; κ=γ=16, ω=ν/2={T.Dm}); sin κ²/4: {T.G1m0:.4f}, {T.G1p0:.4f}. Predicciones 2|α|²(Γ₁₋+Γ₁₊): **{form:.4f}** (con κ²/4) y {form0:.4f} (sin κ²/4), por unidad de t (γ=16 ⇒ ÷16 por unidad de γt). κ/ω={T.gam/T.Dm:.2f}.\n",
      "Tasa medida = −d ln(paridad)/dt sobre γt∈[20,200].\n",
      "| versión | ω_p | tasa medida (1/t) | medida/predicción (con κ²/4) | error de la fórmula | medida/(sin κ²/4) |", "|---|---|---|---|---|---|"]
for (ver, name), r in R.items():
    m = (r['gts'] >= 20); t = r['gts'][m] / T.gam; y = np.log(np.clip(r['par'][m], 1e-6, None)); rate = -np.polyfit(t, y, 1)[0]
    L.append(f"| ({ver}) | {name} | {rate:.4f} | {rate/form:.3f} | {abs(rate/form-1)*100:.0f}% | {rate/form0:.3f} |")
# convergencia en N
ts, st = T.simulate('ii', wr, 28, gts, dscale=DS); r24 = T.analyze(ts, st, wr, 28, alpha=R['ii', 're-sint.']['alpha']); r16 = R['ii', 're-sint.']
L += ["", "## Convergencia N=20→28 (versión (ii), ω_p re-sintonizada, γt=200)\n", "| cantidad | N=20 | N=28 | Δrel |", "|---|---|---|---|"]
for k in ('Pc', 'par', 'Pe', 'F'): L.append(f"| {k} | {r16[k][-1]:.5f} | {r24[k][-1]:.5f} | {abs(r24[k][-1]/r16[k][-1]-1):.1e} |")
L.append(f"| |⟨a²⟩| | {abs(r16['a2'][-1]):.5f} | {abs(r24['a2'][-1]):.5f} | {abs(abs(r24['a2'][-1])/abs(r16['a2'][-1])-1):.1e} |")
te = max(float(r['trace'].max()) for r in R.values()); he = max(float(r['herm'].max()) for r in R.values()); me = min(float(r['mineig'].min()) for r in R.values())
L.append(f"\nValidaciones (4 corridas, todos los tiempos): |Tr ρ−1| ≤ {te:.1e}, ‖ρ−ρ†‖ ≤ {he:.1e}, mín autovalor ≥ {me:.1e} (tol 1e-10 / 1e-10 / −1e-9).")
open("tarea47b_task46_resultados.md", "w").write("\n".join(L)); print("\n".join(L))
