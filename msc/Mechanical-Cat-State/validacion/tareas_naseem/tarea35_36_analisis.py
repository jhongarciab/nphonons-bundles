import numpy as np, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = ["# Tareas 35-36\n"]
# ================= Tarea 35 =================
L += ["## Tarea 35 — el cuádruplete de Floquet (α²=2, (0.048,0.144))\n", "### (a) Convergencia en N (brecha k=4 y cuádruplete)\n",
      "| Γ₂/κ | N | brecha Re | Im | cuádruplete Re | cuádruplete \\|Im\\| | paso 5 |", "|---|---|---|---|---|---|---|"]
D = {}
for G in ("0.13", "0.3", "1.0"):
    for N in (20, 26, 32):
        d = np.load(f"tarea35_cache/G{G}_N{N}.npz"); D[G, N] = d
        lam = d['lam']; q = d['quad']
        qs = f"{lam[q[0]].real:.5f}" if len(q) else "—"; qi = f"{abs(lam[q[0]].imag):.3f}" if len(q) else "—"
        L.append(f"| {G} | {N} | {float(d['gap']):.5f} | {float(d['gap_im']):+.2f} | {qs} | {qi} | {float(d['gap_s5']):.5f} |")
L.append("")
for G in ("0.13", "0.3", "1.0"):
    q = [ (D[G, N]['lam'][D[G, N]['quad'][0]].real, abs(D[G, N]['lam'][D[G, N]['quad'][0]].imag)) if len(D[G, N]['quad']) else (np.nan, np.nan) for N in (20, 26, 32)]
    g = [float(D[G, N]['gap']) for N in (20, 26, 32)]
    L.append(f"- Γ₂/κ={G}: brecha N20→26→32: {g[0]:.5f}→{g[1]:.5f}→{g[2]:.5f} (Δrel {abs(g[1]/g[0]-1):.1e}, {abs(g[2]/g[1]-1):.1e}); "
             f"cuádruplete Re {q[0][0]:.5f}→{q[1][0]:.5f}→{q[2][0]:.5f}, |Im| {q[0][1]:.3f}→{q[1][1]:.3f}→{q[2][1]:.3f}.")
L += ["", "### (b) Plegamiento", f"2π/T_m = ω_m = {float(D['0.13', 20]['omega_r']):.1f} (unidades de κ; T_m=2π/ω_m={float(D['0.13', 20]['T_r']):.4e}).", "",
      "| Γ₂/κ | \\|Im\\| cuádruplete (N=20) | \\|Im\\|/(2π/T_m) | distancia al múltiplo entero más cercano de 2π/T_m |", "|---|---|---|---|"]
for G in ("0.13", "0.3", "1.0"):
    d = D[G, 20]; om = float(d['omega_r']); q = d['quad']
    if len(q):
        im = abs(d['lam'][q[0]].imag); near = round(im / om) * om
        L.append(f"| {G} | {im:.3f} | {im/om:.4f} | {abs(im-near):.3f} (múltiplo {int(round(im/om))}) |")
L.append("\nIm(λ)=[−ln μ]/T_m está definido módulo 2π/T_m=1000; |Im|=7-37 son ≪ 1000 (0.7%-3.7%): el cuádruplete no está plegado desde un múltiplo alto sino cerca del múltiplo 0 (frecuencia genuina de decenas de κ, no un alias).\n")
# ---- (c)
L += ["### (c) Relevancia dinámica: retorno al espacio del código\n",
      "Fuga ℓ(t)=1−Tr(Πρ), Π = proyector del código (|±α⟩⊗|g⟩); tasa = pendiente de ln(ℓ−ℓ_∞) con ℓ_∞ = valor final. Ventanas: temprana (ℓ' de 3e-1 a 1e-2) y tardía (1e-2 a 1e-5).\n",
      "| Γ₂/κ | estado | tasa temprana | tasa tardía | brecha Floquet | paso 5 (rama lenta estática) | cuádruplete Re | fracción del estado en el cuádruplete |", "|---|---|---|---|---|---|---|---|"]
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
for ax, G in zip(axs, ("0.13", "0.3", "1.0")):
    d = D[G, 20]; t = d['t']; lam = d['lam']; q = d['quad']
    for name, lab in (("coh", "|1.3α⟩|g⟩"), ("fock", "|4⟩|g⟩"), ("exc", "|α⟩|e⟩")):
        l = d[name + '_leak']; fl = np.mean(l[-30:]); lp = np.clip(l - fl, 1e-16, None)
        def rate(lo, hi):
            m = (lp < hi) & (lp > lo) & (t > 0)
            return -np.polyfit(t[m], np.log(lp[m]), 1)[0] if m.sum() > 3 else np.nan
        re_, rl = rate(1e-2, 3e-1), rate(1e-5, 1e-2)
        L.append(f"| {G} | {lab} | {re_:.4f} | {rl:.4f} | {float(d['gap']):.4f} | {float(d['gap_s5']):.4f} | {lam[q[0]].real:.4f} | {float(d[name+'_frac_quad']):.1e} |")
        ax.semilogy(t, lp, label=lab)
    ax.set_title(f"Γ₂/κ={G}"); ax.set_xlabel("t (1/κ)"); ax.set_ylim(1e-9, 1); ax.set_xlim(0, 60); ax.legend(fontsize=7)
    ax.axhline(1, color='none')
fig.tight_layout(); fig.savefig("tarea35_retorno.png", dpi=120)
L.append("\nModos que más pesan en la expansión modal del estado inicial (autovectores izquierdos), N=20 (k ordenado por Re; peso relativo a los no-código):\n")
for G in ("0.13", "0.3", "1.0"):
    d = D[G, 20]; lam = d['lam']
    for name in ("coh", "fock", "exc"):
        w = d[name + '_w']; top = np.argsort(-w)[:3]
        L.append(f"- Γ₂/κ={G} {name}: " + "; ".join(f"k={k} (Re {lam[k].real:.4f}, Im {lam[k].imag:+.2f}) w={w[k]:.2f}" for k in top) + f"; recon={float(d[name+'_recon']):.0e}")
dd = np.load("tarea35_cache/G0.13_N26dyn.npz")
L.append(f"\nConvergencia N=26 de (c) en Γ₂/κ=0.13: fracción cuádruplete coh/fock/exc = {float(dd['coh_frac_quad']):.1e}/{float(dd['fock_frac_quad']):.1e}/{float(dd['exc_frac_quad']):.1e}.")
# ================= Tarea 36 =================
c = 299792458 * 0 + 1
kB_h = 1.380649e-23 / 6.62607015e-34   # Hz/K
L += ["\n## Tarea 36 — umbral térmico (Γ₂/κ=0.13), propagador (1e-15,1e-13)\n"]
vfail = []
for a2 in (2, 3):
    fs = sorted(glob.glob(f"tarea36_cache/a2{a2}_nq*_N0.npz"), key=lambda f: float(np.load(f)['nq']))
    R = [np.load(f) for f in fs]; nq = np.array([float(r['nq']) for r in R]); eta = np.array([float(r['eta']) for r in R])
    L += [f"### |α|²={a2}\n", "| n_q | γ_pf | γ_bf | η=γ_pf/γ_bf | P(código) | \\|⟨a²⟩\\| (ideal α²) | pureza | brecha | tr / herm / min λ |", "|---|---|---|---|---|---|---|---|---|"]
    for r in R:
        L.append(f"| {float(r['nq']):.3g} | {float(r['gamma_pf']):.3e} | {float(r['gamma_bf']):.3e} | {float(r['eta']):.3g} | {float(r['p_code']):.4f} | {float(r['a2']):.4f} | {float(r['purity']):.4f} | {float(r['gap']):.4f} | {float(r['trace_err']):.0e}/{float(r['herm']):.0e}/{float(r['min_eig']):.0e} |")
        if not (float(r['trace_err']) < 1e-10 and float(r['herm']) < 1e-10 and float(r['min_eig']) > -1e-10): vfail.append((a2, float(r['nq'])))
    L.append("")
    def cross(target):
        for i in range(len(nq) - 1):
            if (eta[i] - target) * (eta[i + 1] - target) < 0:
                f = (np.log(target) - np.log(eta[i])) / (np.log(eta[i + 1]) - np.log(eta[i]))
                return np.exp(np.log(nq[i]) + f * (np.log(nq[i + 1]) - np.log(nq[i])))
        return np.nan
    L += ["| η objetivo | n_q* | f_min a 10 mK (GHz) | 20 mK | 50 mK |", "|---|---|---|---|---|"]
    for tg in (100, 10, 1):
        n_ = cross(tg)
        fm = [kB_h * T * np.log(1 + 1 / n_) / 1e9 if np.isfinite(n_) else np.nan for T in (0.010, 0.020, 0.050)]
        L.append(f"| {tg} | {n_:.4g} | {fm[0]:.3f} | {fm[1]:.3f} | {fm[2]:.3f} |")
    L.append("")
    plt.figure(figsize=(5, 4)); plt.loglog(nq, eta, 'o-'); [plt.axhline(v, color='gray', ls=':') for v in (1, 10, 100)]
    plt.xlabel("n_q"); plt.ylabel("η=γ_pf/γ_bf"); plt.title(f"|α|²={a2}"); plt.tight_layout(); plt.savefig(f"tarea36_eta_a2{a2}.png", dpi=120); plt.close()
L.append("Convergencia en N (n_q=0.3):")
for a2, Np in ((2, 26), (3, 30)):
    r0 = [np.load(f) for f in glob.glob(f"tarea36_cache/a2{a2}_nq0.3_N0.npz")]
    if not r0: r0 = [min((np.load(f) for f in glob.glob(f"tarea36_cache/a2{a2}_nq*_N0.npz")), key=lambda r: abs(float(r['nq']) - 0.3))]
    r0 = r0[0]; r1 = np.load(f"tarea36_cache/a2{a2}_nq0.3_Nplus.npz")
    ch = lambda k: abs(float(r1[k]) / float(r0[k]) - 1)
    L.append(f"- |α|²={a2}, N {int(r0['N'])}→{int(r1['N'])}: γ_pf {ch('gamma_pf'):.1e}, γ_bf {ch('gamma_bf'):.1e}, η {ch('eta'):.1e}, P(código) {ch('p_code'):.1e}, |⟨a²⟩| {ch('a2'):.1e}, brecha {ch('gap'):.1e} (relativos).")
L.append(f"\nValidaciones de ρ (tol 1e-10): {'TODAS OK' if not vfail else 'FALLAN en (α²,n_q) = ' + str(vfail)}")
open("tarea35_36_resultados.md", "w").write("\n".join(L)); print("\n".join(L))
