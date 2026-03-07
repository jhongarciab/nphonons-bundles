#!/usr/bin/env python3
"""
Baseline 1QD: pureza tipo Bin (MC) + proxy T_N (Muñoz Eq. 20)

Modelo (1QD):
H = wb b†b + Δ σ†σ + λ σ†σ (b + b†) + Ω (σ + σ†)

Pureza MC (referencia):
Pi_N = Pbar_N / sum_{m=1..N} Pbar_m

Proxy Muñoz:
T_N = N <(b†)^N b^N> / ((N-1)! <b†b>)
"""
import argparse
import os
import time
from math import factorial

import numpy as np
import qutip as qt


def build_system(ncut):
    b = qt.destroy(ncut)
    nb = b.dag() * b
    I_b = qt.qeye(ncut)
    I_q = qt.qeye(2)
    sm = qt.sigmam()

    b_sys = qt.tensor(I_q, b)
    nb_sys = qt.tensor(I_q, nb)
    sm_sys = qt.tensor(sm, I_b)
    sp_sys = sm_sys.dag()
    proj_exc = sp_sys * sm_sys

    return b_sys, nb_sys, sm_sys, sp_sys, proj_exc


def solve_ss(H, c_ops):
    for method in ("direct", "eigen", "svd"):
        try:
            rho = qt.steadystate(H, c_ops, method=method, use_rcm=True)
            if abs(rho.tr() - 1) < 1e-4:
                return rho
        except Exception:
            pass
    return None


def tn_proxy(n, rho, b_sys, nb_sys):
    nbar = float(np.real(qt.expect(nb_sys, rho)))
    if nbar < 1e-18:
        return 0.0
    expn = float(np.real(qt.expect((b_sys.dag() ** n) * (b_sys ** n), rho)))
    val = n * expn / (factorial(n - 1) * nbar)
    return float(np.clip(val, 0.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./out_1qd")
    ap.add_argument("--ncut", type=int, default=16)
    ap.add_argument("--ntraj", type=int, default=16)
    ap.add_argument("--n-lam", type=int, default=8)
    ap.add_argument("--n-kappa", type=int, default=8)
    ap.add_argument("--n-delta", type=int, default=9)
    ap.add_argument("--lam-min", type=float, default=0.02)
    ap.add_argument("--lam-max", type=float, default=0.14)
    ap.add_argument("--kappa-min", type=float, default=1e-3)
    ap.add_argument("--kappa-max", type=float, default=1e0)
    ap.add_argument("--omega-b", type=float, default=1.0)
    ap.add_argument("--Omega", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=2e-4)
    ap.add_argument("--gamma-phi", type=float, default=4e-4)
    ap.add_argument("--N-list", default="2,3")
    ap.add_argument("--tmax-cap", type=float, default=3e4)
    ap.add_argument("--sample-periods", type=float, default=4.0)
    ap.add_argument("--transient-periods", type=float, default=2.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    N_list = [int(x.strip()) for x in args.N_list.split(",") if x.strip()]
    lam_arr = np.linspace(args.lam_min, args.lam_max, args.n_lam)
    kappa_arr = np.logspace(np.log10(args.kappa_min), np.log10(args.kappa_max), args.n_kappa)

    b_sys, nb_sys, sm_sys, sp_sys, proj_exc = build_system(args.ncut)
    H_phon = args.omega_b * nb_sys
    H_drive = args.Omega * (sm_sys + sp_sys)
    psi0 = qt.tensor(qt.basis(2, 0), qt.basis(args.ncut, 0))

    phonon_proj = [qt.tensor(qt.qeye(2), qt.fock_dm(args.ncut, m)) for m in range(args.ncut)]

    def build_H(delta, lam):
        return H_phon + delta * proj_exc + lam * proj_exc * (b_sys + b_sys.dag()) + H_drive

    def build_c_ops(kappa):
        return [
            np.sqrt(kappa) * b_sys,
            np.sqrt(args.gamma) * sm_sys,
            np.sqrt(args.gamma_phi) * proj_exc,
        ]

    def find_delta_opt(nb, lam, kappa):
        # Estimador régimen Stokes 1QD
        d_est = -nb * args.omega_b + nb * lam ** 2 / args.omega_b
        scan = np.linspace(d_est - 0.3, d_est + 0.3, args.n_delta)
        c_ops = build_c_ops(kappa)
        opn = (b_sys.dag() ** nb) * (b_sys ** nb)
        best = (-1.0, d_est, None)
        for d in scan:
            rho = solve_ss(build_H(d, lam), c_ops)
            if rho is None:
                continue
            val = float(np.real(qt.expect(opn, rho)))
            if val > best[0]:
                best = (val, d, rho)
        return best[1], best[2]

    t0_all = time.time()

    for nb in N_list:
        print(f"\\n=== N={nb} ===")
        Pi = np.full((args.n_kappa, args.n_lam), np.nan)
        Tn = np.full((args.n_kappa, args.n_lam), np.nan)
        Dopt = np.full((args.n_kappa, args.n_lam), np.nan)
        skipped = 0
        t0 = time.time()

        for j, lam in enumerate(lam_arr):
            for i, kappa in enumerate(kappa_arr):
                dopt, rho_opt = find_delta_opt(nb, lam, kappa)
                Dopt[i, j] = dopt

                if rho_opt is None:
                    skipped += 1
                    continue

                # Proxy T_N
                Tn[i, j] = tn_proxy(nb, rho_opt, b_sys, nb_sys)

                # Escala temporal MC
                omega_eff = args.Omega * (lam / args.omega_b) ** nb / np.sqrt(factorial(nb))
                if omega_eff < 1e-12:
                    skipped += 1
                    continue
                T_rabi = np.pi / omega_eff
                t_trans = args.transient_periods * T_rabi
                t_end = t_trans + args.sample_periods * T_rabi
                if t_end > args.tmax_cap:
                    skipped += 1
                    continue

                n_t_trans = 8
                n_t_samp = 24
                tlist = np.unique(np.concatenate([
                    np.linspace(0, t_trans, n_t_trans),
                    np.linspace(t_trans, t_end, n_t_samp),
                ]))

                try:
                    result = qt.mcsolve(
                        build_H(dopt, lam), psi0, tlist, build_c_ops(kappa), ntraj=args.ntraj,
                        options={"store_states": True, "keep_runs_results": True, "progress_bar": False}
                    )
                except Exception:
                    skipped += 1
                    continue

                idx_start = n_t_trans
                pop = np.zeros(args.ncut)
                ns = 0
                for traj in result.runs_states:
                    for ti in range(idx_start, len(tlist)):
                        if ti >= len(traj):
                            break
                        psi_t = traj[ti]
                        for m in range(min(nb + 3, args.ncut)):
                            pop[m] += float(np.real(qt.expect(phonon_proj[m], psi_t)))
                        ns += 1

                if ns == 0:
                    skipped += 1
                    continue

                pbar = pop / ns
                den = np.sum(pbar[1:nb + 1])
                Pi[i, j] = 0.0 if den < 1e-18 else float(np.clip(pbar[nb] / den, 0.0, 1.0))

            elapsed = time.time() - t0
            eta = elapsed / (j + 1) * (args.n_lam - j - 1)
            col = Pi[:, j]
            v = col[~np.isnan(col)]
            vmax = np.max(v) if len(v) else np.nan
            print(f"N={nb} lambda={lam:.3f} {j+1}/{args.n_lam} Pi_max={vmax:.3f} skipped={skipped} t={elapsed:.0f}s ETA={eta:.0f}s")

        np.save(os.path.join(args.out, f"lambda_arr.npy"), lam_arr)
        np.save(os.path.join(args.out, f"kappa_arr.npy"), kappa_arr)
        np.save(os.path.join(args.out, f"PiN_mc_1qd_N{nb}.npy"), Pi)
        np.save(os.path.join(args.out, f"TN_proxy_1qd_N{nb}.npy"), Tn)
        np.save(os.path.join(args.out, f"Delta_opt_1qd_N{nb}.npy"), Dopt)
        print(f"N={nb} done in {time.time()-t0:.1f}s (skipped={skipped})")

    print(f"\\nAll done in {time.time()-t0_all:.1f}s. Output: {args.out}")


if __name__ == "__main__":
    main()
