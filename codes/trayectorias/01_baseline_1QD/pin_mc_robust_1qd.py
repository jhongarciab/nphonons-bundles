#!/usr/bin/env python3
"""
Pi_N robusta (1QD, Monte Carlo) — sin T_N

Definición objetivo (Bin):
Pi_N = Pbar_N / sum_{m=1..N} Pbar_m
"""
import argparse
import os
import time
from math import factorial

import numpy as np
import qutip as qt


def build_ops(ncut):
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
            if abs(rho.tr() - 1) < 1e-4 and rho.isherm:
                return rho
        except Exception:
            pass
    return None


def bootstrap_ci(samples, nboot=200, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(123)
    x = np.array(samples, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    boots = []
    for _ in range(nboot):
        idx = rng.integers(0, len(x), len(x))
        boots.append(np.mean(x[idx]))
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./out_pin_robust")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--ncut", type=int, default=16)
    ap.add_argument("--n-lam", type=int, default=6)
    ap.add_argument("--n-kappa", type=int, default=6)
    ap.add_argument("--n-delta", type=int, default=9)
    ap.add_argument("--ntraj", type=int, default=80)
    ap.add_argument("--lam-min", type=float, default=0.02)
    ap.add_argument("--lam-max", type=float, default=0.14)
    ap.add_argument("--kappa-min", type=float, default=1e-3)
    ap.add_argument("--kappa-max", type=float, default=1.0)
    ap.add_argument("--omega-b", type=float, default=1.0)
    ap.add_argument("--Omega", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=2e-4)
    ap.add_argument("--gamma-phi", type=float, default=4e-4)
    ap.add_argument("--transient-periods", type=float, default=2.0)
    ap.add_argument("--sample-periods", type=float, default=4.0)
    ap.add_argument("--min-sample-count", type=int, default=600)
    ap.add_argument("--tmax-cap", type=float, default=8e4)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    lam_arr = np.linspace(args.lam_min, args.lam_max, args.n_lam)
    kappa_arr = np.logspace(np.log10(args.kappa_min), np.log10(args.kappa_max), args.n_kappa)

    b_sys, nb_sys, sm_sys, sp_sys, proj_exc = build_ops(args.ncut)
    H_ph = args.omega_b * nb_sys
    H_dr = args.Omega * (sm_sys + sp_sys)
    psi0 = qt.tensor(qt.basis(2, 0), qt.basis(args.ncut, 0))

    phonon_proj = [qt.tensor(qt.qeye(2), qt.fock_dm(args.ncut, m)) for m in range(args.ncut)]

    def H(delta, lam):
        return H_ph + delta * proj_exc + lam * proj_exc * (b_sys + b_sys.dag()) + H_dr

    def cops(kappa):
        return [np.sqrt(kappa) * b_sys, np.sqrt(args.gamma) * sm_sys, np.sqrt(args.gamma_phi) * proj_exc]

    def find_delta_opt(n, lam, kappa):
        d_est = -n * args.omega_b + n * lam ** 2 / args.omega_b
        scan = np.linspace(d_est - 0.3, d_est + 0.3, args.n_delta)
        opn = (b_sys.dag() ** n) * (b_sys ** n)
        c = cops(kappa)
        best = (-1.0, d_est)
        for d in scan:
            rho = solve_ss(H(d, lam), c)
            if rho is None:
                continue
            v = float(np.real(qt.expect(opn, rho)))
            if v > best[0]:
                best = (v, d)
        return best[1]

    Pi = np.full((args.n_kappa, args.n_lam), np.nan)
    Pi_lo = np.full((args.n_kappa, args.n_lam), np.nan)
    Pi_hi = np.full((args.n_kappa, args.n_lam), np.nan)
    Dopt = np.full((args.n_kappa, args.n_lam), np.nan)
    n_samples_map = np.zeros((args.n_kappa, args.n_lam), dtype=int)
    status = np.empty((args.n_kappa, args.n_lam), dtype=object)

    t0 = time.time()
    for j, lam in enumerate(lam_arr):
        for i, kappa in enumerate(kappa_arr):
            d = find_delta_opt(args.n, lam, kappa)
            Dopt[i, j] = d

            # escala temporal efectiva
            omega_eff = args.Omega * (lam / args.omega_b) ** args.n / np.sqrt(factorial(args.n))
            if omega_eff < 1e-13:
                status[i, j] = "omega_eff_too_small"
                continue

            T_rabi = np.pi / omega_eff
            t_trans = args.transient_periods * T_rabi
            t_end = t_trans + args.sample_periods * T_rabi
            if t_end > args.tmax_cap:
                t_end = args.tmax_cap

            n_t_trans = 10
            n_t_samp = max(30, int(args.min_sample_count / max(args.ntraj, 1)))
            tlist = np.unique(np.concatenate([
                np.linspace(0, t_trans, n_t_trans),
                np.linspace(t_trans, t_end, n_t_samp),
            ]))

            try:
                res = qt.mcsolve(
                    H(d, lam), psi0, tlist, cops(kappa), ntraj=args.ntraj,
                    options={"store_states": True, "keep_runs_results": True, "progress_bar": False},
                )
            except Exception:
                status[i, j] = "mcsolve_fail"
                continue

            idx_start = n_t_trans
            pi_samples = []
            for traj in res.runs_states:
                for ti in range(idx_start, len(tlist)):
                    if ti >= len(traj):
                        break
                    psi_t = traj[ti]
                    pops = np.array([float(np.real(qt.expect(phonon_proj[m], psi_t))) for m in range(args.n + 1)])
                    den = np.sum(pops[1:args.n + 1])
                    pi_val = 0.0 if den < 1e-18 else float(np.clip(pops[args.n] / den, 0.0, 1.0))
                    pi_samples.append(pi_val)

            n_samples_map[i, j] = len(pi_samples)
            if len(pi_samples) < max(50, args.min_sample_count // 4):
                status[i, j] = "too_few_samples"
                continue

            Pi[i, j] = float(np.mean(pi_samples))
            lo, hi = bootstrap_ci(pi_samples, nboot=200, rng=rng)
            Pi_lo[i, j], Pi_hi[i, j] = lo, hi
            status[i, j] = "ok"

        elapsed = time.time() - t0
        eta = elapsed / (j + 1) * (args.n_lam - j - 1)
        print(f"lambda={lam:.3f} {j+1}/{args.n_lam} t={elapsed:.1f}s ETA={eta:.1f}s", flush=True)

    np.save(os.path.join(args.out, "lambda_arr.npy"), lam_arr)
    np.save(os.path.join(args.out, "kappa_arr.npy"), kappa_arr)
    np.save(os.path.join(args.out, f"PiN_mc_robust_1qd_N{args.n}.npy"), Pi)
    np.save(os.path.join(args.out, f"PiN_mc_robust_lo_1qd_N{args.n}.npy"), Pi_lo)
    np.save(os.path.join(args.out, f"PiN_mc_robust_hi_1qd_N{args.n}.npy"), Pi_hi)
    np.save(os.path.join(args.out, f"Delta_opt_robust_1qd_N{args.n}.npy"), Dopt)
    np.save(os.path.join(args.out, f"n_samples_1qd_N{args.n}.npy"), n_samples_map)
    np.save(os.path.join(args.out, f"status_1qd_N{args.n}.npy"), status)

    ok = np.sum(status == "ok")
    total = status.size
    print(f"Done. ok={ok}/{total} ({100*ok/total:.1f}%) out={args.out}")


if __name__ == "__main__":
    main()
