#!/usr/bin/env python3
"""
Compara Pi_N (MC, tipo Bin) vs T_N proxy (Muñoz Eq. 20) en 1QD.
Genera métricas y figuras por N.
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def metrics(pi_map, tn_map):
    mask = ~np.isnan(pi_map) & ~np.isnan(tn_map)
    if not np.any(mask):
        return {"n": 0}
    a = pi_map[mask]
    b = tn_map[mask]
    err = b - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else np.nan
    return {
        "n": int(len(a)),
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
        "pi_mean": float(np.mean(a)),
        "tn_mean": float(np.mean(b)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", default="./compare_out")
    ap.add_argument("--N-list", default="2,3")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    N_list = [int(x.strip()) for x in args.N_list.split(",") if x.strip()]

    lam = np.load(os.path.join(args.in_dir, "lambda_arr.npy"))
    kappa = np.load(os.path.join(args.in_dir, "kappa_arr.npy"))

    report_lines = ["# Validación T_N vs Pi_N (1QD)", ""]

    for N in N_list:
        pi_f = os.path.join(args.in_dir, f"PiN_mc_1qd_N{N}.npy")
        tn_f = os.path.join(args.in_dir, f"TN_proxy_1qd_N{N}.npy")
        if not (os.path.exists(pi_f) and os.path.exists(tn_f)):
            report_lines.append(f"- N={N}: faltan archivos")
            continue

        pi_map = np.load(pi_f)
        tn_map = np.load(tn_f)
        diff = tn_map - pi_map

        m = metrics(pi_map, tn_map)
        report_lines += [
            f"## N={N}",
            f"- puntos válidos: {m.get('n',0)}",
            f"- MAE: {m.get('mae', float('nan')):.4f}",
            f"- RMSE: {m.get('rmse', float('nan')):.4f}",
            f"- Corr(Pi,TN): {m.get('corr', float('nan')):.4f}",
            f"- mean(Pi): {m.get('pi_mean', float('nan')):.4f}",
            f"- mean(TN): {m.get('tn_mean', float('nan')):.4f}",
            "",
        ]

        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
        fig.subplots_adjust(wspace=0.10, right=0.90)

        cmap = plt.cm.YlOrRd_r
        norm01 = Normalize(vmin=0, vmax=1)

        im0 = axes[0].pcolormesh(lam, kappa, np.nan_to_num(pi_map, nan=0.0), cmap=cmap, norm=norm01, shading='auto')
        axes[0].set_title(f'Pi_{N} (MC)')

        im1 = axes[1].pcolormesh(lam, kappa, np.nan_to_num(tn_map, nan=0.0), cmap=cmap, norm=norm01, shading='auto')
        axes[1].set_title(f'T_{N} (proxy)')

        vmax = np.nanmax(np.abs(diff))
        vmax = 1e-3 if (not np.isfinite(vmax) or vmax < 1e-3) else vmax
        im2 = axes[2].pcolormesh(lam, kappa, np.nan_to_num(diff, nan=0.0), cmap='coolwarm', norm=Normalize(vmin=-vmax, vmax=vmax), shading='auto')
        axes[2].set_title('T_N - Pi_N')

        for ax in axes:
            ax.set_yscale('log')
            ax.set_xlabel('lambda/omega_b')
            ax.set_xlim(lam[0], lam[-1])
            ax.set_ylim(kappa[0], kappa[-1])
        axes[0].set_ylabel('kappa/omega_b')

        cax1 = fig.add_axes([0.92, 0.14, 0.015, 0.72])
        cb1 = fig.colorbar(im1, cax=cax1)
        cb1.set_label('0..1')

        out_png = os.path.join(args.out, f"compare_1qd_N{N}.png")
        plt.savefig(out_png, dpi=220, bbox_inches='tight')
        plt.close(fig)

    report_path = os.path.join(args.out, "reporte_validacion.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Reporte: {report_path}")


if __name__ == "__main__":
    main()
