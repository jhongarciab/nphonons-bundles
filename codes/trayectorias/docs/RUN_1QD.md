# Ejecución rápida — 1QD

## 1) Baseline robusto (solo Pi_N por MC)
```bash
cd "/Users/jhongarciabarrera/Desktop/Trabajo de Grado/trabajo/codes/trayectorias/01_baseline_1QD"
python3 pin_mc_robust_1qd.py --out ./out_pin_n2_pilot --n 2 --n-lam 6 --n-kappa 6 --ntraj 80
```

## 2) (Opcional) baseline + proxy T_N
```bash
cd "/Users/jhongarciabarrera/Desktop/Trabajo de Grado/trabajo/codes/trayectorias/01_baseline_1QD"
python3 bin_purity_mc_1qd.py --out ./out_pilot --n-lam 8 --n-kappa 8 --ntraj 16 --N-list 2,3
```

## 3) (Opcional) Comparar T_N vs Pi_N
```bash
cd "/Users/jhongarciabarrera/Desktop/Trabajo de Grado/trabajo/codes/trayectorias/02_validacion_TN_vs_PiN"
python3 compare_tn_vs_pin_1qd.py --in-dir ../01_baseline_1QD/out_pilot --out ./compare_pilot --N-list 2,3
```
