"""P10: exceso de la tasa de paridad del modelo completo con filtro sobre la predicción C1 filtrada.
Lee data/filtro_completo/*.npz (con filtro) y el punto plano de data/pfig3/ (g_x = 0.021650635, g_z/κ = 12).
Δγ = γ_pf,completo − 2[Γ₁⁻|α_eff|² + Γ₁⁺(|α_eff|² + 1)], con Γ₁± filtrados (o planos) y |α_eff²| = |⟨(a−d)²⟩|.
Escribe data/p10_exceso.csv y ajusta Δγ ∝ (g_z/κ)^p a κ₂/κ = 0.03, κ_f = 0.3, |α|² = 4.
"""
import os, glob
import numpy as np
import comun as C


def pred(gx, w, kap, kf, a2):
    ke = (lambda d: kap * kf**2 / (4 * d**2 + kf**2)) if kf > 0 else (lambda d: kap)
    gm, gp = gx**2 * ke(w) / (w**2 + (0 if kf else kap**2 / 4)), gx**2 * ke(3 * w) / (9 * w**2 + (0 if kf else kap**2 / 4))
    return 2 * (gm * a2 + gp * (a2 + 1))


def main():
    filas = []
    for f in sorted(glob.glob(os.path.join(C.DATA, 'filtro_completo', '*.npz'))):
        z = np.load(f)
        var = str(z['variante']) if 'variante' in z.files else 'full'
        gx, w, gz, kap, kf, al2 = (float(z[k]) for k in ('gx', 'w', 'gz', 'kap', 'kf', 'al2'))
        a2 = abs(complex(z['al2eff'])); g = float(z['gpf']); gp_ = pred(gx, w, kap, kf, a2)
        pe = float(z['Pe_prom']) if 'Pe_prom' in z.files else np.nan
        filas.append([gz / kap, float(z['kap2']) / kap, kf / w, al2, int(z['N']), var == 'nogz_pair', g, gp_, g - gp_, g / gp_, pe, gz / w])
    for f in glob.glob(os.path.join(C.DATA, 'pfig3', 'gx0.021650635*.npz')):
        z = np.load(f)
        gx, w, gz = float(z['gx']), float(z['w']), float(z['gz'])
        a2 = abs(complex(z['al2eff'])); g = float(z['gpf']); gp_ = pred(gx, w, 0.03, 0, a2)
        filas.append([gz / 0.03, float(z['kap2']) / 0.03, 0.0, float(z['al2']), int(z['N']), False, g, gp_, g - gp_, g / gp_, np.nan, gz / w])
    T = np.array(filas, dtype=float)
    np.savetxt(os.path.join(C.DATA, 'p10_exceso.csv'), T, delimiter=',', comments='',
               header='g_z/kappa, kappa_2/kappa, kappa_f/omega (0 = flat), |alpha|^2 nominal, N, no-gz variant, gamma_pf full, gamma_pf pred C1, '
                      'excess = full - pred, full/pred, P_e period-averaged, g_z/omega')
    print("g_z/κ  κ₂/κ  κ_f/ω  |α|²  N  sin-gz   γ_completo   γ_pred       Δγ          razón   P_e")
    for r in T[np.lexsort((T[:, 3], T[:, 2], T[:, 1], T[:, 0]))]:
        print(f"{r[0]:5.1f} {r[1]:.3f} {r[2]:.3f} {r[3]:3.0f} {r[4]:3.0f}  {int(r[5])}   {r[6]:.4e}  {r[7]:.4e}  {r[8]:+.4e}  {r[9]:6.3f}  {r[10]:.4f}")
    m = (abs(T[:, 1] - 0.03) < 1e-3) & (abs(T[:, 2] - 0.05) < 1e-3) & (T[:, 3] == 4) & (T[:, 5] == 0) & (T[:, 8] > 0)
    if m.sum() >= 3:
        p, lnA = np.polyfit(np.log(T[m, 0]), np.log(T[m, 8]), 1)
        print(f"ajuste Δγ ∝ (g_z/κ)^p a κ₂/κ = 0.03, κ_f/ω = 0.05, |α|² = 4: p = {p:.2f}, A = {np.exp(lnA):.3e}  ({m.sum()} puntos)")
    return T


if __name__ == '__main__':
    main()
