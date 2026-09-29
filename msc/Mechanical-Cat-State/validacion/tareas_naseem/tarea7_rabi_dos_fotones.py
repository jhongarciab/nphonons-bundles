# -*- coding: utf-8 -*-
"""
Tarea 7: medida directa de g_eff por oscilacion de Rabi de dos fotones.

Modelo completo, SIN drive (Omega=0), SIN disipacion (kappa=0, gamma=0):
evolucion unitaria pura. Estado inicial |g>x|n=2>, Nb=12.

En resonancia exacta (omega_q = 2 omega_m), |g,2> y |e,0> son degenerados y
el acoplamiento de segundo orden (virtual, mediado por los terminos de un
fonon) induce una oscilacion de Rabi de DOS FOTONES entre ambos estados,
con frecuencia esperada Omega_R = 2*sqrt(2)*g_eff (elemento de matriz
g_eff*sqrt(2!) entre |e,0> y |g,2>, Rabi = 2*|elemento de matriz|).

Se mide Omega_R por FFT de P(t)=|<e,0|psi(t)>|^2 (unitario, se usa el
proyector como e_op) y se compara con g_eff=g y g_eff=2g. Se repite
escalando g_z por {0.5, 1, 2} MANTENIENDO g_x fijo (no la razon r=gx/gz),
para que g_eff = gz*gx/omega_m escale linealmente con g_z y se pueda
verificar esa escala.

Si la oscilacion no llega a P_max=1, se estima la desintonia efectiva
(tipo Bloch-Siegert/Lamb shift) via:
    Omega_gen = pico FFT (frecuencia generalizada observada)
    P_max     = maximo de P(t)
    Omega_R   = Omega_gen * sqrt(P_max)      (acoplamiento resonante real)
    delta_eff = Omega_gen * sqrt(1-P_max)    (desintonia efectiva)

NO se modifica ningun script original.
"""

import numpy as np
from qutip import tensor, qeye, destroy, basis, mesolve, sigmaz, sigmax, Options

r = 0.1
gz_org, omega_m_org, kappa_org = 2 * np.pi * 6e6, 2 * np.pi * 100e6, 2 * np.pi * 100e3
gz_baseline = gz_org / kappa_org
om_m = omega_m_org / kappa_org
gx_baseline = r * gz_baseline   # se mantiene FIJO al escalar g_z

Na, Nb = 2, 12
omega_q_dim = 2 * om_m

options = Options(nsteps=2_000_000, atol=1e-12, rtol=1e-10)

# NOTA: verificado numericamente que sigmam() mapea basis(Na,0) (sz=+1) -> basis(Na,1)
# (sz=-1), es decir basis(Na,0) es el estado EXCITADO y basis(Na,1) el estado
# BASE fisico (opuesto a como lo etiquetan los comentarios de fig2/fig3, que llaman
# "ground" a thermal_dm(Na,0)=basis(Na,0) sin verificar el signo de sz). La pareja
# resonante para el proceso de dos fonones es |base,2> <-> |excitado,0> (ambas con
# energia +omega_q/2 cuando omega_q=2omega_m), confirmado numericamente (ver abajo).
psi0 = tensor(basis(Na, 1), basis(Nb, 2))              # |base, 2>
psi_target = tensor(basis(Na, 0), basis(Nb, 0))        # |excitado, 0>
P_target = psi_target * psi_target.dag()

tau_max, n_steps = 60.0, 1200
tlist = np.linspace(0, tau_max, n_steps)
dt_out = tlist[1] - tlist[0]


def correr(gz_scale):
    gz = gz_scale * gz_baseline
    gx = gx_baseline   # FIJO (no se re-escala con r)
    g_theory = gz * gx / om_m

    b = tensor(qeye(Na), destroy(Nb))
    bd = b.dag()
    sz = tensor(sigmaz(), qeye(Nb))
    sx = tensor(sigmax(), qeye(Nb))

    H0 = 0.5 * omega_q_dim * sz
    H = [
        H0,
        [gx * sx * b, lambda t, _: np.exp(-1j * om_m * t)],
        [gx * sx * bd, lambda t, _: np.exp(+1j * om_m * t)],
        [gz * sz * b, lambda t, _: np.exp(-1j * om_m * t)],
        [gz * sz * bd, lambda t, _: np.exp(+1j * om_m * t)],
    ]

    res = mesolve(H, psi0, tlist, [], [P_target], options=options)
    P_t = res.expect[0].real

    # FFT (se resta la media, ventana de Hann para reducir fuga espectral)
    sig = P_t - np.mean(P_t)
    window = np.hanning(len(sig))
    spec = np.fft.rfft(sig * window)
    freqs = np.fft.rfftfreq(len(sig), d=dt_out) * 2 * np.pi   # frecuencia angular
    ipeak = np.argmax(np.abs(spec[1:])) + 1
    omega_gen = freqs[ipeak]

    P_max = P_t.max()
    P_max_c = min(P_max, 1.0)
    Omega_R = omega_gen * np.sqrt(P_max_c)
    delta_eff = omega_gen * np.sqrt(max(0.0, 1 - P_max_c))

    return dict(g_theory=g_theory, omega_gen=omega_gen, P_max=P_max,
                Omega_R=Omega_R, delta_eff=delta_eff, P_t=P_t)


print(f"g_x (fijo) = {gx_baseline:.6f} kappa_org-units,  om_m = {om_m:.1f}")
print(f"Resolucion temporal de salida dt_out = {dt_out:.4f}  "
      f"(1/(20*om_m) = {1/(20*om_m):.2e} -- el integrador ODE resuelve internamente "
      f"la escala rapida om_m via su propio control adaptativo de paso; "
      f"dt_out solo necesita resolver la envolvente lenta de Rabi)")

resultados = []
for gz_scale in [0.5, 1.0, 2.0]:
    print(f"\n=== g_z escalado x{gz_scale} ===")
    r_ = correr(gz_scale)
    Omega_pred_g = 2 * np.sqrt(2) * r_["g_theory"]
    Omega_pred_2g = 2 * np.sqrt(2) * (2 * r_["g_theory"])
    print(f"  g_theory = {r_['g_theory']:.6f}")
    print(f"  P_max medido = {r_['P_max']:.5f}")
    print(f"  Omega_gen (pico FFT)      = {r_['omega_gen']:.6f}")
    print(f"  Omega_R (corregido P_max) = {r_['Omega_R']:.6f}")
    print(f"  delta_eff (Bloch-Siegert) = {r_['delta_eff']:.6f}")
    print(f"  Prediccion 2sqrt2*g   = {Omega_pred_g:.6f}  "
          f"(razon medido/pred = {r_['Omega_R']/Omega_pred_g:.4f})")
    print(f"  Prediccion 2sqrt2*2g  = {Omega_pred_2g:.6f}  "
          f"(razon medido/pred = {r_['Omega_R']/Omega_pred_2g:.4f})")
    resultados.append((gz_scale, r_["g_theory"], r_["P_max"], r_["omega_gen"],
                        r_["Omega_R"], r_["delta_eff"], Omega_pred_g, Omega_pred_2g))

print("\n=== TABLA RESUMEN (Tarea 7) ===")
hdr = (f"{'gz_x':>6} {'g_theory':>10} {'P_max':>8} {'Omega_gen':>10} {'Omega_R':>9} "
       f"{'delta_eff':>10} {'pred(g)':>9} {'pred(2g)':>9} {'razon_g':>8} {'razon_2g':>9}")
print(hdr)
for row in resultados:
    gzs, gth, pmax, wgen, wR, deff, pg, p2g = row
    print(f"{gzs:>6.2f} {gth:>10.6f} {pmax:>8.5f} {wgen:>10.6f} {wR:>9.6f} "
          f"{deff:>10.6f} {pg:>9.6f} {p2g:>9.6f} {wR/pg:>8.4f} {wR/p2g:>9.4f}")

# Verificacion de escala lineal g_eff ~ Omega_R/(2sqrt2) vs g_z
g_eff_medidos = np.array([row[4] / (2 * np.sqrt(2)) for row in resultados])
gz_scales = np.array([row[0] for row in resultados])
razon = g_eff_medidos / gz_scales
print(f"\ng_eff medido / g_z_scale (deberia ser constante si g_eff ~ g_z): {razon}")
print(f"Variacion relativa = {(razon.max()-razon.min())/razon.mean():.4%}")

np.savez("tarea7_resultados.npz", resultados=np.array(resultados),
         g_eff_medidos=g_eff_medidos, gz_scales=gz_scales)
print("\nGuardado: tarea7_resultados.npz")
