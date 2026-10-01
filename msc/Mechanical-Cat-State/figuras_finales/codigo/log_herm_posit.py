"""Log crudo de hermiticidad y positividad de las corridas del modelo completo (data/filtro_completo/*gam6e-07*.npz, κ₂/κ = 0.05–0.4 con filtro y plano).
herm_cruda = ‖ρ − ρ†‖_F con Tr ρ = 1 antes de hermitizar; mín. autovalor de (ρ + ρ†)/2. Umbrales (fase 6): plano 1e-10 en hermiticidad y −1e-9 en positividad (sin cambios);
completo con filtro 1e-8 en hermiticidad y −1e-8 en positividad. Los valores crudos se registran siempre; se marca cada umbral superado.
Escribe data/filtro_completo/log_hermiticidad_positividad.txt."""
import os, glob
import numpy as np
import comun as C
filas = []
for f in sorted(glob.glob(os.path.join(C.DATA, 'filtro_completo', '*gam6e-07*.npz'))):
    z = np.load(f)
    filas.append((os.path.basename(f), float(z['kap2']) / float(z['kap']), int(z['N']), float(z['x']), float(z['herm_cruda']), float(z['val'][0]), float(z['val'][2])))
UMB = {True: (1e-10, -1e-9), False: (1e-8, -1e-8)}      # clave: es plano
with open(os.path.join(C.DATA, 'filtro_completo', 'log_hermiticidad_positividad.txt'), 'w') as o:
    o.write('# herm = ||rho - rho^dag||_F (Tr rho = 1, antes de hermitizar); mineig = min eig de (rho + rho^dag)/2\n')
    o.write('# umbrales: plano herm 1e-10, mineig >= -1e-9 (sin cambios); completo con filtro herm 1e-8, mineig >= -1e-8 (fase 6). Se marca el umbral superado.\n')
    o.write(f"{'archivo':95s} k2/k   N   x      herm      |Tr-1|    mineig     marcas\n")
    nh = {True: 0, False: 0}; npo = {True: 0, False: 0}; hh = {True: 0, False: 0}; pp = {True: 0, False: 0}; n = {True: 0, False: 0}
    for nombre, k, N, x, h, tr, me in filas:
        pl = 'plano' in nombre; uh, up = UMB[pl]; n[pl] += 1
        m = []
        if h > uh: m.append(f'HERM>{uh:.0e}'); nh[pl] += 1
        if me < up: m.append(f'**POSITIVIDAD INCUMPLIDA (<{up:.0e})**'); npo[pl] += 1
        if not pl and h > 1e-10: hh[pl] += 1
        if not pl and me < -1e-9: pp[pl] += 1
        o.write(f'{nombre:95s} {k:5.3f} {N:3d} {x:5.2f} {h:9.2e} {tr:9.1e} {me:+10.2e}  {" ".join(m)}\n')
    o.write(f'# plano: {n[True]} corridas; herm > 1e-10: {nh[True]}; positividad < -1e-9: {npo[True]}\n')
    o.write(f'# filtro: {n[False]} corridas; herm > 1e-8: {nh[False]}; positividad < -1e-8: {npo[False]} (con los umbrales anteriores, herm > 1e-10: {hh[False]}, mineig < -1e-9: {pp[False]})\n')
print(open(os.path.join(C.DATA, 'filtro_completo', 'log_hermiticidad_positividad.txt')).read())
