"""Log crudo de hermiticidad y positividad de las corridas del modelo completo (data/filtro_completo/*gam6e-07*.npz, κ₂/κ = 0.05–0.4 con filtro y plano).
herm_cruda = ‖ρ − ρ†‖_F con Tr ρ = 1 antes de hermitizar; umbral 1e-10. mín. autovalor de (ρ + ρ†)/2; criterio −1e-9 (sin cambios).
Escribe data/filtro_completo/log_hermiticidad_positividad.txt."""
import os, glob
import numpy as np
import comun as C
filas = []
for f in sorted(glob.glob(os.path.join(C.DATA, 'filtro_completo', '*gam6e-07*.npz'))):
    z = np.load(f)
    filas.append((os.path.basename(f), float(z['kap2']) / float(z['kap']), int(z['N']), float(z['x']), float(z['herm_cruda']), float(z['val'][0]), float(z['val'][2])))
with open(os.path.join(C.DATA, 'filtro_completo', 'log_hermiticidad_positividad.txt'), 'w') as o:
    o.write('# herm = ||rho - rho^dag||_F (Tr rho = 1, antes de hermitizar), umbral 1e-10; mineig = min eig de (rho + rho^dag)/2, criterio >= -1e-9 (sin cambios)\n')
    o.write(f"{'archivo':95s} k2/k   N   x      herm      |Tr-1|    mineig     marcas\n")
    for n, k, N, x, h, tr, me in filas:
        marcas = ('HERM>1e-10 ' if h > 1e-10 else '') + ('**POSITIVIDAD INCUMPLIDA (<-1e-9)**' if me < -1e-9 else '')
        o.write(f'{n:95s} {k:5.3f} {N:3d} {x:5.2f} {h:9.2e} {tr:9.1e} {me:+10.2e}  {marcas}\n')
    o.write(f'# {len(filas)} corridas; herm > 1e-10: {sum(r[4] > 1e-10 for r in filas)}; positividad incumplida: {sum(r[6] < -1e-9 for r in filas)}\n')
print(open(os.path.join(C.DATA, 'filtro_completo', 'log_hermiticidad_positividad.txt')).read().replace('gx', '\ngx') if False else open(os.path.join(C.DATA, 'filtro_completo', 'log_hermiticidad_positividad.txt')).read())
