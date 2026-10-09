#!/usr/bin/env python3
"""Fase C-prep: propone destino de archivado para cada .md (salvo HANDOFF_nuevo.md) y lista
que referencias (de otros .md/.tex/.py/.sh) se romperian. No ejecuta ningun git mv."""
import os, re, csv, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCL = {'.venv', '.venv_qutip5', '.git.bak', '.git', '__pycache__', 'consolidacion'}

mds = []
for d, ds, fs in os.walk(ROOT):
    ds[:] = [x for x in ds if x not in EXCL]
    for f in fs:
        if f.endswith('.md'):
            mds.append(os.path.relpath(os.path.join(d, f), ROOT))
mds.sort()

# todos los archivos de texto para buscar quien referencia cada .md
textos = {}
for d, ds, fs in os.walk(ROOT):
    ds[:] = [x for x in ds if x not in EXCL]
    for f in fs:
        if f.endswith(('.py', '.sh', '.md', '.tex')):
            p = os.path.relpath(os.path.join(d, f), ROOT)
            if os.path.getsize(os.path.join(ROOT, p)) < 3_000_000:
                textos[p] = open(os.path.join(ROOT, p), errors='ignore').read()

rows = []
for m in mds:
    if m == 'HANDOFF.md':
        continue  # se intercambia por HANDOFF_nuevo.md, no se archiva bajo _archivo/
    dest = os.path.join('_archivo', m)
    refs = sorted(p for p, t in textos.items() if p != m and os.path.basename(m) in t)
    rows.append((m, dest, len(refs), ';'.join(refs[:10]) + ('...' if len(refs) > 10 else '')))

with open(os.path.join(ROOT, 'consolidacion', 'plan_archivo.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['ruta_md', 'destino_propuesto', 'n_referencias_rotas', 'referenciado_por(max10)'])
    for r in rows:
        w.writerow(r)

print('total .md a mover (excepto HANDOFF.md -> reemplazado):', len(rows))
con_refs = [r for r in rows if r[2] > 0]
print('con referencias que se romperian:', len(con_refs))
for r in con_refs:
    print(' ', r[0], '->', r[2], 'refs:', r[3][:160])
