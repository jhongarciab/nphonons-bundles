#!/usr/bin/env python3
"""Fase C-prep: clasifica todo msc/ en nucleo/entregable/cadena/historico.
No mueve nada. Escribe consolidacion/clasificacion.csv y plan_archivo.csv."""
import os, re, csv, glob, fnmatch, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCL = {'.venv', '.venv_qutip5', '.git.bak', '.git', '__pycache__', 'consolidacion'}

files = []
for d, ds, fs in os.walk(ROOT):
    ds[:] = [x for x in ds if x not in EXCL]
    for f in fs:
        files.append(os.path.relpath(os.path.join(d, f), ROOT))
files.sort()

NUCLEO = {
    'HANDOFF.md': 'nucleo registro (indice de secciones; sera reemplazado por el nuevo HANDOFF.md)',
    'README.md': 'nucleo registro (mapa de carpetas)',
    'Mechanical-Cat-State/figuras_finales/README.md': 'nucleo registro (metodos, parametros y validaciones de cada figura)',
    'Mechanical-Cat-State/figuras_finales/PENDIENTES_Y_HALLAZGOS.md': 'nucleo registro (pendientes P1-P10, errores E1-E8)',
    'Mechanical-Cat-State/figuras_finales/C6_polaron.md': 'nucleo registro (base polaronica)',
    'docs/Informe/informe.tex': 'nucleo registro (borrador del paper)',
    'docs/Informe/refs_informe.bib': 'nucleo registro (bibliografia del informe)',
}

ENTREGABLE_DIRS = ['docs/Informe/figuras', 'Mechanical-Cat-State/figuras_finales/figuras']
entregable = set()
for d in ENTREGABLE_DIRS:
    for f in files:
        if f.startswith(d + os.sep):
            entregable.add(f)
entregable.add('docs/Informe/informe.pdf')  # PDF compilado final (entregable, no solo el .tex)

# ---------- cierre transitivo (cadena): desde entregables + informe.tex, resolver referencias de texto/scripts (incluye patrones de carpeta)
textos = {}
for p in files:
    e = os.path.splitext(p)[1].lower()
    if e in ('.py', '.sh', '.md', '.tex') and os.path.getsize(os.path.join(ROOT, p)) < 3_000_000:
        textos[p] = open(os.path.join(ROOT, p), errors='ignore').read()

ext_tok = r'(?:npz|py|sh|md|tex|pdf|png|csv|log|txt|bib)'
tok_re = re.compile(r'[A-Za-z0-9_\-\./\\{}\*]+\.' + ext_tok + r'\b')
byname = collections.defaultdict(list)
bystem = collections.defaultdict(list)
dirs = collections.defaultdict(list)
for p in files:
    byname[os.path.basename(p)].append(p)
    if p.endswith(('.pdf', '.png')):
        bystem[os.path.splitext(os.path.basename(p))[0]].append(p)
    dirs[os.path.dirname(p)].append(p)

def refs_from(path, txt):
    """Devuelve los archivos que `path` (un script/md/tex) referencia, incluyendo patrones de directorio."""
    out = set()
    e = os.path.splitext(path)[1].lower()
    if e == '.tex':
        for m in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', txt):
            b = os.path.basename(m)
            out.update(bystem.get(b, []))
            out.update(byname.get(b, []))
        for m in re.findall(r'\\bibliography\{([^}]+)\}', txt):
            out.update(byname.get(m + '.bib', []))
    for m in set(tok_re.findall(txt)):
        b = os.path.basename(m.replace('\\_', '_'))
        if '*' in b or '{' in b:
            pre = re.split(r'[\*{]', b)[0]
            for n, ps in byname.items():
                if pre and n.startswith(pre):
                    out.update(ps)
            continue
        out.update(byname.get(b, []))
    # patrones de carpeta: si el texto nombra el directorio base de datos (termico, pfig2, minimo, ...)
    GENERIC = {'data', 'codigo', 'figuras', 'pdf', 'fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'validacion',
               'docs', 'papers', 'figs', '.', 'Mechanical-Cat-State'}
    for d, fl in dirs.items():
        base = os.path.basename(d)
        if not base or base in GENERIC or len(base) < 5 or os.path.dirname(path) == d:
            continue
        if not d.startswith('Mechanical-Cat-State/figuras_finales/'):
            continue
        if re.search(r'(?:^|[^\w])' + re.escape(base) + r'(?:$|[^\w])', txt):
            out.update(fl)
    return out

# import <modulo> dentro de figuras_finales/codigo -> modulo.py
PYDIR = 'Mechanical-Cat-State/figuras_finales/codigo'
def py_imports(path, txt):
    out = set()
    if os.path.dirname(path) != PYDIR:
        return out
    for m in re.findall(r'^\s*import\s+([a-zA-Z_][\w]*)|^\s*from\s+([a-zA-Z_][\w]*)\s+import', txt, re.M):
        mod = m[0] or m[1]
        cand = os.path.join(PYDIR, mod + '.py')
        if cand in textos:
            out.add(cand)
    return out

# Cadena: "codigo oficial del paper" (README raiz) = todo Mechanical-Cat-State/figuras_finales/
# (codigo/, data/) salvo lo ya clasificado como nucleo o entregable. Se complementa con el cierre
# transitivo de referencias desde informe.tex (figuras_informe.py y lo que este importa/lee),
# que vive fuera de figuras_finales.
FF = 'Mechanical-Cat-State/figuras_finales/'
chain = {p for p in files if p.startswith(FF)} - entregable - set(NUCLEO)

seeds = list(entregable) + ['docs/Informe/informe.tex']
frontier = set(seeds); seen = set(seeds) | chain
while frontier:
    nxt = set()
    for p in frontier:
        txt = textos.get(p)
        if txt is None:
            continue
        rs = refs_from(p, txt) | py_imports(p, txt)
        for r in rs:
            if r not in seen:
                seen.add(r)
                chain.add(r)
                nxt.add(r)
    frontier = nxt
chain -= entregable
chain -= set(NUCLEO)

rows = []
for p in files:
    full = os.path.join(ROOT, p)
    size = os.path.getsize(full)
    if p in NUCLEO:
        cat, motivo = 'nucleo', NUCLEO[p]
    elif p in entregable:
        cat, motivo = 'entregable', 'figura/tabla final (docs/Informe/figuras o figuras_finales/figuras)'
    elif p in chain:
        cat, motivo = 'cadena', 'referenciado (directo o por patron de carpeta) desde un entregable o desde informe.tex'
    else:
        cat, motivo = 'historico', ''
    rows.append((p, cat, motivo, size))

with open(os.path.join(ROOT, 'consolidacion', 'clasificacion.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['ruta', 'categoria', 'motivo', 'bytes'])
    for r in rows:
        w.writerow(r)

c = collections.Counter(r[1] for r in rows)
sz = collections.Counter()
for r in rows:
    sz[r[1]] += r[3]
print('conteos:', dict(c))
print('bytes:', {k: f'{v/1e6:.1f} MB' for k, v in sz.items()})

# dudosos: archivos de tipo reporte/script de 1er nivel en carpetas raras que no cayeron en nucleo/entregable/cadena
dudoso = []
for p, cat, motivo, size in rows:
    if cat == 'historico':
        if p.endswith(('.md', '.tex')) and 'tareas_naseem' not in p and 'tareas_ma' not in p and 'independiente' not in p and 'codigo_original' not in p and not p.startswith('docs/'):
            dudoso.append(p)
print('dudosos (%d):' % len(dudoso))
for d in dudoso:
    print(' ', d)
