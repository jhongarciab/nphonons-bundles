"""Fase A: inventario de msc/. Solo lectura. Salida: consolidacion/inventario.csv"""
import os, re, csv, hashlib, collections, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCL = {'.venv', '.venv_qutip5', '.git.bak', '.git', '__pycache__', 'consolidacion'}
files = []
for d, ds, fs in os.walk(ROOT):
    ds[:] = [x for x in ds if x not in EXCL]
    for f in fs:
        files.append(os.path.relpath(os.path.join(d, f), ROOT))
files.sort()
tracked = set(subprocess.run(['git', 'ls-files', '.'], cwd=ROOT, capture_output=True, text=True).stdout.split('\n'))
# git ls-files . da rutas relativas a cwd
def clasif(p):
    e = os.path.splitext(p)[1].lower()
    if e in ('.npz', '.csv', '.json', '.npy', '.pkl'): return 'dato'
    if e in ('.py', '.sh'): return 'script'
    if e in ('.png', '.pdf') and 'papers' not in p: return 'figura'
    if e == '.md': return 'reporte_md'
    if e == '.tex': return 'tex'
    if e in ('.log', '.txt'): return 'log'
    return 'otro'
TEXT = {'.md', '.tex', '.py', '.sh', '.bib', '.txt', '.log', '.csv', '.gitignore'}
def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()[:12]
byname = collections.defaultdict(list)
bystem = collections.defaultdict(list)
for p in files:
    if p.endswith(('.pdf', '.png')): bystem[os.path.splitext(os.path.basename(p))[0]].append(p)
for p in files: byname[os.path.basename(p)].append(p)
EXT = r'(?:npz|py|sh|md|tex|pdf|png|csv|log|txt|bib)'
tok = re.compile(r'[A-Za-z0-9_\-\.\/\\{}\*]+\.' + EXT + r'\b')
refs = collections.defaultdict(set)   # destino -> set de fuentes
broken = []
for p in files:
    e = os.path.splitext(p)[1].lower()
    if e not in TEXT: continue
    if os.path.getsize(os.path.join(ROOT, p)) > 5_000_000: continue
    try: txt = open(os.path.join(ROOT, p), errors='ignore').read()
    except Exception: continue
    if e == '.tex':   # \includegraphics sin extension
        for m in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', txt):
            for q in bystem.get(os.path.basename(m), []) + byname.get(os.path.basename(m), []):
                if q != p: refs[q].add(p)
        # fuente .tex -> su pdf
    if e == '.tex':
        for q in bystem.get(os.path.splitext(os.path.basename(p))[0], []):
            if q.endswith('.pdf'): refs[q].add(p)
    for m in set(tok.findall(txt)):
        t = m.replace('\\_', '_')
        b = os.path.basename(t)
        if any(c in b for c in '*{}'):   # patrones glob: marcar como refs por prefijo
            pre = re.split(r'[\*{]', b)[0]
            if pre:
                for n, ps in byname.items():
                    if n.startswith(pre):
                        for q in ps:
                            if q != p: refs[q].add(p)
            continue
        if b in byname:
            for q in byname[b]:
                if q != p: refs[q].add(p)
        elif len(b) > 4 and not b.startswith('.') and b not in ('np.log', '-np.log'):
            broken.append((p, t))
with open(os.path.join(ROOT, 'consolidacion', 'inventario.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['ruta', 'tipo', 'ext', 'bytes', 'sha256_12', 'en_git', 'n_refs_entrantes', 'referenciado_por(max5)', 'huerfano'])
    for p in files:
        r = sorted(refs.get(p, ()))
        w.writerow([p, clasif(p), os.path.splitext(p)[1], os.path.getsize(os.path.join(ROOT, p)), sha(p),
                    int(p in tracked or ('msc/' + p) in tracked), len(r), ';'.join(r[:5]), int(len(r) == 0)])
with open(os.path.join(ROOT, 'consolidacion', 'referencias_rotas.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['archivo_fuente', 'referencia_rota'])
    for r in sorted(set(broken)): w.writerow(r)
print(len(files), 'archivos;', len(set(broken)), 'refs rotas (cand.)')
