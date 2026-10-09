import csv,re,collections,sys,os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows=list(csv.DictReader(open(os.path.join(ROOT,'consolidacion','afirmaciones.csv'),encoding='utf-8')))
cov=collections.defaultdict(set)
for r in rows:
    for u in r['ubicaciones'].split(' | '):
        if ':' in u:
            f,l=u.rsplit(':',1)
            if l.isdigit() and int(l)>0: cov[f].add(int(l))
L1=['HANDOFF.md','README.md','Mechanical-Cat-State/figuras_finales/README.md','Mechanical-Cat-State/figuras_finales/PENDIENTES_Y_HALLAZGOS.md','Mechanical-Cat-State/figuras_finales/C6_polaron.md','Mechanical-Cat-State/figuras_finales/data/preregistro_kf0.2.md','docs/Informe/informe.tex']
def sust(l):
    t=re.sub(r'`[^`]*`','',l); t=re.sub(r'§\s?\d+(\.\d+)?','',t); t=re.sub(r'\b[PECV]\d+(b)?\b','',t); t=re.sub(r'Tareas? \d+(\s*[-–a y]\s*\d+)?','',t); t=re.sub(r'\d{4}-\d\d-\d\d','',t)
    t=re.sub(r'\\label\{[^}]*\}|\\ref\{[^}]*\}|\\eqref\{[^}]*\}|\\cite\{[^}]*\}|\\includegraphics.*','',t)
    return bool(re.search(r'\d+\.\d+|\de-\d|\d\s?%|\d+ de \d+|[NK]\s*=\s*\d+|\d+\s*(GB|s\b|min|kB|MHz|GHz|Hz|mK)|×10|10\^|\\times|\d{3,}',t))
tot=0
for f in L1:
    L=open(os.path.join(ROOT,f)).read().split('\n')
    unc=[i for i,l in enumerate(L,1) if i not in cov[f] and not l.startswith('#') and sust(l) and len(l)>15]
    tot+=len(unc)
    print(f,len(unc))
    if len(sys.argv)>1 and sys.argv[1]==f: 
        for i in unc: print(i,L[i-1][:200])
print(tot)
