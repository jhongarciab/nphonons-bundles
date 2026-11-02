import sys,urllib.request,urllib.parse,xml.etree.ElementTree as ET
ns={'a':'http://www.w3.org/2005/Atom','r':'http://arxiv.org/schemas/atom'}
def q(query,n=8,abs_chars=0):
    u='https://export.arxiv.org/api/query?'+urllib.parse.urlencode({'search_query':query,'max_results':n,'sortBy':'submittedDate','sortOrder':'descending'})
    d=urllib.request.urlopen(u,timeout=40).read()
    for e in ET.fromstring(d).findall('a:entry',ns):
        t=' '.join(e.find('a:title',ns).text.split()); i=e.find('a:id',ns).text.split('/abs/')[1]
        dt=e.find('a:published',ns).text[:10]
        j=e.find('r:journal_ref',ns); j=j.text if j is not None else '-'
        print(f"{dt} {i} | {t} | {j}")
        if abs_chars: print('    ',' '.join(e.find('a:summary',ns).text.split())[:abs_chars])
if __name__=='__main__':
    q(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]) if len(sys.argv)>3 else 0)
