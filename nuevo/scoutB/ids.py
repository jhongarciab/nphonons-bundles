import sys,urllib.request,xml.etree.ElementTree as ET
ns={'a':'http://www.w3.org/2005/Atom','r':'http://arxiv.org/schemas/atom'}
d=urllib.request.urlopen('https://export.arxiv.org/api/query?id_list='+','.join(sys.argv[1:])+'&max_results=20',timeout=40).read()
for e in ET.fromstring(d).findall('a:entry',ns):
    j=e.find('r:journal_ref',ns)
    print('##',e.find('a:id',ns).text.split('/abs/')[1],' '.join(e.find('a:title',ns).text.split()),'|',j.text if j is not None else '-')
    print(' '.join(e.find('a:summary',ns).text.split())[:900]);print()
