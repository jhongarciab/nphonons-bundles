import numpy as np
from PIL import Image
from scipy import ndimage as ndi
im=np.array(Image.open('crop-05.png').convert('RGB')).astype(int)
R,G,B=im[...,0],im[...,1],im[...,2]
H,W=R.shape
black=(R<80)&(G<80)&(B<80)
# axes frame: columns / rows with long black runs
cols=black.sum(0); rows=black.sum(1)
print('black cols',[ (i,c) for i,c in enumerate(cols) if c>500][:10])
print('black rows',[ (i,c) for i,c in enumerate(rows) if c>600][:10])
# gridlines: light gray (~ 215-235) rows/cols
gl=(abs(R-G)<6)&(abs(G-B)<6)&(R>200)&(R<236)
gr=gl.sum(1); gc=gl.sum(0)
yr=[i for i in range(H) if gr[i]>500]; xc=[i for i in range(W) if gc[i]>400]
def groups(a):
    out=[];s=a[0];p=a[0]
    for v in a[1:]:
        if v!=p+1: out.append((s+p)/2);s=v
        p=v
    out.append((s+p)/2);return out
print('grid y',groups(yr) if yr else None,'grid x',groups(xc) if xc else None)
red=(R>220)&(G<120)&(B<120)
lab,n=ndi.label(red);print('red comps',n)
# dots: open with disk
rr=np.arange(-7,8);disk=(rr[:,None]**2+rr[None,:]**2<=49)
dots=ndi.binary_opening(red,structure=disk)
lab,n=ndi.label(dots);cs=ndi.center_of_mass(dots,lab,range(1,n+1));sz=ndi.sum(dots,lab,range(1,n+1))
pts=[(c[1],c[0],s) for c,s in zip(cs,sz)]
pts.sort();
for p in pts: print('dot x=%.1f y=%.1f area=%d'%p)
np.save('dots_px.npy',np.array(pts))
# bars
res=[]
for x,y,s in pts:
    xi=int(round(x))
    col=red[:,xi-2:xi+3].any(1)
    ys=np.where(col)[0]
    # cap rows: horizontal runs wide
    capm=red[:, int(x)-25:int(x)+26].sum(1)>=35
    cy=np.where(capm)[0]
    res.append((x,y,s,cy.min() if len(cy) else np.nan,cy.max() if len(cy) else np.nan,ys.min(),ys.max()))
np.save('bars_px.npy',np.array(res))
