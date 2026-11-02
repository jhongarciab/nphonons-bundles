import numpy as np
r=np.load('bars_px.npy')[1:]
x0,x60=249.5,1280.0; y0,y1=743.0,46.0
X=lambda x:(x-x0)/(x60-x0)*60; Y=lambda y:(y0-y)/(y0-y1)
mine=np.array([[X(a),Y(b),Y(ctop)-Y(b),Y(b)-Y(cbot)] for a,b,s,ctop,cbot,_,_ in r])
np.set_printoptions(precision=4,suppress=True)
ref=np.load('../../phi_hitchcock_fig4e.npy')
print(np.c_[mine,ref])
d=mine-ref
for i,n in enumerate(['dw','mean','+err','-err']): print(n,'max|d|=%.4f rms=%.4f mean d=%.4f'%(abs(d[:,i]).max(),np.sqrt((d[:,i]**2).mean()),d[:,i].mean()))
print(mine[0])
# robust err = mean of symmetric
