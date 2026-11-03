import numpy as np, mem
from mem import *
S3 = Sys(h3(), [.2, .14, 0], slots=[1, 0], chi=1.0)
print('paper pulses closed transfer:', S3.transfer(0, (.197,170,-.02)), S3.transfer(1, (.29,150,-.025)))
for k, g in ((0,(.197,170,-.02)),(1,(.29,150,-.025))):
    print(k, S3.optimize(k, g))
