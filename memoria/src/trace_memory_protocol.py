"""Record the actual F17c load/write/hold/read trajectory on pcjhon.

Instruments the mesolve calls of f17.run without modifying its equations.
One T=2000 trajectory, real 10-kappa loading and readout.
"""
from pathlib import Path
import json
import numpy as np
import qutip as qt
import f17_real_load_reduced as f17
import f9_dual_rail_n3 as f9

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data"/"memory_figs"/"protocol_trace.npz"
original=qt.mesolve
records=[]
stages=["load 1","write 1","bin gap","load 2","write 2","hold","read 2","emit 2","read 1","emit 1"]
nc=f9.NC
R0,R1=qt.basis(2,0),qt.basis(2,1)
PR0=f17.op(R0*R0.dag(),2); PR1=f17.op(R1*R1.dag(),2)
PD1=f17.lift(f9.op(f9.P[2],0)); PD2=f17.lift(f9.op(f9.P[3],0))
nb=f17.lift(f9.bX.dag()*f9.bX)
coh=f17.op(R1*R0.dag(),2)*f17.lift(f9.op(f9.ket4[3]*f9.ket4[2].dag(),0))

def recorder(H,r,tlist,c_ops,*args,**kwargs):
    # Last call (isolated readout-efficiency calibration) has smaller Hilbert space.
    if r.shape[0] != int(np.prod(f17.dims)):
        return original(H,r,tlist,c_ops,*args,**kwargs)
    k=len(records)
    a,b=float(tlist[0]),float(tlist[-1])
    count=61 if k==5 else 31
    grid=np.linspace(a,b,count)
    result=original(H,r,grid,c_ops,*args,**kwargs)
    states=result.states
    vals=lambda op: np.real(np.asarray([qt.expect(op,s) for s in states]))
    c=np.asarray([qt.expect(coh,s) for s in states])
    records.append(dict(stage=stages[k] if k<len(stages) else f"stage {k}",
                        t=grid,mode=vals(nb),d1=vals(PR0*PD1)*2,
                        d2=vals(PR1*PD2)*2,coherence=2*np.abs(c),
                        phase=np.unwrap(np.angle(c))))
    return result

qt.mesolve=recorder
try:
    result=f17.run(2000,"real",kin_frac=10,chi=1,tau12=100,
                   readout="real",kout_frac=10)
finally:
    qt.mesolve=original

time=[]; mode=[]; d1=[]; d2=[]; coherence=[]; phase=[]; intervals=[]
offset=0.0
for row in records:
    tt=row["t"]+offset
    for target,key in ((time,"t"),(mode,"mode"),(d1,"d1"),(d2,"d2"),
                       (coherence,"coherence"),(phase,"phase")):
        target.extend(tt if key=="t" else row[key])
    intervals.append([row["stage"],offset,float(tt[-1])])
    offset=float(tt[-1])
np.savez_compressed(OUT,t=time,mode=mode,d1=d1,d2=d2,coherence=coherence,
                    phase=phase,stages=np.array(intervals,dtype=str),
                    result=json.dumps({k:result[k] for k in ("F","s","F_h")}))
print(json.dumps({"out":str(OUT),"stages":intervals,
                  "F":result["F"],"s":result["s"],"F_h":result["F_h"]}),flush=True)
