import json,math
d=json.load(open('../../src/design_eval_si.json'))
r=[x for x in d if x['g']==3 and x['fsr']==3.3 and x['T1q']==300 and x['sc'].startswith('S1')]
def sr(T,rate):  # single-rail Fock qubit, 6-state avg fidelity, loss T1=534, coherence dephasing rate
    eta=math.exp(-T/534); c=math.sqrt(eta)*math.exp(-T*rate)
    return (((1+eta)/2)+2*(0.5+c/2))/3
print('T  Fc  p  Funcb2 | Feff=pF+(1-p)/2  Q=p(2F-1) Qunc | SR(code conv 1/107) SR(corr 1/214) | p*Fc')
for x in r:
    T=x['T'];F=x['F_c'];p=x['p_pass'];U=x['B2_Func']
    print(T,round(F,3),round(p,3),round(U,3),'|',round(p*F+(1-p)*.5,3),round(p*(2*F-1),3),round(2*U-1,3),'|',round(sr(T,1/107),3),round(sr(T,1/214),3),'|',round(p*F,3))
