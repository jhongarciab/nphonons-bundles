"""V4b: razones con |α|² nominal vs |⟨a²⟩| estacionario, y fórmula A: 2|α|²(Γ⁻+Γ⁺) vs B: 2[Γ⁻|α|² + Γ⁺(|α|²+1)]."""
import glob, numpy as np
KAP = 0.03
def tasas(gx, w, al2):
    gm = gx**2*KAP/(w**2 + KAP**2/4); gp = gx**2*KAP/(9*w**2 + KAP**2/4)
    return 2*al2*(gm+gp), 2*(gm*al2 + gp*(al2+1))
print("archivo                         |α|²nom |⟨a²⟩|   A(nom)  A(|a2|)  B(nom)  B(|a2|)  P_c")
for f in sorted(glob.glob('res_v4/*.npz')):
    z = np.load(f); gx, w, r = float(z['gx']), float(z['w']), float(z['r_par'])
    al2n = 4.0 if 'b_al' not in f else float(f.split('b_al')[1][0])
    a2 = abs(complex(z['a2_ss']))
    An, Bn = tasas(gx, w, al2n); Aa, Ba = tasas(gx, w, a2)
    print(f"{f[7:]:31s} {al2n:5.1f}  {a2:7.4f}  {r/An:.4f}  {r/Aa:.4f}  {r/Bn:.4f}  {r/Ba:.4f}  {float(z['Pc_ss']):.5f}")
