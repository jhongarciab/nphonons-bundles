"""Fig. 2 (cálculo pesado): estado estacionario de Floquet del modelo completo de Ma en un punto (gx, wp, wq).
Marco de laboratorio, propagador de un período T_p = 2π/wp, estado estroboscópico t = nT_p (autovector λ=1).
Guarda en data/fig2/<clave>.npz: ρ, P_c (código nuevo y viejo), α_eff², P_e, validaciones.
  código nuevo (C6): {D(d)|±α_eff>}, d = +gz/w, α_eff² = ⟨(a-d)²⟩ medido
  código viejo:      {|±α>}, α² = Ω/G nominal, sin desplazamiento
Uso: python calc_fig2.py gx wp wq Om N [--rerun]    (g_z, w, κ fijos: Ma)
"""
import sys, os
import numpy as np
import comun as C

W, GZ, KAP = 6.0, 0.3 * np.cos(np.pi / 4), 0.03


def clave(gx, wp, wq, Om, N):
    return os.path.join(C.DATA, 'fig2', f'gx{gx:.6f}_wp{wp:.6f}_wq{wq:.6f}_Om{Om:.6f}_N{N}.npz')


def punto(gx, wp, wq, Om, N, rerun=False):
    f = clave(gx, wp, wq, Om, N)
    if os.path.exists(f) and not rerun:
        return dict(np.load(f))
    U, tprop = C.propagador(N, W, wq, gx, GZ, Om, wp, KAP)
    M, lam = C.estacionario(U, N)
    a, sm, sz, sx = C.ops(N)
    d = GZ / W
    A = a.full() - d * np.eye(2 * N)
    al2eff = np.trace(A @ A @ M)
    G = 2 * gx * GZ / W
    al_nom = np.sqrt(complex(Om / G))
    res = dict(gx=gx, wp=wp, wq=wq, Om=Om, N=N, G=G, d=d, al2eff=al2eff,
               Pc_nuevo=C.Pc(M, N, np.sqrt(al2eff), d), Pc_viejo=C.Pc(M, N, al_nom, 0.0),
               Pe=np.real(np.trace((sm.dag() * sm).full() @ M)), a2=np.trace((a * a).full() @ M),
               val=np.array(C.validar(M)), tprop=tprop, rho=M)
    # C5: P_e y paridad promediados sobre un período (invariantes al pasar al marco rotante)
    import qutip as qt
    H = C.hamiltoniano(N, W, wq, gx, GZ, Om, wp)
    ts = np.linspace(0, 2 * np.pi / wp, 41)
    par = qt.tensor((1j * np.pi * qt.num(N)).expm(), qt.qeye(2))
    r = qt.mesolve(H, qt.Qobj(M, dims=[[N, 2], [N, 2]]), ts, [np.sqrt(KAP) * sm],
                   e_ops={'Pe': sm.dag() * sm, 'par': par}, options=C.OPTS)
    res['Pe_prom'] = np.mean(np.real(r.e_data['Pe'][:-1])); res['par_prom'] = np.mean(np.real(r.e_data['par'][:-1]))
    res['Pe_rango'] = np.array([np.min(np.real(r.e_data['Pe'])), np.max(np.real(r.e_data['Pe']))])
    np.savez(f, **res)
    return res


if __name__ == '__main__':
    gx, wp, wq, Om, N = map(float, sys.argv[1:6])
    r = punto(gx, wp, wq, Om, int(N), '--rerun' in sys.argv)
    print(f"gx={gx} wp={wp} wq={wq} N={int(N)} Pc_nuevo={r['Pc_nuevo']:.6f} Pc_viejo={r['Pc_viejo']:.6f} "
          f"α_eff²={complex(r['al2eff']):.4f} Pe={float(r['Pe']):.5f} val={r['val']} t={float(r['tprop']):.0f}s")
