# Derivacion simbolica: probabilidad de que |e,0> termine en un par (via g2 -> clicks) frente a perdida por kappa_q. V=sqrt(2) G (|e0>,|g2>), amplitud de g2 decae kappa_m.
import sympy as sp
G,km,kq=sp.symbols('G kappa_m kappa_q',positive=True); V=sp.sqrt(2)*G
M=sp.Matrix([[-kq/2,-sp.I*V],[-sp.I*V,-km]])
A,C=sp.symbols('A C',real=True)
Br,Bi=sp.symbols('Br Bi',real=True); Bc=Br+sp.I*Bi
X=sp.Matrix([[A,Bc],[sp.conjugate(Bc),C]])
eq=(M*X+X*M.H+sp.Matrix([[1,0],[0,0]])).applyfunc(sp.expand)
sol=sp.solve([sp.re(eq[0,0]),sp.re(eq[1,1]),sp.re(eq[0,1]),sp.im(eq[0,1])],[A,C,Br,Bi],dict=True)[0]
eta=sp.simplify(2*km*sol[C]); print('eta =',sp.factor(eta))
print('1/eta-1 =',sp.factor(sp.simplify(1/eta-1)))
print('km=2sqrt2 G:',sp.simplify(eta.subs(km,2*sp.sqrt(2)*G)))
print('kq_max(eta=0.95)=',[sp.simplify(x) for x in sp.solve(sp.Eq(eta,sp.Rational(19,20)),kq)])
