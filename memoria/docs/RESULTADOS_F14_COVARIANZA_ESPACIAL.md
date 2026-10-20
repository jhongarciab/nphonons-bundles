# F14. Matriz espacial de desfase para tres puntos apilados

## Qué se reemplaza

F10–F13 usaron la matriz adimensional

\[
K_{ij}=(1-\chi)\delta_{ij}+\chi,
\qquad \langle\xi_i(t)\xi_j(t')\rangle\propto\gamma_\phi K_{ij}\delta(t-t').
\]

Los tres pares tienen la misma correlación. Para posiciones
\(z=(0,d,2d)\), los pares vecinos están a \(d\) y el par extremo a \(2d\).
F14 sustituye esa matriz por

\[
K=\begin{pmatrix}1&a&b\\a&1&a\\b&a&1\end{pmatrix},
\quad a=k(d),\quad b=k(2d),
\]

con los núcleos positivos \(k(r)=\exp[-r^2/(2\ell^2)]\) (gaussiano)
y \(k(r)=\exp(-r/\ell)\) (exponencial). Son **modelos de sensibilidad**,
no una longitud de correlación medida para esta plataforma.
Los saltos Lindblad son los autovectores de \(\gamma_\phi K\):
\(L_\alpha=\sqrt{\gamma_\phi\lambda_\alpha}
\sum_i u_{i\alpha}n_i\). Así se conserva positividad completa.

En la base oscura ideal usada por F13,
\(w_i=|D_{1i}|^2-|D_{2i}|^2=(-2,1,1)/3\). La varianza de la
desintonía diferencial es proporcional a

\[
w^T K w=(6-2a-4b)/9.
\]

Para este observable, un \(\chi\) uniforme equivalente es
\(\chi_{\rm ef}=(a+2b)/3\). Esto no sustituye la dinámica completa:
la matriz también puede inducir mezcla entre los excitones vestidos.
Una correlación fuerte entre vecinos puede ser insuficiente si la de
los extremos es débil.

Con \(d\simeq8.3\) nm de la estimación dipolar de F13, para
\(\chi_{\rm ef}\gtrsim0.9\) se requiere aproximadamente
\(\ell\gtrsim4d\simeq33\) nm para el núcleo gaussiano y
\(\ell\gtrsim16d\simeq133\) nm para el exponencial. Es una
condición sobre el modelo espacial, no una estimación material.

## Dinámica numérica

`src/f14_spatial_noise_model.py` extiende el canal F9 con cualquier
matriz de covarianza positiva 3×3. `src/f14_spatial_covariance.py`
usa los pulsos y tasas de F13 y ejecuta el barrido en `pcjhon`.
La corrida en `pcjhon` terminó y se sincronizó a
`data/f14_spatial_covariance.json` (32 puntos). Los extremos reproducen
F13: ruido local \(\ell=0\) da F=0.448, s=0.680, F|h=0.659 a
T=2000; ruido perfectamente común \(\ell=\infty\) da 0.753, 0.808,
0.932. A T=8000, F|h es 0.332 y 0.897 respectivamente.

| Núcleo | \(\ell/d\) | F|h, T=2000 | F|h, T=8000 | s, T=2000 |
|---|---:|---:|---:|---:|
| gaussiano | 1 | 0.773 | 0.477 | 0.730 |
| gaussiano | 2 | 0.868 | 0.689 | 0.776 |
| gaussiano | 4 | **0.913** | **0.829** | 0.799 |
| gaussiano | 8 | 0.927 | 0.878 | 0.806 |
| exponencial | 4 | 0.847 | 0.627 | 0.765 |
| exponencial | 8 | 0.885 | 0.733 | 0.784 |
| exponencial | 16 | **0.907** | **0.805** | 0.795 |

En ambos núcleos, la condición aproximada \(\chi_{\rm ef}\gtrsim0.9\)
predice bien el umbral empírico F|h≈0.9 a T=2000 y ≈0.8 a T=8000.
Incluso con ruido local la fidelidad incondicional F=0.448 a T=2000
supera el F=0.081 del pasivo de F13, pero el estado condicionado tiene
calidad mucho menor (0.659). Por eso la afirmación fuerte de doble carril
coherente requiere correlación espacial larga, no solo una mejora de tasa.

## Límites

- Se supone igual amplitud de Stark por punto. Una fluctuación perfectamente
  espacialmente común aún puede ser diferencial si las sensibilidades
  excitónicas no son iguales: en ese caso \(K_{ij}\) adquiere factores
  \(\alpha_i\alpha_j\).
- El ruido blanco markoviano de F10 no equivale automáticamente a ruido de
  carga lento ni al baño LA superóhmico. Debe especificarse qué proceso
  representa \(\gamma_\phi\) antes de asignar \(\ell\) físico.
- Se heredan del modelo F9 el cambio de marco rotante entre láseres y la
  base electrónica oscura fija al variar brechas. Esta prueba aísla la
  hipótesis sobre correlaciones de ruido; no resuelve esas auditorías.

## F14b. Amplitudes Stark desiguales con campo común

Corrida finalizada en `pcjhon`: `src/f14b_stark_mismatch.py`, datos
`data/f14b_stark_mismatch.json`. Se usó \(K=\alpha\alpha^T\), es decir,
una fluctuación espacial exactamente compartida con distinta respuesta
por punto. Pulsos, tasas y pérdidas son los de F13 a 100 GHz.

| \(\alpha_1,\alpha_2,\alpha_3\) | F|h, T=2000 | F|h, T=8000 | s, T=8000 |
|---|---:|---:|---:|
| 1, 1, 1 | 0.932 | 0.897 | 0.805 |
| 1.1, 1, 1 | 0.931 | 0.893 | 0.803 |
| 1.2, 1, 1 | 0.928 | 0.882 | 0.797 |
| 1, 1.1, 1 | 0.927 | 0.889 | 0.803 |
| 1, 0.9, 1.1 | 0.928 | 0.883 | 0.800 |

Una desigualdad del 10–20 % reduce F|h solo 0.004–0.015 a T=8000
en los patrones probados. Así, el requisito exigente de F14 es la
**correlación espacial del campo**, no una igualdad perfecta de la
respuesta Stark. No se han explorado desbalances mayores ni ruido lento
no markoviano.
