# Revisión de F16b: discrepancia 0.878 frente a 0.920

La corrida conjunta del modelo completo terminó con Δxx=−9ω_b,
κ_in=10κ, T=2000/ω_b, χ=1: F=0.693, s=0.753, F|h=0.920.
El reducido F17 con carga real da 0.677, 0.770, 0.878; con carga
ideal ambos dan F|h≈0.93. La existencia del efecto en el completo
es alentadora, pero la discrepancia real requiere un control antes de
fijar una cifra para el manuscrito.

## Qué explicaciones no bastan por sí solas

En el subespacio de cero y un excitón, el salto del completo
\(\sum_{ij}l_i h_j\sigma_i^\dagger\sigma_j\) restringe exactamente a
\(|l\rangle\langle h|\), usado en el reducido. El salto de ruido común
\(\sum_i n_i\) restringe al proyector sobre el sector de un excitón,
igual a \(P_B+P_{D1}+P_{D2}\). Por tanto esas dos diferencias formales
no explican nada si la dinámica permanece en ese sector. El segundo
bin sin η registrada impide diagnosticar el mecanismo, pero no cambia
la simulación.

## Orden de controles

1. Comparar matrices de Hamiltoniano y saltos del completo proyectadas
   en {G,B,D1,D2} con las del reducido (incluida la cascada), a los
   mismos parámetros. Esto es barato y detecta errores de implementación.
2. Registrar en **ambos** modelos η del segundo bin y los tres términos
   de la fidelidad: poblaciones correctas a,b y módulo |c|, al final
   de la carga, después de cada pulso y al final. Localiza el momento
   en que divergen.
3. Solo si la diferencia nace en los pulsos, correr el completo con
   |Δxx| mucho mayor y misma carga real. Si converge a 0.878, el
   desplazamiento finito de niveles es una explicación física; si
   permanece ≈0.92, buscar diferencia de código o truncación.

No reportar 0.88–0.92 como un intervalo de confianza: son salidas de
dos modelos cuya equivalencia en el límite relevante falta establecer.
Una sola cifra de memoria se fija después del control.

La lectura real hacia la guía puede modelarse después. F16b todavía
usa descarga ideal y heraldo por detección del fonón de salida.
