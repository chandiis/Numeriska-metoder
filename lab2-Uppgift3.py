# -*- coding: utf-8 -*-
"""
T2: Temperatur i stav - randvärdesproblem (Numeriska metoder, laboration 2)

Löser:
T2.a  Diskretisering för N=4 (central differens), skriver ut A och HL
T2.b  Allmän diskretisering för N
T2.c  Funktion diskretisering_temperatur som returnerar gles matris A och HL
T2.d  Lös med N=100, plotta T(x), skriv T vid x=0.2
T2.e  Konvergensstudie vid x=0.7 med steglängdshalvering, verifiera ordning ~2
T2.f  Testa TL != TR och notera påverkan

Problem:
k*T''(x) = q(x), 0<x<L
T(0)=TL, T(L)=TR

Default-värden i uppgiften:
L=1, k=2, q(x)=50*x^3*ln(x+1), TL=TR=2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import os

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')
    
# Källterm q(x) i uppgiften
def q_default(x):
    return 50.0 * x**3 * np.log(x + 1.0)

# T2.c: Diskretisering (central differens) -> A (gles), HL
def diskretisering_temperatur(N, q_fun, k, TL, TR, L=1.0):
    """
    Diskretiserar k*T''(x)=q(x) på (0,L) med Dirichlet randvillkor T(0)=TL, T(L)=TR
    med centrala differenser.

    N = antal delintervall => h = L/N
    Okända: T_1,...,T_{N-1} (inre punkter)

    Returnerar:
      A  : (N-1)x(N-1) gles CSR-matris (tridiagonal)
      HL : högerled (N-1)-vektor inkl randbidrag
      x  : nodvektor x_0..x_N (längd N+1)
      h  : steglängd
    """
    if N < 2:
        raise ValueError("N måste vara minst 2 (så att det finns inre punkter).")

    h = L / N
    x = np.linspace(0.0, L, N + 1)           # x_0..x_N
    xi = x[1:-1]                              # inre noder x_1..x_{N-1}
    m = N - 1                                 # antal okända

    alpha = k / h**2

    # Tridiagonal matris: alpha * tridiag(1, -2, 1)
    main = (-2.0 * alpha) * np.ones(m)
    off  = ( 1.0 * alpha) * np.ones(m - 1)
    A = diags([off, main, off], offsets=[-1, 0, 1], format="csr")

    # Högerled: q(x_i)
    HL = q_fun(xi).astype(float)

    # Randvillkor påverkar första och sista ekvationen:
    # alpha*(T_0 -2T_1 + T_2) = q(x_1) => HL_1 -= alpha*T_0
    # alpha*(T_{N-2} -2T_{N-1} + T_N) = q(x_{N-1}) => HL_{N-1} -= alpha*T_N
    HL[0]  -= alpha * TL
    HL[-1] -= alpha * TR

    return A, HL, x, h

# Hjälpfunktion: Lös och bygg T
def los_temperatur(N, q_fun, k, TL, TR, L=1.0):
    """
    Löser randvärdesproblemet numeriskt med N delintervall.
    Returnerar full temperaturvektor T vid alla noder (längd N+1) och x.
    """
    A, HL, x, h = diskretisering_temperatur(N, q_fun, k, TL, TR, L=L)
    Ti = spsolve(A, HL)            # inre temperaturer T_1..T_{N-1}

    T = np.zeros(N + 1, dtype=float)
    T[0] = TL
    T[-1] = TR
    T[1:-1] = Ti
    return x, T

# T2.a: visa N=4 systemet
def skriv_ut_T2a_N4(q_fun, k, TL, TR, L=1.0):
    N = 4
    A, HL, x, h = diskretisering_temperatur(N, q_fun, k, TL, TR, L=L)

    print("\n=== T2.a: N=4 ===")
    print(f"L={L}, N={N}, h={h}, k={k}")
    print("Noder x_j:", x)

    print("\nSystemmatris A (som tät matris för utskrift):")
    print(A.toarray())

    print("\nHögerled HL:")
    print(HL)

    # Kontroll jämfört med uppgifts-expected form: A = 32 * [[-2,1,0],[1,-2,1],[0,1,-2]]
    alpha = k / h**2
    print(f"\nalpha = k/h^2 = {alpha} (ska bli 32 för L=1,k=2,N=4)")

    # Visa även HL-komponenter med mer info:
    xi = x[1:-1]
    qvals = q_fun(xi)
    print("\nInre punkter x_1..x_{N-1} =", xi)
    print("q(x_i) =", qvals)
    print("HL[0] = q(x1) - alpha*TL =", qvals[0] - alpha*TL)
    print("HL[1] = q(x2)           =", qvals[1])
    print("HL[2] = q(x3) - alpha*TR =", qvals[2] - alpha*TR)



# T2.d: N=100 + plot + T(0.2)
def t2d_N100_och_plot(q_fun, k, TL, TR, L=1.0):
    N = 100
    x, T = los_temperatur(N, q_fun, k, TL, TR, L=L)

    # x=0.2 ligger exakt på nod om h=0.01 => j=20
    j = int(round(0.2 / (L / N)))
    T_x02 = T[j]

    print("\n=== T2.d: N=100 ===")
    print(f"T(0.2) approx = {T_x02:.10f} (nod j={j}, x_j={x[j]:.3f})")

    plt.figure()
    plt.plot(x, T, marker=".", linewidth=1.0)
    plt.xlabel("x")
    plt.ylabel("T(x)")
    plt.title("Temperaturfördelning i staven (N=100)")
    plt.grid(True)
    plt.show()

    return T_x02

# T2.e: Konvergensstudie vid x=0.7 (N=50->)
def t2e_konvergensstudie_x07(q_fun, k, TL, TR, L=1.0, N_start=50, steg=5):
    """
    Dubblar N steg gånger: N_start, 2N_start, 4N_start, ...
    Mäter T vid x=0.7 (som ska ligga på nod om N multipel av 10)
    och skattar ordning p ~ 2.
    """
    x_mal = 0.7
    Ns = [N_start * (2**i) for i in range(steg)]
    Ts = []

    print("\n=== T2.e: Konvergensstudie vid x=0.7 ===")
    for N in Ns:
        x, T = los_temperatur(N, q_fun, k, TL, TR, L=L)
        h = L / N

        # Välj nodindex så att x_j = 0.7 exakt om möjligt.
        j = int(round(x_mal / h))
        xm = x[j]
        if abs(xm - x_mal) > 1e-12:
            # Om inte exakt, varna och ändå ta närmaste nod
            print(f"VARNING: N={N} ger inte exakt x=0.7 (närmaste x_j={xm}).")

        Ts.append(T[j])
        print(f"N={N:5d}, h={h:.6f}, x_j={xm:.6f}, T(x≈0.7)={T[j]:.10f}")

    # Skatta ordning p med tre på varandra följande lösningar:
    # p ≈ log2( |T_N - T_2N| / |T_2N - T_4N| )
    print("\nSkattad ordning p (bör bli ~2):")
    for i in range(len(Ts) - 2):
        e1 = abs(Ts[i] - Ts[i + 1])
        e2 = abs(Ts[i + 1] - Ts[i + 2])
        if e2 == 0:
            p = np.nan
        else:
            p = np.log(e1 / e2) / np.log(2.0)
        print(f"p mellan N={Ns[i]}->{Ns[i+1]}->{Ns[i+2]}: {p:.4f}")

    return Ns, Ts

# T2.f: testa TL != TR och plotta
def t2f_testa_olika_randvillkor(q_fun, k, L=1.0):
    N = 200

    TL1, TR1 = 2.0, 2.0
    TL2, TR2 = 2.0, 5.0

    x1, T1 = los_temperatur(N, q_fun, k, TL1, TR1, L=L)
    x2, T2 = los_temperatur(N, q_fun, k, TL2, TR2, L=L)

    print("\n=== T2.f: Testa TL != TR ===")
    print(f"Fall A: TL={TL1}, TR={TR1}")
    print(f"Fall B: TL={TL2}, TR={TR2}")
    print("Notera: TL != TR ger en tydlig global lutning/tilt i hela profilen.")

    plt.figure()
    plt.plot(x1, T1, linewidth=1.5, label=f"TL={TL1}, TR={TR1}")
    plt.plot(x2, T2, linewidth=1.5, label=f"TL={TL2}, TR={TR2}")
    plt.xlabel("x")
    plt.ylabel("T(x)")
    plt.title("Påverkan av randvillkor (N=200)")
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    clear_console()
    # Parametrar enligt uppgiften
    L = 1.0
    k = 2.0
    TL = 2.0
    TR = 2.0
    q_fun = q_default

    # T2.a (inkl kontroll av A och HL för N=4)
    skriv_ut_T2a_N4(q_fun, k, TL, TR, L=L)

    # T2.d (N=100, plot + T(0.2))
    t2d_N100_och_plot(q_fun, k, TL, TR, L=L)

    # T2.e (konvergens vid x=0.7)
    Ns, Ts = t2e_konvergensstudie_x07(q_fun, k, TL, TR, L=L, N_start=50, steg=5)
    print("\nUppgiftens riktvärde: T(0.7) bör konvergera mot cirka 1.6379544")

    # T2.f (testa TL != TR)
    t2f_testa_olika_randvillkor(q_fun, k, L=L)
    
if __name__ == '__main__':
    main() 