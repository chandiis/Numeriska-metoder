#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Uppgift 8.1
# formulera olinjära systemet och Jacobianen

def task_8_1():
    print("\nUppgift 8.1 ")
    print("Vi diskretiserar randvärdesproblemet med centrala differenser.")
    print()
    print("För inre punkter används:")
    print("(T[i-1] - 2*T[i] + T[i+1]) / h^2")
    print()
    print("Det olinjära systemet blir:")
    print("F_i(T) = (T[i-1] - 2*T[i] + T[i+1]) / h^2")
    print("         - alpha1*(T[i] - T_inf)")
    print("         - alpha2*(T[i]^4 - T_inf^4) = 0")
    print()
    print("Jacobianen blir tridiagonal:")
    print("J[i,i-1] = 1/h^2")
    print("J[i,i]   = -2/h^2 - alpha1 - 4*alpha2*T[i]^3")
    print("J[i,i+1] = 1/h^2")
    
# Uppgift 8.2
# Funktion som löser randvärdesproblemet med Newton
def solve_problem(N, L, alpha1, alpha2, Ts, TL, T_inf):
    x = np.linspace(0, L, N + 1)
    h = L / N

    # startgissning, rak linje mellan randvärdena
    T = np.linspace(Ts, TL, N + 1)

    tol = 1e-8
    max_iter = 50

    for k in range(max_iter):
        F = np.zeros(N + 1)
        J = np.zeros((N + 1, N + 1))
        
        # randvillkor
        F[0] = T[0] - Ts
        F[N] = T[N] - TL
        J[0, 0] = 1
        J[N, N] = 1

        # inre punkter
        for i in range(1, N):
            F[i] = (T[i - 1] - 2 * T[i] + T[i + 1]) / h**2 \
                   - alpha1 * (T[i] - T_inf) \
                   - alpha2 * (T[i]**4 - T_inf**4)

            J[i, i - 1] = 1 / h**2
            J[i, i] = -2 / h**2 - alpha1 - 4 * alpha2 * T[i]**3
            J[i, i + 1] = 1 / h**2

        delta = np.linalg.solve(J, -F)
        T = T + delta
        
        if np.max(np.abs(delta)) < tol:
            break

    return x, T

def task_8_2():
    print("\nUppgift 8.2")
    print("Programmet solve_problem(...) löser randvärdesproblemet")
    print("för givna N, L, alpha1, alpha2, Ts, TL och T_inf.")
    
# Analytisk lösning för 8.3a
def exact_solution(x, Ts, T_inf, alpha1):
    return T_inf + (Ts - T_inf) * np.exp(-np.sqrt(alpha1) * x)

# Diskret 2-fel
def discrete_error(T_num, T_exact, N):
    summa = 0
    for i in range(1, N):
        summa = summa + (T_num[i] - T_exact[i])**2
    return np.sqrt(summa / (N - 1))

# Uppgift 8.3a
def task_8_3a():
    print("\nUppgift 8.3a")
    
    hc = 40
    K = 240
    D = 4.13e-3
    Ts = 450
    T_inf = 293
    TL = T_inf
    L = 2.5
    N = 400

    alpha1 = 4 * hc / (D * K)
    alpha2 = 0

    print("Beräknat alpha1 =", alpha1)

    x, T_num = solve_problem(N, L, alpha1, alpha2, Ts, TL, T_inf)
    T_exact = exact_solution(x, Ts, T_inf, alpha1)

    error = discrete_error(T_num, T_exact, N)
    print("Diskret 2-fel =", error)

    plt.figure()
    plt.plot(x, T_num, label="Numerisk lösning")
    plt.plot(x, T_exact, "--", label="Analytisk lösning")
    plt.xlabel("x (m)")
    plt.ylabel("T (K)")
    plt.title("Temperaturfördelning längs flänsen")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(x, np.abs(T_num - T_exact))
    plt.xlabel("x (m)")
    plt.ylabel("Absolutfel")
    plt.title("Fel mellan numerisk och analytisk lösning")
    plt.grid()
    plt.show()

# Uppgift 8.3b
def task_8_3b():
    print("\nUppgift 8.3b")

    hc = 40
    K = 240
    D = 4.13e-3
    Ts = 450
    T_inf = 293
    TL = T_inf
    L = 2.5

    alpha1 = 4 * hc / (D * K)
    alpha2 = 0
    
    N_list = [50, 100, 200, 400, 800]
    errors = []
    
    print("Tabell för fel:")
    print("N\tFel")

    for N in N_list:
        x, T_num = solve_problem(N, L, alpha1, alpha2, Ts, TL, T_inf)
        T_exact = exact_solution(x, Ts, T_inf, alpha1)
        e = discrete_error(T_num, T_exact, N)
        errors.append(e)
        print(N, "\t", e)
    
    orders = []
    
    for i in range(1, len(N_list)):
        p = np.log(errors[i - 1] / errors[i]) / np.log(N_list[i] / N_list[i - 1])
        orders.append(p)
        print(str(N_list[i - 1]) + "-" + str(N_list[i]), "\t", p)

    plt.figure()
    plt.semilogy(N_list, errors, "o-")
    plt.xlabel("N")
    plt.ylabel("Diskret 2-fel")
    plt.title("Konvergensstudie")
    plt.grid()
    plt.show()

# Uppgift 8.4a
def task_8_4a():
    print("\nUppgift 8.4a")

    L = 0.30
    N = 400
    D = 5.0e-3
    T_inf = 293.15
    Ts = 373.15
    TL = T_inf
    sigma = 5.67e-8

    materials = {
        "SS AISI 316": [14, 100, 0.17],
        "Aluminium": [180, 100, 0.82],
        "Koppar": [398, 100, 0.03]
    }

    plt.figure()
    
    print("Material\talpha1\t\t     alpha2")
    
    for name in materials:
        K = materials[name][0]
        hc = materials[name][1]
        eps = materials[name][2]

        alpha1 = 4 * hc / (D * K)
        alpha2 = 4 * eps * sigma / (D * K)

        print(name, alpha1, alpha2)

        x, T = solve_problem(N, L, alpha1, alpha2, Ts, TL, T_inf)
        plt.plot(x, T, label=name)

    plt.xlabel("x (m)")
    plt.ylabel("T (K)")
    plt.title("Temperaturprofiler för olika material")
    plt.legend()
    plt.grid()
    plt.show()

# Uppgit 8.4b

def task_8_4b():
    print("\nUppgift 8.4b")

    D = 5.0e-3
    T_inf = 293.15
    Ts = 373.15
    sigma = 5.67e-8

    materials = {
        "SS AISI 316": [14, 100, 0.17],
        "Aluminium": [180, 100, 0.82],
        "Koppar": [398, 100, 0.03]
    }
    
    print("Kriterium: exp(-sqrt(alpha1)*L) <= 0.01")
    print("=> L_min = -ln(0.01) / sqrt(alpha1) = 4.605 / sqrt(alpha1)\n")

    for name in materials:
        K   = materials[name][0]
        hc  = materials[name][1]
        eps = materials[name][2]

        alpha1 = 4 * hc / (D * K)
        alpha2 = 4 * eps * sigma / (D * K)

        L_min = -np.log(0.01) / np.sqrt(alpha1)

        print(name)
        print("  alpha1 =", round(alpha1, 2), "m^-2")
        print("  alpha2 =", round(alpha2, 10), "m^-2 K^-3")
        print("  L_min  =", round(L_min, 4), "m")
        print()

def main():
    task_8_1()
    task_8_2()
    task_8_3a()
    task_8_3b()
    task_8_4a()
    task_8_4b()

if __name__ == "__main__":
    main()