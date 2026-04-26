#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np   #numeriska beräkningar
import matplotlib.pyplot as plt   #plot/rita grafer
import os  
from scipy import optimize   #tillgång till biblioteket scipy.optimize

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')
    
def func(d):
    #Givna värden
    Tm, Q, Ds, Aex, Rfi, Rfo, hs, ht, kw = 29.6, 801368, 1.219, 64.15, 1.76e-4, 1.76e-4, 356, 356, 60
    g = d/(Ds*ht) + (d*Rfi)/Ds + (d*np.log(d/Ds))/(2*kw) + Rfo + 1/hs
    gprim = 1/(Ds*ht) + Rfi/Ds + (np.log(d/Ds) + 1)/(2*kw)
    f = (Aex * Tm)/g - Q #f(d)
    fprim = (-1 * Aex * Tm * gprim)/(g**2) #f'(d)
    return f, fprim

def newtonWhileLoop(func, d0, tol, max_iter):

    d = d0
    DeltaD = tol + 1.0
    n = 0
    deltas = []     #Samlar in det absolutafelen |d_n+1 - d_n|
    
    print(" n          d_n         |d_n+1 - d_n|")
    print("-------------------------------------")
    while DeltaD > tol:
        f, fp = func(d)
        d_new = d - f/fp
        DeltaD = np.abs(d_new-d)
        deltas.append(DeltaD)
        d = d_new
        n = n + 1
        
        print(f"{n:2d}  {d: .12e}   {DeltaD:.2e}")
        if n > max_iter:
          raise RuntimeError(
              "Newtons metod did not converge within the maximum number \
               of iterations.")
    return d, n, deltas  #returnerar den approximativa nollstället d*, antal iterationer och felen

def f_only(d):
    Tm, Q, Ds, Aex, Rfi, Rfo, hs, ht, kw = 29.6, 801368, 1.219, 64.15, 1.76e-4, 1.76e-4, 356, 356, 60
    g = d/(Ds*ht) + (d*Rfi)/Ds + (d*np.log(d/Ds))/(2*kw) + Rfo + 1/hs
    f = (Aex * Tm)/g - Q #f(d)
    return f

def generate_plot_data(f, d_min, d_max, n_points=500):
    d_vals = np.linspace(d_min, d_max, n_points)  #genererar n punkter inom intervallet d_min < d < d_max
    f_vals = [f(d) for d in d_vals] #skapar en lista med funktionsvärden f(d) för varje d i d_vals
    return d_vals, f_vals

def plot_function(d_vals, f_vals, d_star=None):
    plt.figure()
    plt.plot(d_vals, f_vals, label="f(d)")  #ritar grafen f(d)
    plt.axhline(0, linestyle="--")

    if d_star is not None:
        plt.plot(d_star, 0, 'ro', label="Nollställe")

    plt.xlabel("d (m)")
    plt.ylabel("f(d)")
    plt.title("Newton-funktionen f(d)")
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    
    clear_console()
    #UPPGIFT 2
    print("\n--Uppgift 2--\nYtterdiametern bör vara:\n")
    d0 = 0.007    #ett nollställe som inte gick att bestämma med fixpunktsmetoden
    tol = 1e-8
    max_iter = 50
    
    d_star, n, deltas = newtonWhileLoop(func, d0, tol, max_iter)
    print(f"\nNollstället hittades efter {n} iterationer")
    print(f"d ≈ {d_star:.12f} m")
    
    #UPPGIFT 3a)
    print("\n--Uppgift 3--\na) Konvergensordningen beräknas med p ≈ (log(e_n+1 / e_n))/(log(e_n / e_n-1))")
    
    p_values = []

    for i in range(2, len(deltas)):
        e_n1 = deltas[i]
        e_n = deltas[i-1]
        e_n_1 = deltas[i-2]
        
        p = np.log(e_n1/e_n) / np.log(e_n/e_n_1)
        p_values.append(p)
        print(f"p ≈ {p:.4f}")
    
    print(f"\nSlutsats: p ≈ {p_values[-1]:.4f} ≈ 2 (kvadratisk konvergens)")  
    
    #UPPGIFT 3b)
    root = optimize.fsolve(f_only, d0, xtol=tol)
    print("\nb) Verifiering med optimize.fsolve ger:")
    print(f"d ≈ {root[0]:.12f} m")    
    
    #UPPGIFT 3c)
    d_vals, f_vals = generate_plot_data(f_only, 0.001, 0.05)

    # Plotta
    plot_function(d_vals, f_vals, d_star)
    
    print("\nc) Lösningen finns grafiskt ovan.")

if __name__ == '__main__':
    main() 