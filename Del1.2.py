#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os

# ----- Givna parametrar -----
K1 = 0.249
n1 = 2.207
Tm = 29.6
Q = 801368
c = 0.389
St = 0.016
P = 49080
Rfi = 1.76e-4
Rfo = 1.76e-4
hs = 356
ht = 356
kw = 60

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')
    
# Definiera den vektorsvärda funktionen F(\vec{x})
def F(xk):
    
    d, Ds = xk[0], xk[1]
    C = np.pi*K1*Ds**(n1)*Tm
    g = d/(Ds*ht) + (d*Rfi)/Ds + (d*np.log(d/Ds))/(2*kw) + Rfo + 1/hs
    
    f1 = C/(d**(n1-1)*g) - Q
    f2 = c/(Ds**(2)*(St-d)**2) - P
    
    F = np.array( [f1, f2] )
    
    return F 

#Konstruhera jakobimatris J(\vec{xk})
def J(xk):
        
    d, Ds = xk[0], xk[1]
    g = d/(Ds*ht) + (d*Rfi)/Ds + (d*np.log(d/Ds))/(2*kw) + Rfo + 1/hs
    gd_prim = 1/(Ds*ht) + Rfi/Ds + (np.log(d/Ds) + 1)/(2*kw)
    gDs_prim = -d/(Ds**2 * ht) - (d*Rfi)/(Ds**2) - d/(Ds*2*kw)
    C = np.pi*K1*Ds**(n1)*Tm
    Cprim = np.pi*K1*n1*Ds**(n1-1)*Tm
    
    J11 = -(C*((n1-1)*d**(n1-2)*g+d**(n1-1)*gd_prim))/((d**(n1-1)*g)**2)
    J12 = (Cprim*(d**(n1-1)*g) - (C*(d**(n1-1)*gDs_prim)))/((d**(n1-1)*g)**2)
    J21 = (2*c)/(Ds**2*(St-d)**3)
    J22 = -(2*c)/(Ds**3*(St-d)**2)
    J = np.array([[ J11, J12], [ J21, J22]])
    
    return J 

def mynewton_system(F, J, xk, tol, maxiter):
    
    #Implemtenterar Newtons metod för system
    
    error = tol + 1.0
    k = 0
    errors = []
    
    while error > tol and k < maxiter:
        #Lös det linjära ekvationssystemet för steg k
        step = -np.linalg.solve(J(xk), F(xk)) 
        x1 = xk + step
        error = np.linalg.norm(step)
        errors.append(error)
        xk = x1
        k = k + 1
        
    return xk, k, errors

def plot_convergence(errors):

    ek = errors[:-1]
    ek1 = errors[1:]

    plt.figure()

    plt.loglog(ek, ek1, 'o-')
    
    plt.xlabel(r"$e_k$")
    plt.ylabel(r"$e_{k+1}$")
    plt.title("Loglog-plot av Newtons konvergens")
    plt.grid(True)

    plt.show()
    
def main():
    clear_console() 
    
    #UPPGIFT 5a)
    x0 = np.array([0.015, 0.8])
    
    root, iter, errors = mynewton_system(F, J, x0, tol=1e-8, maxiter=20)
    print("\n--Uppgift 5a--\n")
    print("d =", root[0], "m")
    print("Ds =", root[1], "m")
    print("Number of iterations: ", iter)
    
    #UPPGIFT 5b)
    print("\n\n--Uppgift 5b--")
    print("Konvergensordningen p är: ")
    for i in range(2, len(errors)):

        p = np.log(errors[i]/errors[i-1]) / np.log(errors[i-1]/errors[i-2])
    
    print(f"p ≈ {p:.4f}")
    plot_convergence(errors)
    
    
if __name__ == '__main__':
    main()