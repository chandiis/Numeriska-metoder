#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np   #numeriska beräkningar
import matplotlib.pyplot as plt   #plot/rita grafer
import os

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')
    
    #UPPGIFT a)
def fixpunktf():
    x = np.linspace(0,1,100)      #intervallet 0 < x < L
    f= lambda x: (8/3)*x - 3*x**2 + (1/3)*x**3 - (2/3)*np.sin(np.pi*x) #f(x)
    y = f(x)                      #vektor med y-värden
    fig, ax = plt.subplots()      #skapa instanserna fig och ax
    ax.plot(x,y)                  #ritar grafen f(x)
    
    #Ritningen ger 4 nollställen, dvs där f(x) = 0
    #gissningar på rötterna
    xsol1 = 0.3
    ax.scatter(xsol1,f(xsol1), color="red")
    xsol1 = 0.842
    ax.scatter(xsol1,f(xsol1), color="red")

    #plot egenskaper
    ax.tick_params(labelsize=14)
    plt.grid(True,linestyle='-.')
    #plt.ylim([0, 60])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()
    
    #UPPGIFT c)
def fixpunkt_iteration(g, x0, tol, max_iter):
    
    x = x0               #starta iterationen vid x0
    DeltaX = tol + 1.0   #för att loopen ska starta
    n = 0                #antal iterationer
    deltas = []          #Samlar in det absolutafelen |x_n+1 - x_i|
    
    print(" n          x_n         |x_n+1 - x_n|")
    print("-------------------------------------")
    while DeltaX > tol:
        xold = x                    #x_n
        x = g(xold)                 #x_n+1 = g(x_n)
        DeltaX = np.abs(x-xold)     #|x_n+1 - x_n|
        deltas.append(DeltaX)
        n = n + 1
        
        print(f"{n:2d}  {x: .12e}   {DeltaX:.2e}")
        
        if n > max_iter:
            raise RuntimeError("Fixpunkten konvergerar inte")
    return x, n, deltas 
    #returnerar den approximativa fixpunkten, antal iterationer och 

    #UPPGIFT d)
def func(x):
    f = (8/3)*x - 3*x**2 + (1/3)*x**3 - (2/3)*np.sin(np.pi*x) #f(x)
    fprim = 8/3 - 6*x + x**2 - (2/3)*np.pi*np.cos(np.pi * x) #f'(x)
    return f, fprim

def NewtonWhileLoop(func, x0, tol, max_iter):

    x = x0
    DeltaX = tol + 1.0
    n = 0
    deltas = []
    
    print(" n          x_n         |x_n+1 - x_n|")
    print("-------------------------------------")
    while DeltaX > tol:
        f, fp = func(x)
        xnew = x - f/fp
        DeltaX = np.abs(xnew-x)
        deltas.append(DeltaX)
        x = xnew
        n = n + 1
        
        print(f"{n:2d}  {x: .12e}   {DeltaX:.2e}")
        if n > max_iter:
          raise RuntimeError(
              "Newtons metod did not converge within the maximum number \
               of iterations.")
    return x, n, deltas

def rita(deltas_fp, deltas_newton):
    plt.figure()
    plt.semilogy(deltas_fp, 'o-', label='Fixpunktsmetoden')
    plt.semilogy(deltas_newton, 's-', label='Newtons metod')
    plt.xlabel('Iteration n')
    plt.ylabel(r'$|x_{n+1} - x_n|$')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()

def main():
    clear_console()
    #fixpunktf()     #UPPGIFT a)
    """
    #UPPGIFT c)
    g = lambda x: (3/8)*(3*x**2 - (1/3)*x**3 + (2/3)*np.sin(np.pi*x)) #g(x)
    
    x0 = 0.842
    tol = 1e-10
    max_iter = 300
    
    x_star, n = fixpunkt_iteration(g, x0, tol, max_iter) #x*
    
    print(f"\nFixpunkten hittades efter {n} iterationer")
    print(f"x ≈ {x_star:.12f}")
    """
    
    """
    #UPPGIFT d)
    x0 = 0.3    #ett nollställe som inte gick att bestämma med fixpunktsmetoden
    tol = 1e-10
    max_iter = 100
    
    x_star, n = NewtonWhileLoop(func, x0, tol, max_iter)
    print(f"\nFixpunkten hittades efter {n} iterationer")
    print(f"x ≈ {x_star:.12f}")
    """
    
    #UPPGIFT e)
    
    g = lambda x: (3/8)*(3*x**2 - (1/3)*x**3 + (2/3)*np.sin(np.pi*x)) #g(x)
    
    x0 = 0.842  #ett nollställe som går att bestämma med båda metoderna
    tol = 1e-10
    max_iter = 200
    
    print("Fixpunktsmetoden:\n")
    x_fp, n_fp, d_fp = fixpunkt_iteration(g, x0, tol, max_iter)
    print(f"\nFixpunkten hittades efter {n_fp} iterationer")
    print(f"x ≈ {x_fp:.12f}\n")
    print("Newtons Metod:\n")
    x_new, n_new, d_new = NewtonWhileLoop(func, x0, tol, max_iter)
    print(f"\nFixpunkten hittades efter {n_new} iterationer")
    print(f"x ≈ {x_new:.12f}")
    
    rita(d_fp, d_new)

if __name__ == '__main__':
    main()    

