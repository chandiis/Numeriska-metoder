# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')
    
def cond_inf(M):
    """Konditionstal i maxnorm (infinity norm): ||M||_inf * ||inv(M)||_inf"""
    return np.linalg.norm(M, ord=np.inf) * np.linalg.norm(np.linalg.inv(M), ord=np.inf)

def newton_divdiff(x, y):
    """Newton-koefficienter via delade differenser."""
    x = np.array(x, dtype=float)
    dd = np.array(y, dtype=float).copy()
    coeffs = [dd[0]]
    for k in range(1, len(x)):
        dd = (dd[1:] - dd[:-1]) / (x[k:] - x[:-k])
        coeffs.append(dd[0])
    return np.array(coeffs)

def eval_newton(x_nodes, coeffs, x):
    """Evaluera Newtonpolynom i punkter x."""
    x_nodes = np.array(x_nodes, dtype=float)
    x = np.array(x, dtype=float)
    p = np.zeros_like(x)

    for i in range(len(x)):
        val = coeffs[0]
        prod = 1.0
        for k in range(1, len(coeffs)):
            prod *= (x[i] - x_nodes[k-1])
            val += coeffs[k] * prod
        p[i] = val
    return p

def newton_matrix(x):
    """Matrisen för Newtonbasen: kol j = Π_{k=0..j-1} (x - x_k)."""
    x = np.array(x, dtype=float)
    n = len(x)
    M = np.zeros((n, n))
    M[:, 0] = 1.0
    for j in range(1, n):
        prod = np.ones(n)
        for k in range(j):
            prod *= (x - x[k])
        M[:, j] = prod
    return M

def eval_monomial(c, x):
    """Evaluerar p(x) = c0 + c1 x + c2 x^2 + ..."""
    p = np.zeros_like(x)
    for k in range(len(c)):
        p += c[k] * x**k
    return p

def modell_funktion(t,a):
    #andra grads polynom, uppgift c)
    
    t = np.array(t)
    p = a[0] + a[1]*t + a[2]*t**2
    
    
    #tredje grads polynom, uppgift d)
    """
    t = np.array(t)
    p = a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3
    """
    
    #uppgift e)
    """
    t = np.array(t)
    p = a[0] + a[1]*np.cos(((2*np.pi)/12)*t) + a[2]*np.sin(((2*np.pi)/12)*t)
    """
    return p

def minstakvadratmetoden(x,y):   
        
        #uppgift c)
        
        n = len(x)  #antal datapunkter
        x = np.array(x).reshape(n,1)    #x är kolonnvektor
        y = np.array(y).reshape(n,1)    #y är kolonnvektor
        
        A = np.hstack([x**0, x, x**2])  #x**0 är kolonn med ettor
        
        #normalekvationen
        AT = np.transpose(A)            
        a = np.linalg.solve(AT@A,AT@y)
        
        #Beräkna 2-normen av residualvektorn
        r = y - A@a                     #residualvektorn
        r_2norm = np.linalg.norm(r,2)   #normen ||r||
        SE = r_2norm**2                 #minstakvadratfel
    
        print(f"Fitted parameters: a0 = {a[0][0]:.6f}, a1 = {a[1][0]:.6f}, a2 = {a[2][0]:.6f}")
        print(f"\n2-normen av r = {r_2norm:6f}")
        print(f"\nSquare Error = {SE:6f}")
        
        #uppgift d)
        """
        n = len(x)  #antal datapunkter
        x = np.array(x).reshape(n,1)    #x är kolonnvektor
        y = np.array(y).reshape(n,1)    #y är kolonnvektor
        
        A = np.hstack([x**0, x, x**2, x**3])  #x**0 är kolonn med ettor
        
        #normalekvationen
        AT = np.transpose(A)            
        a = np.linalg.solve(AT@A,AT@y)
        
        #Beräkna 2-normen av residualvektorn
        r = y - A@a                     #residualvektorn
        r_2norm = np.linalg.norm(r,2)   #normen ||r||
        SE = r_2norm**2                 #minstakvadratfel
    
        print(f"Fitted parameters: a0 = {a[0][0]:.6f}, a1 = {a[1][0]:.6f}, a2 = {a[2][0]:.6f}, a3 = {a[3][0]:.6f}")
        print(f"\n2-normen av r = {r_2norm:6f}")
        print(f"\nSquare Error = {SE:6f}")
        """
        #uppgift e)
        """
        n = len(x)  #antal datapunkter
        x = np.array(x).reshape(n,1)    #x är kolonnvektor
        y = np.array(y).reshape(n,1)    #y är kolonnvektor
        
        k = np.pi*2/12
        A = np.hstack([x**0, np.cos(k*x), np.sin(k*x)])  #x**0 är kolonn med ettor
        
        #normalekvationen
        AT = np.transpose(A)            
        a = np.linalg.solve(AT@A,AT@y)
        
        #Beräkna 2-normen av residualvektorn
        r = y - A@a                     #residualvektorn
        r_2norm = np.linalg.norm(r,2)   #normen ||r||
        SE = r_2norm**2                 #minstakvadratfel
    
        print(f"Fitted parameters: a0 = {a[0][0]:.6f}, a1 = {a[1][0]:.6f}, a2 = {a[2][0]:.6f}")
        print(f"\n2-normen av r = {r_2norm:6f}")
        print(f"\nSquare Error = {SE:6f}")
        """
        return a
    
def plotta(xdata,ydata,a):
    # Plot data points
    fig, ax = plt.subplots()
    plt.scatter(xdata, ydata, color='blue', label='Datapunkter')
    x_vec= np.linspace(min(xdata), max(xdata), 100)
    y_vec = modell_funktion(x_vec, a)
    plt.plot(x_vec, y_vec, color='red', label='Modellfunktion')
    
    # Plot residuals
    y_pred = modell_funktion(xdata, a)
    for xi, yi, ypi in zip(xdata, ydata, y_pred):
        plt.plot([xi, xi], [yi, ypi], color='gray', linestyle='--')
       
    #Plot properties 
    ax.set_xlabel('Tid i månader',fontsize =14)
    ax.set_ylabel('Soltid i min',fontsize =14)
    plt.legend()
    plt.grid(False)
    plt.show()
    
def main():
    clear_console()
    
    # Data (tabell 1)
    t = np.arange(1, 13, dtype=float)
    y = np.array([421,553,709,871,1021,1109,1066,929,771,612,463,374], dtype=float)
    tm = t.mean()
    
    # Uppgift 2a: p1, p2, p3
    # p1: naiv Vandermonde (t^0..t^11)
    A1 = np.vander(t, N=12, increasing=True)
    c1 = np.linalg.solve(A1, y)
    
    # p2: centrerad ((t-tm)^0..(t-tm)^11)
    A2 = np.vstack([(t - tm)**k for k in range(12)]).T
    c2 = np.linalg.solve(A2, y)
    
    # p3: Newton (delade differenser)
    A3 = newton_matrix(t)
    c3 = newton_divdiff(t, y)
    
    # Evaluera i 1000 punkter mellan 0 och 12
    xvec = np.linspace(0, 12, 1000)
    
    p1 = eval_monomial(c1, xvec)
    p2 = eval_monomial(c2, xvec - tm)
    p3 = eval_newton(t, c3, xvec)
    
    plt.figure(figsize=(8,5))
    
    # Datapunkter
    plt.scatter(t, y, color="black", zorder=5, label="Datapunkter")
    
    # Modeller
    plt.plot(xvec, p1, label="p1: Vandermonde", linewidth=3)
    plt.plot(xvec, p2, "--", label="p2: Centrerad", linewidth=3)
    plt.plot(xvec, p3, ":", label="p3: Newton", linewidth=3)
    
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("Interpolationspolynom (grad 11)")
    
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("Uppgift 2a) maxskillnader på [0,12]:")
    print("max |p1-p2| =", np.max(np.abs(p1 - p2)))
    print("max |p1-p3| =", np.max(np.abs(p1 - p3)))
    print("max |p2-p3| =", np.max(np.abs(p2 - p3)))
    
    # Uppgift 2b: konditionstal för A1, A2, A3 i maxnorm
    
    print("\nUppgift 2b) konditionstal (infinity norm):")
    print("cond_inf(A1) =", cond_inf(A1))
    print("cond_inf(A2) =", cond_inf(A2))
    print("cond_inf(A3) =", cond_inf(A3))
    
    #uppgift 2c)
    """
    print("Uppgift c) ger:\n")
    xdata = [4,5,6,7,8]
    ydata = [871,1021,1109,1066,929]   
    a = minstakvadratmetoden(xdata, ydata)
    plotta(xdata, ydata, a)
    """
    #uppgift 2d)
    """
    print("Uppgift d) ger:\n")
    xdata = [4,5,6,7,8]
    ydata = [871,1021,1109,1066,929]   
    a = minstakvadratmetoden(xdata, ydata)
    plotta(xdata, ydata, a)
    """
    #uppgift 2e)
    """
    print("Uppgift e) ger:\n")
    xdata = [1,2,3,4,5,6,7,8,9,10,11,12]
    ydata = [421, 553, 709, 871, 1021, 1109, 1066, 929, 771, 612, 463, 374]
    a = minstakvadratmetoden(xdata, ydata)
    plotta(xdata, ydata, a)
    """
    # Uppgift 2f) konditionstal för normalekvationernas matris (A^T A) i c-e
    """
    # c,d: april-augusti t=4..8
    tc = np.arange(4, 9, dtype=float)
    
    Ac = np.vstack([np.ones_like(tc), tc, tc**2]).T
    Ad = np.vstack([np.ones_like(tc), tc, tc**2, tc**3]).T
    
    # e: trig alla månader
    omega = 2*np.pi/12
    Ae = np.vstack([np.ones_like(t), np.cos(omega*t), np.sin(omega*t)]).T
    
    ATAc = Ac.T @ Ac
    ATAd = Ad.T @ Ad
    ATAe = Ae.T @ Ae
    
    print("\nUppgift 2f) cond_inf(A^T A):")
    print("c) cond_inf(ATA) =", cond_inf(ATAc))
    print("d) cond_inf(ATA) =", cond_inf(ATAd))
    print("e) cond_inf(ATA) =", cond_inf(ATAe))
    """
if __name__=="__main__":
    main()   