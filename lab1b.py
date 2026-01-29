# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')
    
def modell_funktion(t,a):
    #andra grads polynom, uppgift c)
    t = np.array(t)
    p = a[0] + a[1]*t + a[2]*t**2
    
    
    #tredje grads polynom, uppgift d)
    """
    t = np.array(t)
    p = a[0] + a[1]*t + a[2]*t**2 + a[2]*t**3
    """
    return p
    

def minstakvadratmetoden(x,y):   
        """
        #uppgift c)
        """
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
    
        print(f"Fitted parameters: a0 = {a[0][0]:.6f}, a1 = {a[1][0]:.6f}, a3 = {a[2][0]:.6f}")
        print(f"\n2-normen av r = {r_2norm:6f}")
        print(f"\nSquare Error = {SE:6f}")
        
        """
        #uppgift d)
        
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
    
        print(f"Fitted parameters: a0 = {a[0][0]:.6f}, a1 = {a[1][0]:.6f}, a3 = {a[2][0]:.6f}, a4={a[3][0]:.6f}")
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
    xdata = [4,5,6,7,8]
    ydata = [871,1021,1109,1066,929]   
    a = minstakvadratmetoden(xdata, ydata)
    plotta(xdata, ydata, a)
    
if __name__=="__main__":
    main()   