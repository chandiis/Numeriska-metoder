# -*- coding: utf-8 -*-
import numpy as np   #numeriska beräkningar
import matplotlib.pyplot as plt   #plot/rita grafer
import os   

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')
    
# ODE: y' = 1 + t - y
def f(t, y):
    return 1 + t - y

# Exakt lösning (given i uppgiften)
def y_exact(t):
    return np.exp(-t) + t

# Euler framåt 
def euler_framat(f, t0, y0, T, h):
    N = round((T - t0) / h) # antal steg
    t = np.zeros(N + 1)
    y = np.zeros(N + 1)
    t[0] = t0
    y[0] = y0
    
    for k in range(N):
        y[k+1] = y[k] + h * f(t[k], y[k]) # Euler-formeln
        t[k+1] = t[k] + h
        
    return t, y

def direction_field(f, tmin, tmax, ymin, ymax, density, scale):
    xs = np.linspace(tmin, tmax, density)
    ys = np.linspace(ymin, ymax, density)
    TT, YY = np.meshgrid(xs, ys)
    
    S = f(TT, YY)              # lutning
    U = np.ones_like(S)
    V = S
    
    L = np.hypot(U, V)       
    U = U / L
    V = V / L
    
    plt.figure()
    plt.quiver(TT, YY, U, V)
    
    #Plotta analystiska lösningar i samma figur som riktningsfältet
    t_vec = np.linspace(tmin, tmax, 400)
    plt.plot(t_vec, y_exact(t_vec), color='red', label='Exakt lösning')

    plt.xlim(tmin, tmax)
    plt.ylim(ymin, ymax)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("F1a): riktningsfält + exakt")
    plt.grid(True, linestyle="-.")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
#def plot_solution(t, y_num=None, y_ex=None, title=""):


def main():
    clear_console()
    # ------------------------------------------
    # UPPGIFT F1a) riktningsfält + exakt lösning
    # ------------------------------------------
    T = 1.2
    direction_field(f, 0, T, 0, 2.2, 25, 20)
    
    print("F1a) Se den plottade grafen ovan.")
    
    # ----------------------------------------
    # UPPGIFT F1b) Euler framåt (h=0.1) + plot
    # ----------------------------------------
    t0 = 0.0
    y0 = 1.0
    h = 0.1
    t_fine = np.linspace(0, T, 400)
    
    t_eu, y_eu = euler_framat(f, t0, y0, T, h)
    plt.figure()
    plt.plot(t_eu, y_eu, 'o-', label="Euler h=0.1")
    plt.plot(t_fine, y_exact(t_fine), label="Exakt")
    plt.title("F1b: Euler vs exakt")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.grid(True, linestyle='-.')
    plt.legend()
    plt.show()
    
    print("\nF1b) Se den plottade grafen ovan.")
    
    # ---------------------------------
    # UPPGIFT F1c) fel vid T för h=0.1
    # ---------------------------------
    yT_num = y_eu[-1]
    yT_ex = y_exact(T)
    fel = abs(yT_num - yT_ex)
        
    print("\nF1c)")
    print("y_k(T) =", yT_num)
    print("y_exact(T) =", yT_ex)
    print("fel =", fel)
    # ---------------------------------------
    # UPPGIFT F2a) beräkna y_k(T) för flera h
    # UPPGIFT F2b) beräkna fel e_h
    # ---------------------------------------
    hs = [0.2, 0.1, 0.05, 0.025, 0.0125]
    yT_list = []
    e_list = []
    print("\nF2a) och F2b)")
    print("h            y_k(T)          fel")
    print("---------------------------------------")
    
    yT_exact = y_exact(T)
    
    for h in hs:
        t_k, y_k = euler_framat(f, t0, y0, T, h)
        yT = y_k[-1]
        e = abs(yT - yT_exact)
        
        yT_list.append(yT)
        e_list.append(e)
        
        print(f"{h:<7} {yT:.11f} {e:.11e}")
          
    # -------------------------
    # UPPGIFT F2c) ordning p
    # -------------------------
    print("\nF2c) ordning p")
    print("mellan h och h/2")
    print("----------------")
    
    for i in range(len(e_list) - 1):
        p = np.log(e_list[i] / e_list[i+1]) / np.log(2)
        print(f"h={hs[i]} -> {hs[i+1]} p ≈ {p:.6f}")
        
    print("\nSlutsats: p ≈ 1 => Euler framåt är av ordning 1.")
    
if __name__ == '__main__':
    main()

