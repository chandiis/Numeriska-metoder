# -*- coding: utf-8 -*-
import numpy as np   #numeriska beräkningar
import matplotlib.pyplot as plt   #plot/rita grafer
import scipy.integrate as integrate
import os   

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')
    
#Uppgift 2b)
def F(t, y, R, L, C):
    q, i = y
    dy1dt = i
    dy2dt = -(1/(L*C))*q - (R/L)*i
    return np.array([dy1dt, dy2dt])

#Uppgift 2d)
def framatEulerSystem(F, tspan, U0, h):
    """ 
    Implementerar Framåt Euler för ODE-System
    
    Parameters:
        F       : Vektorvärd funktion F(t, y)
        tspan   : Tidsintervall
        U0      : initial state U0
        h       : steglängd, eller tidssteg
    
    Returns:
        tk      : numpy array of time points
        Uk      : numpy array of state values
    """
    
    n_steps = round(np.abs(tspan[1]-tspan[0])/h)    
    tk = np.zeros(n_steps+1)    #sparar alla tidpunkter t0, t1, ..., tk
    Uk = np.zeros((n_steps+1, len(U0)))     #sparar alla lösningsvärden

    tk[0] = tspan[0]    #t0
    Uk[0] = U0          #begynnelsevillkor

    for k in np.arange(n_steps):    #från k till N-1
        Uk[k+1] = Uk[k] + h * F(tk[k], Uk[k])   #Framåt Euler
        tk[k+1] = tk[k] + h

    return tk, Uk

def plot_solutions(t_list, U_list, N_values):
    """   
    Parameters:
        t_list  : lista med tidsvektorer [tk1, tk2, ...] 
        U_list  : lista med lösningar [Uk1, Uk2, ...] (matris n_steps+1 x 2)
        N_values: lista med antal tidssteg som motsvarar lösningarna
    """
    plt.figure(figsize=(12,6))
    
    for t, U, N in zip(t_list, U_list, N_values):
        plt.plot(t, U[:,0], label=f'N={N}')  # q(t)
    
    plt.xlabel("Tid t [s]")
    plt.ylabel("q(t) [C]")  
    plt.title("Framåt Euler, dämpad svängning")
    plt.grid(True, linestyle='-.')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def main():
    clear_console()
    #Uppgift 2a)
    print("Uppgift 2a) Gjordes teoretiskt, vilket gav:")
    print("Systemet y' = F(t,y), där:")
    print("q' = i")
    print("i' = -(1/(L*C))*q - (R/L)*i")
    print("y = [q, i]^T med begynnelsevillkor y(0) = [Q0, 0]^T")
    
    #Uppgift 2b)
    print("\nUppgift 2b) löstes via funktionen F(t, y, R, L, C)")
    
    #Uppgift 2c)
    tspan = np.array([0, 20]) #tidsintervallet
    Q0 = 1
    y0 = np.array([Q0, 0]) #begynnelsevillkor
    R1, L, C = 1, 2, 0.5
    t0, tend = tspan[0], tspan[-1]
    
    sol1 = integrate.solve_ivp(F, tspan, y0, args=(R1,L,C), method='RK45', t_eval=np.linspace(t0, tend, 10))

    print("\nLösning för 2c): \nDämpad svängning (R=1) i):")
    print("t:", np.round(sol1.t,3))
    print("q(t):", np.round(sol1.y[0],4))
    print("i(t):", np.round(sol1.y[1],4))
    
    R2, L, C = 0, 2, 0.5
    
    sol2 = integrate.solve_ivp(F, tspan, y0, args=(R2,L,C), method='RK45', t_eval=np.linspace(t0, tend, 10))
    
    print("\nOdämpad svängning (R=0) ii):")
    print("t:", np.round(sol2.t,3))
    print("q(t):", np.round(sol2.y[0],4))
    print("i(t):", np.round(sol2.y[1],4))
    
    #Uppgift 2d)
    R, L, C = 1, 2, 0.5 
    Fsystem = lambda t, y: F(t, y, R, L, C)
    
    N_values = [20, 40, 80, 160]    
    t_list = []
    U_list = []

    print("\nLösning för 2d): Se den plottade funktionen ovan.")
    
    for N in N_values:
        h = (20 - 0)/N  #h = (b-a)/N
        
        tk, Uk = framatEulerSystem(Fsystem, [0,20], y0, h)
        t_list.append(tk)
        U_list.append(Uk)      
        print(f"\n=== Euler-lösning för N = {N}, h = {h:.3f} ===")
        print("t:", np.round(tk, 3))
        print("q(t):", np.round(Uk[:,0], 4))
        print("i(t):", np.round(Uk[:,1], 4))
    
    plot_solutions(t_list, U_list, N_values)
    
    print("\nTeorin ger att Euler Framåt är asymptotisk stabilt om h < 0.5, det vill säga N > 40.")
    
    #Uppgift 2e)
    t_ref = np.linspace(0, 20, 10000)
    
    sol_ref = integrate.solve_ivp(Fsystem, [0,20], y0, t_eval=t_ref, method='RK45')
    U_ref_T = sol_ref.y[:, -1] #både i(t) & q(t) vid T = 20
    
    N_values = [80, 160, 320, 640, 1280, 2650] #N > 40 för stabil numerisk lösning, dubblar N
    
    errors_q = []
    errors_i = []
    h_values = []
    
    print("\nN           h       fel_q(T)       fel_i(T)")
    print("---------------------------------------------")
    
    for N in N_values:
        h = 20 / N
        tk, Uk = framatEulerSystem(Fsystem, [0,20], y0, h)
        
        # Fel vid sluttiden T
        error_q = abs(Uk[-1,0] - U_ref_T[0])
        error_i = abs(Uk[-1,1] - U_ref_T[1])
        
        errors_q.append(error_q)
        errors_i.append(error_i)
        h_values.append(h)
        
        print(f"{N:<8} {h:<8.5f} {error_q:.6e}   {error_i:.6e}")
        
    print("\nOrdning p för q-komponenten:")
    print("mellan h och h/2")
    for i in range(len(errors_q)-1):
        p_q = np.log(errors_q[i]/errors_q[i+1]) / np.log(2)
        print(f"h={h_values[i]:.5f} -> {h_values[i+1]:.5f}   p ≈ {p_q:.4f}")

    print("\nOrdning p för i-komponenten:")
    print("mellan h och h/2")
    for i in range(len(errors_i)-1):
        p_i = np.log(errors_i[i]/errors_i[i+1]) / np.log(2)
        print(f"h={h_values[i]:.5f} -> {h_values[i+1]:.5f}   p ≈ {p_i:.4f}")

print("\nSlutsats: p ≈ 1 ⇒ Framåt Euler är av ordning 1.")
    
if __name__ == '__main__':
    main()  

