# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# f(x) = x^3 * e^x
def f(x):
    return x**3 * np.exp(x)

def trapets(func, intervall, n):
    a, b = intervall
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    I_T = h * (y[0] + 2*np.sum(y[1:-1]) + y[-1]) / 2
    return I_T


def trapets2(f_values, t, n):
    """
    Beräkna trapetsapproximation enligt formeln:
    T = h/2 * [f(a) + 2*sum(f_i) + f(b)]

    Parametrar:
        f_values : datapunkter [f0, f1, ..., fn]
        h        : steglängd mellan punkterna
        t        : datapunkter [x0, x1,..., xn]
    """
    f_values = np.array(f_values, dtype=float)
    t = np.array(t, dtype=float)

    h = (t[-1] - t[0]) / n
    T = h/2 * (f_values[0] + 2*np.sum(f_values[1:-1]) + f_values[-1])
    return T

def subsample_data(t, fvals, step):
    """
    Tar ut var 'step':te punkt (inkluderar alltid sista punkten).
    Ex: step=2 -> 2014,2016,2018,2020,2022
    """
    t = np.array(t, dtype=float)
    fvals = np.array(fvals, dtype=float)

    idx = np.arange(0, len(t), step)
    if idx[-1] != len(t) - 1:
        idx = np.append(idx, len(t) - 1)
    return t[idx], fvals[idx]

def simpson_from_data(f_values, t_values):
    """
    Simpsons regel på tabellvärden (kräver jämnt antal intervall n).
    S = h/3 [f0 + fn + 4*(sum odd) + 2*(sum even)]
    """
    f_values = np.array(f_values, dtype=float)
    t_values = np.array(t_values, dtype=float)
    n = len(t_values) - 1
    if n % 2 != 0:
        raise ValueError("Simpsons regel kräver jämnt antal intervall (n jämnt).")

    h = (t_values[-1] - t_values[0]) / n
    odd_sum = np.sum(f_values[1:-1:2])
    even_sum = np.sum(f_values[2:-1:2])
    return h/3 * (f_values[0] + f_values[-1] + 4*odd_sum + 2*even_sum)

def main():
    clear_console()

   
    # a) och b)
 
    intervall = [0, 2]
    I_exact = 6 + 2*np.exp(2)

    n_vals = [10, 20, 40, 80, 160, 320, 640, 1280]
    h_vals = []    #steglängder
    T_vals = []    #trapphetsapproximationer
    error = []     #fel e_h
    p = [np.nan]   #första p går ej att beräkna, finns ingen tidigare fel att jämföra med

    for n in n_vals:
        h = (intervall[1] - intervall[0]) / n   #n fördubblas -> h halveras
        Tn = trapets(f, intervall, n)
        eh = abs(I_exact - Tn)

        h_vals.append(h)
        T_vals.append(Tn)
        error.append(eh)

    #noggrannhetsordningen p
    for i in range(1, len(error)):
        pi = np.log(error[i-1] / error[i]) / np.log(2)
        p.append(pi)

    df = pd.DataFrame({
        "n": n_vals,
        "h": h_vals,
        "T_h": T_vals,
        "e_h": error,
        "p": p
    })

    print("Uppgift 3a) och 3b) – Trapetsregeln för f(x)=x^3 e^x på [0,2]\n")
    print(df)

   
    # c)
   
    t = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    f_t = [12.00, 15.10, 19.01, 23.92, 30.11, 37.90, 47.70, 60.03, 75.56]
    n = 8  # 2022 - 2014

    T_total = trapets2(f_t, t, n)
    print("\nUppgift 3c) – Trapets med tabellvärden (2014–2022)\n")
    print(f"Total producerad energi (2014–2022): {T_total:.2f} kWår")

    #  d)
   
    steps = [1, 2, 4, 8]  # "h" i år
    Th = {}

    for step in steps:
        tt, ff = subsample_data(t, f_t, step)
        Th[step] = trapets2(ff, tt, len(tt)-1)

    # Feluppskattningar: e_{2h} ≈ |T_h - T_{2h}|
   
    err_est = {2: abs(Th[1] - Th[2]),
               4: abs(Th[2] - Th[4]),
               8: abs(Th[4] - Th[8])}

    # p_4 ≈ log2(err(4)/err(2)), p_8 ≈ log2(err(8)/err(4))
    p2 = np.nan
    p4 = np.log(err_est[4] / err_est[2]) / np.log(2)
    p8 = np.log(err_est[8] / err_est[4]) / np.log(2)

    df_d = pd.DataFrame({
        "h (år)": [1, 2, 4, 8],
        "T_h (kWår)": [Th[1], Th[2], Th[4], Th[8]],
        "fel ≈ |T_h - T_2h|": [np.nan, err_est[2], err_est[4], err_est[8]],
        "p-skattning": [np.nan, p2, p4, p8]
    })

    print("\nUppgift 3d) – Konvergensstudie med tabell-data\n")
    print(df_d.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

   
    #  e)
   
    T1 = Th[1]
    T2 = Th[2]
    R = (4*T1 - T2) / 3
    S = simpson_from_data(f_t, t)

    print("\nUppgift 3e) – Richardson och Simpsons regel\n")
    print(f"T_1 (h=1): {T1:.6f} kWår")
    print(f"T_2 (h=2): {T2:.6f} kWår")
    print(f"Richardson R = (4*T1 - T2)/3: {R:.6f} kWår")
    print(f"Simpson (h=1):               {S:.6f} kWår")
    print(f"Skillnad |R - S|:            {abs(R - S):.6e} kWår")

 
    #  f)
   
    x = np.array(t, dtype=float) - 2014.0
    y = np.log(np.array(f_t, dtype=float))

    A = np.column_stack([np.ones_like(x), x])  # [1, x]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    c0, b = coef
    a = np.exp(c0)

    print("\nUppgift 3f) – Minstakvadrat för exponentiell modell\n")
    print(f"a = {a:.6f}")
    print(f"b = {b:.6f}")
    print("Modell: f(t) = a * exp(b*(t-2014))")

   
    #  g)
   
    f2023 = a * np.exp(b * (2023.0 - 2014.0))
    t_ext = np.append(np.array(t, dtype=float), 2023.0)
    f_ext = np.append(np.array(f_t, dtype=float), f2023)

    T_2014_2023 = trapets2(f_ext, t_ext, len(t_ext)-1)

    cond1 = f2023 > 100.0
    cond2 = T_2014_2023 > 350.0

    print("\nUppgift 3g) – Prognos och beslut\n")
    print(f"Prognos f(2023): {f2023:.2f} kW")
    print(f"Integral 2014–2023 (trapets, h=1): {T_2014_2023:.2f} kWår")
    print(f"Villkor 1: f(2023) > 100 kW  -> {cond1}")
    print(f"Villkor 2: Energi > 350 kWår -> {cond2}")
    print("Beslut: Projektet är lyckat." if (cond1 or cond2) else "Beslut: Projektet är inte lyckat.")
    print("Ett av villkoren är uppfyllda.")

if __name__ == "__main__":
    main()