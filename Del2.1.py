#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Del 2.1
# Funktion och exakt derivata

def f(x):
    return np.sin(np.exp(x))

def f_derivata(x):
    return np.exp(x) * np.cos(np.exp(x))


def main():

    # Punkt där derivatan ska approximeras
    x = 0.75
    exakt = f_derivata(x)

    # Steglängder h = 2^(-k), k = 1,...,8
    k_values = np.arange(1, 9)
    h_values = 2.0 ** (-k_values)

    fel_centrerad = []
    fel_framaat = []
    fel_bakaat = []

    for h in h_values:

        # Formel (8): centrerad differens
        D_c = (f(x + h) - f(x - h)) / (2*h)

        # Formel (12): framåtdifferens av ordning 2
        D_f = (-f(x + 2*h) + 4*f(x + h) - 3*f(x)) / (2*h)

        # Formel (13): bakåtdifferens av ordning 2
        D_b = (3*f(x) - 4*f(x - h) + f(x - 2*h)) / (2*h)

        fel_centrerad.append(abs(D_c - exakt))
        fel_framaat.append(abs(D_f - exakt))
        fel_bakaat.append(abs(D_b - exakt))

    # Gör om till numpy-arrayer
    fel_centrerad = np.array(fel_centrerad)
    fel_framaat = np.array(fel_framaat)
    fel_bakaat = np.array(fel_bakaat)

    # Referenskurva proportional mot h^2
    C = fel_centrerad[0] / (h_values[0]**2)
    referens = C * h_values**2

    # Skriver ut resultat
    print("Exakt derivata f'(0.75) =", exakt)
    print()

    print("k     h            fel centrerad       fel framåt        fel bakåt")

    for k, h, e1, e2, e3 in zip(k_values, h_values,
                                 fel_centrerad,
                                 fel_framaat,
                                 fel_bakaat):

        print(k, "  ", h, "  ", e1, "  ", e2, "  ", e3)

    # Plot
    plt.figure(figsize=(8, 5))

    plt.loglog(1/h_values, fel_centrerad, 'o-',
               label='Centrerad differens (8)')

    plt.loglog(1/h_values, fel_framaat, 's-',
               label='Framåtdifferens (12)')

    plt.loglog(1/h_values, fel_bakaat, '^-',
               label='Bakåtdifferens (13)')

    plt.loglog(1/h_values, referens, '--',
               label=r'Referenskurva $\sim h^2$')

    plt.xlabel(r'$1/h$')
    plt.ylabel('Absolutfel')
    plt.title(r"Absolutfel för differensapproximationer av $f'(0.75)$")

    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig("del21_loglog.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()