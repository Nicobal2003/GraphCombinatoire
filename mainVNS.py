from read_input import read_input
from GivenMethod import fobj
from VNS import vns_factorization
import time

def main():
    # Lecture de l'input
    X, m, n, r, LW, UW, LH, UH = read_input('input.txt')

    print("X =\n", X)
    print("m =", m)
    print("n =", n)
    print("r =", r)
    print("LW =", LW, "UW =", UW)
    print("LH =", LH, "UH =", UH)

    # Lancement de la métaheuristique VNS
    start = time.time()
    W, H, F = vns_factorization(X, r, LW, UW, LH, UH)
    elapsed = time.time() - start

    # Calcul de la fonction objectif
    fval = fobj(X, W, H)

    print("Valeur objectif :", fval)
    print("Valeur objectif :", F)
    print("Temps :", elapsed, "secondes")

    # Écriture du fichier output.txt
    print("Erreur :", F)
    print("Temps éxécution de la fonction objectif RS =", elapsed)

if __name__ == "__main__":
    main()
