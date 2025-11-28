from read_input import read_input
from GivenMethod import fobj,metaheuristicVNS,metaheuristicRS, metaheuristicRSRAPID  # importe la VNS et fobj
import time


# ------------------ Lecture du fichier input ------------------
X, m, n, r, LW, UW, LH, UH = read_input('input.txt')

print("X =\n", X)
print("m =", m)
print("n =", n)
print("r =", r)
print("LW =", LW, "UW =", UW)
print("LH =", LH, "UH =", UH)


# ------------------ Lancement de la métaheuristique (VNS) ------------------
# print("\n--- Lancement du VNS ---")

# start_time = time.time()          

# bestW, bestH = metaheuristicVNS(X, r, LW, UW, LH, UH)

# end_time = time.time()           
# elapsed = end_time - start_time   # durée totale en secondes

# print("\n--- Résultat final ---")
# print("W =\n", bestW)
# print("H =\n", bestH)

# # Calcul de la valeur de l’objectif
# val = fobj(X, bestW, bestH)
# print("\nValeur finale de la fonction objectif =", val)
# print("\temps éxécution de la fonction objectif VNS =", elapsed)

# ------------------ Lancement de la métaheuristique (RS) ------------------
print("\n--- Lancement du RS ---")

start_timeRS = time.time()          

bestW, bestH = metaheuristicRS(X, r, LW, UW, LH, UH,60) #adapter le temps ici

end_timeRS = time.time()            
elapsedRS = end_timeRS - start_timeRS   # durée totale en secondes

print("\n--- Résultat final ---")
print("W =\n", bestW)
print("H =\n", bestH)

# Calcul de la valeur de l’objectif
val = fobj(X, bestW, bestH)
print("\nValeur finale de la fonction objectif =", val)
print("\temps éxécution de la fonction objectif RS =", elapsedRS)
