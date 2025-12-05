from read_input import read_input
from GivenMethod import fobj,metaheuristicVNS,metaheuristicRS, metaheuristicRS2  # importe la VNS et fobj
from test import metaheuristicRStest
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

# bestW, bestH = metaheuristicRStest(X, r, LW, UW, LH, UH) #adapter le temps ici

bestW, bestH = metaheuristicRStest(
    X, r, LW, UW, LH, UH,
    time_limit=5,              
    restart_guided_prob=0.7,
    noise_rate_W=0.05,
    noise_rate_H=0.05)

end_timeRS = time.time()            
elapsedRS = end_timeRS - start_timeRS   # durée totale en secondes

print("\n--- Résultat final ---")
print("W =\n", bestW)
print("H =\n", bestH)

# Calcul de la valeur de l’objectif
val = fobj(X, bestW, bestH)
print("\nValeur finale de la fonction objectif =", val)
print("\temps éxécution de la fonction objectif RS =", elapsedRS)
