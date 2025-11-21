import numpy as np
import random
import os

# ============================
# 📥 Lecture du fichier input
# ============================
def read_input(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    m, n, r, LW, UW, LH, UH = map(int, lines[0].split())
    X = np.array([list(map(int, l.split())) for l in lines[1:]])
    return X, m, n, r, LW, UW, LH, UH

# ============================
# 💾 Écriture du fichier output
# ============================
def write_output(filepath, best_f, W, H):
    with open(filepath, 'w') as f:
        f.write(f"{int(best_f)}\n")
        for i in range(W.shape[0]):
            f.write(" ".join(map(str, W[i])) + "\n")
        for i in range(H.shape[0]):
            f.write(" ".join(map(str, H[i])) + "\n")
# ============================
# 🧩 Génération d’un individu
# ============================
def generate_individual(m, n, r, LW, UW, LH, UH):
    W = np.random.randint(LW, UW + 1, size=(m, r))
    H = np.random.randint(LH, UH + 1, size=(r, n))
    return {"W": W, "H": H}
# ============================
# 🎯 Fonction objectif
# ============================
def evaluate(indiv, X):
    W, H = indiv["W"], indiv["H"]
    diff = X - np.dot(W, H)
    return np.sum(diff ** 2)

# ============================
# 🔀 Croisement
# ============================
def crossover(parent1, parent2, crossover_rate=0.8):
    child1, child2 = {}, {}
    if random.random() < crossover_rate:
        mask_W = np.random.rand(*parent1["W"].shape) < 0.5
        mask_H = np.random.rand(*parent1["H"].shape) < 0.5
        child1["W"] = np.where(mask_W, parent1["W"], parent2["W"])
        child1["H"] = np.where(mask_H, parent1["H"], parent2["H"])
        child2["W"] = np.where(mask_W, parent2["W"], parent1["W"])
        child2["H"] = np.where(mask_H, parent2["H"], parent1["H"])
    else:
        child1, child2 = parent1.copy(), parent2.copy()
    return child1, child2

# ============================
# ⚡ Mutation
# ============================
def mutate(indiv, LW, UW, LH, UH, mutation_rate=0.1):
    W, H = indiv["W"], indiv["H"]
    if random.random() < mutation_rate:
        i, j = np.random.randint(0, W.shape[0]), np.random.randint(0, W.shape[1])
        W[i, j] += np.random.choice([-1, 1])
        W[i, j] = np.clip(W[i, j], LW, UW)
    if random.random() < mutation_rate:
        i, j = np.random.randint(0, H.shape[0]), np.random.randint(0, H.shape[1])
        H[i, j] += np.random.choice([-1, 1])
        H[i, j] = np.clip(H[i, j], LH, UH)
    indiv["W"], indiv["H"] = W, H

# ============================
# 🧬 Algorithme génétique
# ============================
def genetic_algorithm(X, m, n, r, LW, UW, LH, UH,
                      pop_size=50, generations=200,
                      crossover_rate=0.8, mutation_rate=0.1):
    population = [generate_individual(m, n, r, LW, UW, LH, UH) for _ in range(pop_size)]
    best_indiv, best_f = None, float("inf")

    for gen in range(generations):
        fitness = []
        for indiv in population:
            f = evaluate(indiv, X)
            fitness.append(f)
            if f < best_f:
                best_f = f
                best_indiv = indiv

        if gen % 10 == 0:
            print(f"Génération {gen} | meilleure erreur = {best_f:.2f}")

        # Sélection par tournoi
        new_pop = []
        for _ in range(pop_size // 2):
            parents = random.sample(population, 4)
            parents.sort(key=lambda ind: evaluate(ind, X))
            parent1, parent2 = parents[0], parents[1]
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            mutate(child1, LW, UW, LH, UH, mutation_rate)
            mutate(child2, LW, UW, LH, UH, mutation_rate)
            new_pop.extend([child1, child2])
        population = new_pop

    return best_indiv, best_f

# ============================
# 🚀 Main - Google Drive
# ============================
# Adapter ce chemin à TON Drive
input_path = "input.txt"
output_path = "output.txt"

# Lecture du fichier
X, m, n, r, LW, UW, LH, UH = read_input(input_path)

# Exécution
best_indiv, best_f = genetic_algorithm(X, m, n, r, LW, UW, LH, UH)

# Sauvegarde du résultat
write_output(output_path, best_f, best_indiv["W"], best_indiv["H"])

print("\n✅ Exécution terminée !")
print(f"Meilleure erreur trouvée : {best_f:.2f}")
print(f"Résultats enregistrés dans : {output_path}")