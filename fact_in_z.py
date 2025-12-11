import numpy as np
import time


# ==========================================================
#  Fonction objectif et outils de base
# ==========================================================

def fobj(X, W, H):
    """
    ||X - W H||_F^2
    """
    R = X - W @ H
    return int((R ** 2).sum())


def compute_state(X, W, H):
    """
    Calcule Y = W H, R = X - Y, f = ||R||^2.
    """
    Y = W @ H
    R = X - Y
    f = int((R ** 2).sum())
    return Y, R, f


def delta_flip_W(i, k, delta, R, H):
    """
    Variation de f quand W[i,k] <- W[i,k] + delta.
    Seule la ligne i de R est modifiée.
    """
    row_R = R[i, :]
    row_Hk = H[k, :]
    new_row = row_R - delta * row_Hk
    return int((new_row ** 2 - row_R ** 2).sum())


def delta_flip_H(k, j, delta, R, W):
    """
    Variation de f quand H[k,j] <- H[k,j] + delta.
    Seule la colonne j de R est modifiée.
    """
    col_R = R[:, j]
    col_Wk = W[:, k]
    new_col = col_R - delta * col_Wk
    return int((new_col ** 2 - col_R ** 2).sum())


# ==========================================================
#  Recherche Tabou (Tabu Search)
# ==========================================================

def tabu_search_fact_in_z(
    X,
    r,
    LW,
    UW,
    LH,
    UH,
    max_iter=1000,
    tabu_tenure=10,
    max_no_improve=200,
    rng=None,
    verbose=False,
    W_init=None,
    H_init=None,
):
    """
    Recherche tabou pour FactInZ.
    Si W_init et H_init sont fournis, ils sont utilisés comme solution de départ
    (pratique pour l'hybridation avec un AG).
    """
    if rng is None:
        rng = np.random.default_rng()

    m, n = X.shape

    # Initialisation
    if W_init is None or H_init is None:
        # Init autour de 0 (mieux que full [-16,16] pour les grandes matrices)
        init_scale = 2
        lowW = max(LW, -init_scale)
        highW = min(UW, init_scale)
        lowH = max(LH, -init_scale)
        highH = min(UH, init_scale)
        W = rng.integers(lowW, highW + 1, size=(m, r))
        H = rng.integers(lowH, highH + 1, size=(r, n))
    else:
        W = W_init.copy()
        H = H_init.copy()

    Y, R, f_curr = compute_state(X, W, H)
    best_W, best_H = W.copy(), H.copy()
    best_f = f_curr

    tabu_until_W = np.zeros((m, r), dtype=int)
    tabu_until_H = np.zeros((r, n), dtype=int)

    iter_since_best = 0

    for it in range(1, max_iter + 1):
        best_move = None
        best_move_df = None
        best_move_newf = None

        # On parcourt les indices dans un ordre aléatoire
        indicesW = [(i, k) for i in range(m) for k in range(r)]
        rng.shuffle(indicesW)
        indicesH = [(k, j) for k in range(r) for j in range(n)]
        rng.shuffle(indicesH)

        # ----- Voisinage sur W -----
        for (i, k) in indicesW:
            val = W[i, k]
            for delta in (-1, 1):
                new_val = val + delta
                if new_val < LW or new_val > UW:
                    continue

                df = delta_flip_W(i, k, delta, R, H)
                new_f = f_curr + df

                is_tabu = tabu_until_W[i, k] > it
                # Critère d'aspiration : accepter si améliore la meilleure solution globale
                if is_tabu and new_f >= best_f:
                    continue

                if best_move is None or new_f < best_move_newf:
                    best_move = ("W", i, k, delta)
                    best_move_df = df
                    best_move_newf = new_f

        # ----- Voisinage sur H -----
        for (k, j) in indicesH:
            val = H[k, j]
            for delta in (-1, 1):
                new_val = val + delta
                if new_val < LH or new_val > UH:
                    continue

                df = delta_flip_H(k, j, delta, R, W)
                new_f = f_curr + df

                is_tabu = tabu_until_H[k, j] > it
                if is_tabu and new_f >= best_f:
                    continue

                if best_move is None or new_f < best_move_newf:
                    best_move = ("H", k, j, delta)
                    best_move_df = df
                    best_move_newf = new_f

        if best_move is None:
            if verbose:
                print("Aucun mouvement admissible à l’itération", it)
            break

        # ----- Appliquer le meilleur mouvement -----
        kind, a, b, delta = best_move

        if kind == "W":
            i, k = a, b
            W[i, k] += delta
            Hk = H[k, :]
            Y[i, :] += delta * Hk
            R[i, :] -= delta * Hk
            tabu_until_W[i, k] = it + tabu_tenure
        else:
            k, j = a, b
            H[k, j] += delta
            Wk = W[:, k]
            Y[:, j] += delta * Wk
            R[:, j] -= delta * Wk
            tabu_until_H[k, j] = it + tabu_tenure

        f_curr += best_move_df

        if f_curr < best_f:
            best_f = f_curr
            best_W, best_H = W.copy(), H.copy()
            iter_since_best = 0
        else:
            iter_since_best += 1

        if verbose and it % 100 == 0:
            print(f"[Tabu] it={it}, f={f_curr}, best={best_f}")

        if iter_since_best >= max_no_improve:
            if verbose:
                print(f"Stop Tabu (pas d'amélioration depuis {max_no_improve} itérations).")
            break

    return best_W, best_H, best_f


# ==========================================================
#  Algorithme Génétique (AG) pour FactInZ
# ==========================================================

class Individual:
    """
    Individu de la population (AG) : solution (W,H) + valeur de f.
    """
    def __init__(self, W, H, f):
        self.W = W
        self.H = H
        self.f = f


def init_individual(X, r, LW, UW, LH, UH, rng):
    """
    Crée un individu aléatoire (W,H) dans les bornes, avec f.
    Init limitée autour de 0 pour éviter les valeurs énormes.
    """
    m, n = X.shape
    init_scale = 3
    lowW = max(LW, -init_scale)
    highW = min(UW, init_scale)
    lowH = max(LH, -init_scale)
    highH = min(UH, init_scale)

    W = rng.integers(lowW, highW + 1, size=(m, r))
    H = rng.integers(lowH, highH + 1, size=(r, n))
    f = fobj(X, W, H)
    return Individual(W, H, f)


def init_population(X, r, LW, UW, LH, UH, pop_size, rng):
    return [init_individual(X, r, LW, UW, LH, UH, rng) for _ in range(pop_size)]


def tournament_selection(population, k, rng):
    """
    Sélection par tournoi : on tire k individus au hasard,
    on retourne le meilleur (f le plus petit).
    """
    candidates = rng.choice(population, size=k, replace=False)
    best = min(candidates, key=lambda ind: ind.f)
    return best


def crossover(parent1, parent2, rng, p_swap_rows=0.5, p_swap_cols=0.5):
    """
    Croisement simple :
    - pour W : on échange certaines lignes entre les parents
    - pour H : on échange certaines colonnes
    """
    W1, H1 = parent1.W, parent1.H
    W2, H2 = parent2.W, parent2.H

    m, r = W1.shape
    r2, n = H1.shape
    assert r == r2

    child1_W = W1.copy()
    child2_W = W2.copy()
    for i in range(m):
        if rng.random() < p_swap_rows:
            child1_W[i, :], child2_W[i, :] = child2_W[i, :].copy(), child1_W[i, :].copy()

    child1_H = H1.copy()
    child2_H = H2.copy()
    for j in range(n):
        if rng.random() < p_swap_cols:
            child1_H[:, j], child2_H[:, j] = child2_H[:, j].copy(), child1_H[:, j].copy()

    return child1_W, child1_H, child2_W, child2_H


def mutate(individual, LW, UW, LH, UH, rng,
           prob_mut_W=0.001, prob_mut_H=0.001):
    """
    Mutation : pour chaque gène, avec une petite probabilité,
    on ajoute -1 ou +1, en respectant les bornes.
    """
    W = individual.W
    H = individual.H
    m, r = W.shape
    r2, n = H.shape
    assert r == r2

    # Mutation sur W
    for i in range(m):
        for k in range(r):
            if rng.random() < prob_mut_W:
                delta = rng.choice([-1, 1])
                new_val = W[i, k] + delta
                if LW <= new_val <= UW:
                    W[i, k] = new_val

    # Mutation sur H
    for k in range(r):
        for j in range(n):
            if rng.random() < prob_mut_H:
                delta = rng.choice([-1, 1])
                new_val = H[k, j] + delta
                if LH <= new_val <= UH:
                    H[k, j] = new_val

    # Recalcule le f
    return individual


def evaluate_individual(ind, X):
    ind.f = fobj(X, ind.W, ind.H)


def genetic_algorithm_fact_in_z(
    X,
    r,
    LW,
    UW,
    LH,
    UH,
    pop_size=20,
    n_generations=50,
    tournament_k=3,
    crossover_rate=0.9,
    mutation_prob_W=0.001,
    mutation_prob_H=0.001,
    rng=None,
    verbose=False,
):
    """
    Algorithme génétique simple pour FactInZ.
    Retourne le meilleur individu (W,H).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Initialisation de la population
    population = init_population(X, r, LW, UW, LH, UH, pop_size, rng)
    best = min(population, key=lambda ind: ind.f)

    if verbose:
        print(f"[AG] Génération 0, meilleure f = {best.f}")

    for gen in range(1, n_generations + 1):
        new_population = []

        while len(new_population) < pop_size:
            # Sélection de deux parents par tournoi
            p1 = tournament_selection(population, tournament_k, rng)
            p2 = tournament_selection(population, tournament_k, rng)

            # Croisement
            if rng.random() < crossover_rate:
                Wc1, Hc1, Wc2, Hc2 = crossover(p1, p2, rng)
            else:
                Wc1, Hc1 = p1.W.copy(), p1.H.copy()
                Wc2, Hc2 = p2.W.copy(), p2.H.copy()

            c1 = Individual(Wc1, Hc1, None)
            c2 = Individual(Wc2, Hc2, None)

            # Mutation
            mutate(c1, LW, UW, LH, UH, rng, mutation_prob_W, mutation_prob_H)
            mutate(c2, LW, UW, LH, UH, rng, mutation_prob_W, mutation_prob_H)

            # Évaluation
            evaluate_individual(c1, X)
            evaluate_individual(c2, X)

            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        # Elitisme : on garde le meilleur ancien individu
        new_population.append(best)
        # On garde les pop_size meilleurs
        new_population = sorted(new_population, key=lambda ind: ind.f)[:pop_size]

        population = new_population
        best = min(population, key=lambda ind: ind.f)

        if verbose and gen % 5 == 0:
            print(f"[AG] Génération {gen}, meilleure f = {best.f}")

    return best.W.copy(), best.H.copy(), best.f


# ==========================================================
#  Hybridation : AG + Tabu Search (algorithme mémétique)
# ==========================================================

def hybrid_ga_tabu(
    X,
    r,
    LW,
    UW,
    LH,
    UH,
    rng=None,
    verbose=True,
):
    """
    Hybridation simple :
    1) Algorithme génétique pour trouver une bonne solution globale
    2) Recherche Tabou pour affiner cette solution localement
    """
    if rng is None:
        rng = np.random.default_rng()

    # ----- Phase 1 : AG -----
    if verbose:
        print("=== Phase 1 : Algorithme génétique ===")

    W_ga, H_ga, f_ga = genetic_algorithm_fact_in_z(
        X, r, LW, UW, LH, UH,
        pop_size=15,
        n_generations=40,
        tournament_k=3,
        crossover_rate=0.9,
        mutation_prob_W=0.001,
        mutation_prob_H=0.001,
        rng=rng,
        verbose=verbose,
    )

    if verbose:
        print(f"[Fin AG] meilleure f = {f_ga}")

    # ----- Phase 2 : Tabu Search depuis la meilleure solution de l'AG -----
    if verbose:
        print("=== Phase 2 : Recherche Tabou à partir de la solution AG ===")

    W_best, H_best, f_best = tabu_search_fact_in_z(
        X, r, LW, UW, LH, UH,
        max_iter=2000,
        tabu_tenure=15,
        max_no_improve=400,
        rng=rng,
        verbose=verbose,
        W_init=W_ga,
        H_init=H_ga,
    )

    if verbose:
        print(f"[Fin Tabu] f_ga = {f_ga} → f_hybride = {f_best}")

    return W_best, H_best, f_best


# ==========================================================
#  Lecture / écriture fichiers
# ==========================================================

def read_instance(path):
    with open(path, "r") as f:
        lines = f.read().strip().splitlines()
    m, n, r, LW, UW, LH, UH = map(int, lines[0].split())
    X = np.array([[int(x) for x in ln.split()] for ln in lines[1:]])
    return X, m, n, r, LW, UW, LH, UH


def write_solution(path, fval, W, H):
    lines = []
    lines.append(str(fval))
    for i in range(W.shape[0]):
        lines.append(" ".join(str(int(v)) for v in W[i]))
    for i in range(H.shape[0]):
        lines.append(" ".join(str(int(v)) for v in H[i]))
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ==========================================================
#  Main : exemple avec un fichier d'instance
# ==========================================================

if __name__ == "__main__":
    # Remplace le nom du fichier par ton instance :
    # - petite : "input.txt"
    # - grosse : "testmatrice3_houdain_547x745_r10_m1616m1616.txt"
    instance_file = "testmatrice3_houdain_547x745_r10_m1616m1616.txt"

    X, m, n, r, LW, UW, LH, UH = read_instance(instance_file)

    print(f"Instance : {m} x {n}, r={r}, W in [{LW},{UW}], H in [{LH},{UH}]")

    start = time.time()
    W_best, H_best, f_best = hybrid_ga_tabu(X, r, LW, UW, LH, UH, verbose=True)
    end = time.time()

    print("\n=== Résultat final (AG + Tabu) ===")
    print("Meilleure valeur trouvée :", f_best)
    print(f"Temps d'exécution : {end - start:.2f} secondes")

    write_solution("output_hybride.txt", f_best, W_best, H_best)
