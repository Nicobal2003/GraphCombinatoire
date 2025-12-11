import numpy as np
import time
import matplotlib.pyplot as plt



#  Fonction objectif donnée dans pdf
def fobj(X, W, H):
    """
    Retourne ||X - W H||_F^2.
    """
    R = X - W @ H
    return int((R ** 2).sum())



# Vérif de faisabilité donnée dans pdf

def solution_is_feasible(W, H, X, r, LW, UW, LH, UH):
    """
    Vérifie :
    - Dimensions compatibles
    - Types entiers
    - Respect des bornes
    """
    m, n = X.shape

    if W.shape != (m, r):
        return False
    if H.shape != (r, n):
        return False

    # type entier
    if not np.issubdtype(W.dtype, np.integer):
        return False
    if not np.issubdtype(H.dtype, np.integer):
        return False

    # bornes
    if not (LW <= W).all() or not (W <= UW).all():
        return False
    if not (LH <= H).all() or not (H <= UH).all():
        return False

    return True



#  utile pour incrémenter
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
    Variation de f quand W[i,k] ← W[i,k] + delta.
    Seule la ligne i de R est modifiée.
    """
    row_R = R[i, :]
    row_Hk = H[k, :]
    new_row = row_R - delta * row_Hk
    return int((new_row ** 2 - row_R ** 2).sum())


def delta_flip_H(k, j, delta, R, W):
    """
    Variation de f quand H[k,j] ← H[k,j] + delta.
    Seule la colonne j de R est modifiée.
    """
    col_R = R[:, j]
    col_Wk = W[:, k]
    new_col = col_R - delta * col_Wk
    return int((new_col ** 2 - col_R ** 2).sum())



#  Recherche tabou stochastique, grosse matrice uniquement

def tabuSto_search_fact_in_z(
    X,
    r,
    LW,
    UW,
    LH,
    UH,
    max_iter=20000,
    tabu_tenure=20,
    max_no_improve=1000,
    rng=None,
    verbose=False,
    print_every=200,
    n_candidates_W=800,
    n_candidates_H=800,
    use_svd_init=True,
):
    """
    Recherche tabou stochastique pour FactInZ,
    adaptée aux grosses matrices.
    - explore un sous-ensemble aléatoire du voisinage
    - peut utiliser une initialisation SVD (use_svd_init=True)
    - renvoie aussi l'historique des valeurs f
    """
    if rng is None:
        rng = np.random.default_rng()

    m, n = X.shape
    history = []

    # Init
    if use_svd_init:
        W, H = smart_init_svd(X, r, LW, UW, LH, UH)
    else:
        W = rng.integers(LW, UW + 1, size=(m, r))
        H = rng.integers(LH, UH + 1, size=(r, n))

    Y, R, f_curr = compute_state(X, W, H)
    best_W, best_H = W.copy(), H.copy()
    best_f = f_curr

    tabu_until_W = np.zeros((m, r), dtype=int)
    tabu_until_H = np.zeros((r, n), dtype=int)

    total_W = m * r
    total_H = r * n

    if verbose:
        print(f"[Tabu] Début — f = {f_curr}")

    iter_since_best = 0

    #  Boucle principale 
    for it in range(1, max_iter + 1):

        best_move = None
        best_move_df = None
        best_move_newf = None

        # voisinage stochastique W 
        for _ in range(min(n_candidates_W, total_W)):
            i = rng.integers(0, m)
            k = rng.integers(0, r)

            val = W[i, k]
            for delta in (-1, 1):
                new_val = val + delta
                if not (LW <= new_val <= UW):
                    continue

                df = delta_flip_W(i, k, delta, R, H)
                new_f = f_curr + df

                if tabu_until_W[i, k] > it and new_f >= best_f:
                    continue

                if best_move is None or new_f < best_move_newf:
                    best_move = ("W", i, k, delta)
                    best_move_df = df
                    best_move_newf = new_f

        #voisinage stochastique H 
        for _ in range(min(n_candidates_H, total_H)):
            k = rng.integers(0, r)
            j = rng.integers(0, n)

            val = H[k, j]
            for delta in (-1, 1):
                new_val = val + delta
                if not (LH <= new_val <= UH):
                    continue

                df = delta_flip_H(k, j, delta, R, W)
                new_f = f_curr + df

                if tabu_until_H[k, j] > it and new_f >= best_f:
                    continue

                if best_move is None or new_f < best_move_newf:
                    best_move = ("H", k, j, delta)
                    best_move_df = df
                    best_move_newf = new_f

        # Aucun mouvement 
        if best_move is None:
            if verbose:
                print("[Tabu] Aucun mouvement admissible → arrêt")
            break

        # appliquer le meilleur mouvement
        kind, a, b, delta = best_move

        if kind == "W":
            i, k = a, b
            W[i, k] += delta
            Hk = H[k, :]
            Y[i, :] += delta * Hk
            R[i, :] -= delta * Hk
            tabu_until_W[i, k] = it + tabu_tenure

        else:  # "H"
            k, j = a, b
            H[k, j] += delta
            Wk = W[:, k]
            Y[:, j] += delta * Wk
            R[:, j] -= delta * Wk
            tabu_until_H[k, j] = it + tabu_tenure

        f_curr += best_move_df

        # arrêt optimal
        if f_curr == 0:
            if verbose:
                print("[Tabu] Solution optimale trouvée (f = 0), arrêt immédiat.")
            history.append(f_curr)
            return W, H, f_curr, history

        # maj si meilleur
        if f_curr < best_f:
            best_f = f_curr
            best_W, best_H = W.copy(), H.copy()
            iter_since_best = 0
        else:
            iter_since_best += 1

        history.append(f_curr)

        # print pour suivre
        if verbose and it % print_every == 0:
            print(f"[Tabu] it={it}, f={f_curr}, best={best_f}")

        # condition d'arret 
        if iter_since_best >= max_no_improve:
            if verbose:
                print(f"[Tabu] Arrêt — {max_no_improve} itérations sans amélioration.")
            break

    if verbose:
        print(f"[Tabu] Fin — Best f = {best_f}")

    return best_W, best_H, best_f, history



# Tabu stochastique multi-start (grosse instance)

def metaheuristicTabuSto(X, r, LW, UW, LH, UH, n_restarts=1):
    best_global_W = best_global_H = None
    best_global_f = None
    best_history = None

    for s in range(n_restarts):
        rng_s = np.random.default_rng()  

        W_s, H_s, f_s, hist_s = tabuSto_search_fact_in_z(
            X,
            r,
            LW,
            UW,
            LH,
            UH,
            rng=rng_s,
            verbose=True,
        )

        if best_global_f is None or f_s < best_global_f:
            best_global_f = f_s
            best_global_W, best_global_H = W_s, H_s
            best_history = hist_s
            print(f"[metaheuristic] restart {s} → nouvelle meilleure f = {best_global_f}")

    return best_global_W, best_global_H, best_history


def smart_init_svd(X, r, LW, UW, LH, UH):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    U_r = U[:, :r]
    S_r_sqrt = np.diag(np.sqrt(s[:r]))
    V_r = Vt[:r, :]

    W_real = U_r @ S_r_sqrt
    H_real = S_r_sqrt @ V_r

    W_int = np.clip(np.rint(W_real), LW, UW).astype(int)
    H_int = np.clip(np.rint(H_real), LH, UH).astype(int)

    return W_int, H_int


if __name__ == "__main__":
    X, m, n, r, LW, UW, LH, UH = read_instance("input.txt")

    start = time.time()
    # n_restarts peut être augmenté si tu veux plusieurs essais
    W_best, H_best, history = metaheuristicTabuSto(X, r, LW, UW, LH, UH, n_restarts=1)
    end = time.time()

    f_best = fobj(X, W_best, H_best)
    print("Meilleur valeur trouvée :", f_best)
    print(f"Temps d'execution : {end - start:.4f} secondes")
    write_solution("output_grosse.txt", f_best, W_best, H_best)

    # ---- Plot de l'historique ----
    plt.figure(figsize=(10, 4))
    plt.plot(history)
    plt.title("Évolution de l erreur f au fil des itérations")
    plt.xlabel("Itérations")
    plt.ylabel("f(X - W H)^2")
    plt.grid(True)
    plt.show()
