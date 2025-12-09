import numpy as np
import time
import matplotlib.pyplot as plt


# -----------------------------
#  Fonction objectif
# -----------------------------
def fobj(X, W, H):
    """
    Retourne ||X - W H||_F^2.
    """
    R = X - W @ H
    return int((R ** 2).sum())


# -----------------------------
#  Vérification de faisabilité
# -----------------------------
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


# -----------------------------
#  Outils incrémentaux
# -----------------------------
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


# -----------------------------
#  Recherche tabou stochastique
# -----------------------------
def tabu_search_fact_in_z(
    X,
    r,
    LW,
    UW,
    LH,
    UH,
    max_iter=5000,
    tabu_tenure=10,
    max_no_improve=500,
    rng=None,
    verbose=False,
    print_every=100,
    n_candidates_W=1000,
    n_candidates_H=1000,
    use_svd_init=False,
):
    """
    Recherche tabou stochastique pour FactInZ.
    - explore seulement un sous-ensemble aléatoire du voisinage
    - inclut un affichage (print_every)
    - peut utiliser une initialisation SVD via use_svd_init=True
    """
    if rng is None:
        rng = np.random.default_rng()

    m, n = X.shape
    history = []
    # -------- Initialisation ----------
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
    # -------- Boucle principale ----------
    for it in range(1, max_iter + 1):

        best_move = None
        best_move_df = None
        best_move_newf = None

        # -------- Voisinage stochastique W ----------
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

        # -------- Voisinage stochastique H ----------
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

        # ---------- Aucun mouvement ? ----------
        if best_move is None:
            if verbose:
                print("[Tabu] Aucun mouvement admissible → arrêt")
            break

        # ---------- Appliquer le meilleur mouvement ----------
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
        if f_curr == 0:
            if verbose:
                print("[Tabu] Solution optimale trouvée (f = 0), arrêt immédiat.")
            history.append(f_curr)
            return W, H, f_curr, history

        # ---------- Mise à jour du meilleur ----------
        if f_curr < best_f:
            best_f = f_curr
            best_W, best_H = W.copy(), H.copy()
            iter_since_best = 0
        else:
            iter_since_best += 1

        history.append(f_curr)

        # ---------- Affichage ----------
        if verbose and it % print_every == 0:
            print(f"[Tabu] it={it}, f={f_curr}, best={best_f}")

        # ---------- Condition d'arrêt ----------
        if iter_since_best >= max_no_improve:
            if verbose:
                print(f"[Tabu] Arrêt — {max_no_improve} itérations sans amélioration.")
            break

    if verbose:
        print(f"[Tabu] Fin — Best f = {best_f}")

    return best_W, best_H, best_f,history



# -----------------------------
#  Interface demandée
# -----------------------------
def metaheuristic(X, r, LW, UW, LH, UH, big_instance=False):
    """
    Version multi-start :
    - petites matrices : un seul Tabu suffit
    - grandes matrices : plusieurs Tabu stochastiques, on garde le meilleur
    """
    rng = np.random.default_rng(42)

    if big_instance:
        n_restarts = 1  #adapter si on veut comparer plusieurs résultats
        max_iter = 2000 #réduire si trop lent
        tabu_tenure = 20
        max_no_improve = 500 #essai max sans amélio
        nW = 800
        nH = 800
        use_svd_init = True
    else:   
        n_restarts = 1 #adapter si on veut comparer plusieurs résultats
        max_iter = 5000 #réduire si trop lent
        tabu_tenure = 10
        max_no_improve = 800 #essai max sans amélio
        nW = 400
        nH = 400
        use_svd_init = False

    best_global_W = best_global_H = None
    best_global_f = None
    best_history = None   # <--- historique du meilleur restart

    for s in range(n_restarts):
        rng_s = np.random.default_rng(42 + s)

        W_s, H_s, f_s, hist_s = tabu_search_fact_in_z(
            X,
            r,
            LW,
            UW,
            LH,
            UH,
            max_iter=max_iter,
            tabu_tenure=tabu_tenure,
            max_no_improve=max_no_improve,
            rng=rng_s,
            verbose=True,
            print_every=200,
            n_candidates_W=nW,
            n_candidates_H=nH,
            use_svd_init=(use_svd_init and s == 0),
        )

        if best_global_f is None or f_s < best_global_f:
            best_global_f = f_s
            best_global_W, best_global_H = W_s, H_s
            best_history = hist_s          # <--- on garde l'historique associé
            print(f"[metaheuristic] restart {s} → nouvelle meilleure f = {best_global_f}")

    return best_global_W, best_global_H, best_history



# -----------------------------
#  Lecture / écriture fichier
# -----------------------------
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

    big = (m * n > 10000)

    start = time.time()
    W_best, H_best, history = metaheuristic(X, r, LW, UW, LH, UH, big_instance=big)
    end = time.time()

    f_best = fobj(X, W_best, H_best)
    print("Meilleure valeur trouvée :", f_best)
    print(f"Temps d'exécution : {end - start:.4f} secondes")
    write_solution("output_grosse.txt", f_best, W_best, H_best)

    # ---- Plot de l'historique ----
    plt.figure(figsize=(10, 4))
    plt.plot(history)
    plt.title("Évolution de l’erreur f au fil des itérations")
    plt.xlabel("Itérations")
    plt.ylabel("f(X - W H)^2")
    plt.grid(True)
    plt.show()
