import numpy as np
import time
from fact_in_z_tabuStochastique import smart_init_svd
#  F objectif
def fobj(X, W, H):
    """
    Retourne ||X - W H||_F^2.
    """
    R = X - W @ H
    return int((R ** 2).sum())


# Vérification de faisabilité

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


#  pour incrémenté

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



#  Recherche tabou principale

def tabu_search_fact_in_z(
    X,
    r,
    LW,
    UW,
    LH,
    UH,
    max_iter=15000,
    tabu_tenure=10,
    max_no_improve=2000,
    rng=None,
    verbose=False,
):
    """
    Recherche tabou pour FactInZ.
    Retourne aussi l'historique des valeurs de f.
    """
    if rng is None:
        rng = np.random.default_rng()

    m, n = X.shape

    # Init en respectant les bornes
    W, H = smart_init_svd(X, r, LW, UW, LH, UH)

    Y, R, f_curr = compute_state(X, W, H)
    best_W, best_H = W.copy(), H.copy()
    best_f = f_curr

    # Historique
    history = [f_curr]

    # Liste tabou 
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

        # voisinage sur W 
        for (i, k) in indicesW:
            val = W[i, k]
            for delta in (-1, 1):
                new_val = val + delta
                if new_val < LW or new_val > UW:
                    continue

                df = delta_flip_W(i, k, delta, R, H)
                new_f = f_curr + df

                is_tabu = tabu_until_W[i, k] > it

                if is_tabu and new_f >= best_f:
                    continue

                if best_move is None or new_f < best_move_newf:
                    best_move = ("W", i, k, delta)
                    best_move_df = df
                    best_move_newf = new_f

        # voisinage sur H
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
                print("Aucun mouvement admissible à l itération", it)
            break

        # meilleur mouvement
        kind, a, b, delta = best_move

        if kind == "W":
            i, k = a, b
            W[i, k] += delta

            # Mise à jour de Y et R sur la ligne i
            Hk = H[k, :]
            Y[i, :] += delta * Hk
            R[i, :] -= delta * Hk

            tabu_until_W[i, k] = it + tabu_tenure

        else:  # kind == "H"
            k, j = a, b
            H[k, j] += delta

            # Mise à jour de Y et R sur la colonne j
            Wk = W[:, k]
            Y[:, j] += delta * Wk
            R[:, j] -= delta * Wk

            tabu_until_H[k, j] = it + tabu_tenure

        f_curr += best_move_df
        history.append(f_curr)

        # Arrêt immédiat si f = 0 (optionnel mais pratique)
        if f_curr == 0:
            if verbose:
                print(f"it={it}, f=0 → solution optimale trouvée, arrêt.")
            best_f = 0
            best_W, best_H = W.copy(), H.copy()
            break

        # maj du meilleur trouvé
        if f_curr < best_f:
            best_f = f_curr
            best_W, best_H = W.copy(), H.copy()
            iter_since_best = 0
        else:
            iter_since_best += 1

        if verbose and it % 100 == 0:
            print(f"it={it}, f={f_curr}, best={best_f}")

        if iter_since_best >= max_no_improve:
            if verbose:
                print(
                    f"Aucune amélioration depuis {max_no_improve} itérations, arrêt."
                )
            break

    return best_W, best_H, best_f, history


#  Interface demandée (multi-start)
def metaheuristicTabu(X, r, LW, UW, LH, UH, n_restarts=10): #multi start a changer ici
    """
    Fonction appelée dans le projet : lance la recherche tabou
    en mode multi-start et renvoie (W_best, H_best, history_best).
    """
    best_global_W = None
    best_global_H = None
    best_global_f = None
    best_history = None

    for s in range(n_restarts):
        # Nouveau générateur aléatoire à chaque restart (pas de graine fixe)
        rng = np.random.default_rng()

        W_s, H_s, f_s, hist_s = tabu_search_fact_in_z(
            X,
            r,
            LW,
            UW,
            LH,
            UH,
            max_iter=3000,
            tabu_tenure=10,
            max_no_improve=1000,
            rng=rng,
            verbose=False,
        )

        print(f"[multi-start] restart {s}: f = {f_s}")

        if best_global_f is None or f_s < best_global_f:
            best_global_f = f_s
            best_global_W = W_s
            best_global_H = H_s
            best_history = hist_s
            print(f" → nouvelle meilleure solution globale: f = {best_global_f}")

    return best_global_W, best_global_H, best_history