import numpy as np
import time


def _local_descent(
    X,
    W,
    H,
    LW,
    UW,
    LH,
    UH,
    max_ls_iters=50,
    moves_per_iter=2000,
    rng=None,
    time_limit=None,
    start_time=None,
):
    """
    Descente locale stochastique (première amélioration) :
    - à chaque itération, on essaie 'moves_per_iter' mouvements aléatoires
    - dès qu'on trouve un mouvement améliorant, on l'applique et on repart
    - on s'arrête si plus d'amélioration ou si max_ls_iters atteint
    """
    if rng is None:
        rng = np.random.default_rng()

    m, r = W.shape
    r2, n = H.shape
    assert r == r2

    # calcul initial
    WH = W @ H
    diff = X - WH
    currentF = float(np.sum(diff * diff))

    for _ in range(max_ls_iters):
        improved = False

        # contrôle temps
        if time_limit is not None and (time.time() - start_time) >= time_limit:
            break

        for _ in range(moves_per_iter):

            # contrôle temps
            if time_limit is not None and (time.time() - start_time) >= time_limit:
                break

            modify_W = rng.random() < 0.5

            if modify_W:
                # mouvement sur W[i,j]
                i = rng.integers(0, m)
                j = rng.integers(0, r)
                old_val = W[i, j]
                delta = rng.choice([-1, 1])
                new_val = int(np.clip(old_val + delta, LW, UW))
                d = new_val - old_val
                if d == 0:
                    continue

                # mise à jour locale de la ligne i de diff
                new_diff_row = diff[i, :] - d * H[j, :]
                old_row_sq = float(np.sum(diff[i, :] ** 2))
                new_row_sq = float(np.sum(new_diff_row ** 2))
                newF = currentF - old_row_sq + new_row_sq

                if newF < currentF:
                    # on accepte immédiatement (première amélioration)
                    W[i, j] = new_val
                    diff[i, :] = new_diff_row
                    currentF = newF
                    improved = True
                    break

            else:
                # mouvement sur H[j,k]
                j = rng.integers(0, r)
                k = rng.integers(0, n)
                old_val = H[j, k]
                delta = rng.choice([-1, 1])
                new_val = int(np.clip(old_val + delta, LH, UH))
                d = new_val - old_val
                if d == 0:
                    continue

                # mise à jour locale de la colonne k de diff
                new_diff_col = diff[:, k] - d * W[:, j]
                old_col_sq = float(np.sum(diff[:, k] ** 2))
                new_col_sq = float(np.sum(new_diff_col ** 2))
                newF = currentF - old_col_sq + new_col_sq

                if newF < currentF:
                    H[j, k] = new_val
                    diff[:, k] = new_diff_col
                    currentF = newF
                    improved = True
                    break

        if not improved:
            # aucun move améliorant trouvé dans ce tour
            break

    return W, H, currentF


def _shake(W, H, LW, UW, LH, UH, k, rng):
    """
    Shaking : on applique k petites perturbations au hasard
    (sans se soucier d'améliorer ou non).
    """
    m, r = W.shape
    r2, n = H.shape
    assert r == r2

    W2 = W.copy()
    H2 = H.copy()

    for _ in range(k):
        modify_W = rng.random() < 0.5
        if modify_W:
            i = rng.integers(0, m)
            j = rng.integers(0, r)
            delta = rng.choice([-1, 1])
            W2[i, j] = int(np.clip(W2[i, j] + delta, LW, UW))
        else:
            j = rng.integers(0, r)
            k_idx = rng.integers(0, n)
            delta = rng.choice([-1, 1])
            H2[j, k_idx] = int(np.clip(H2[j, k_idx] + delta, LH, UH))

    return W2, H2


def vns_factorization(
    X,
    r,
    LW,
    UW,
    LH,
    UH,
    k_max=5,
    max_ls_iters=50,
    moves_per_iter=2000,
    max_no_improv=20,
    time_limit=None,
    seed=None,
):
    """
    VNS pour factorisation entière X ≈ W H.

    Paramètres recommandés (matrices jusqu'à ~500x700) :
      - k_max : 3 à 7
      - max_ls_iters : 20 à 80
      - moves_per_iter : 1000 à 10000 (plus grand = plus intensif = plus lent)
      - time_limit : limite dure en secondes (ex: 60, 300, ...)
    """
    rng = np.random.default_rng(seed)
    m, n = X.shape

    if time_limit is not None:
        start_time = time.time()
    else:
        # valeur fictive, jamais utilisée si pas de time_limit
        start_time = 0.0

    # --------- solution initiale aléatoire ---------
    W = rng.integers(LW, UW + 1, size=(m, r))
    H = rng.integers(LH, UH + 1, size=(r, n))

    W, H, currentF = _local_descent(
        X, W, H, LW, UW, LH, UH,
        max_ls_iters=max_ls_iters,
        moves_per_iter=moves_per_iter,
        rng=rng,
        time_limit=time_limit,
        start_time=start_time,
    )

    bestW = W.copy()
    bestH = H.copy()
    bestF = currentF

    no_improv = 0

    # --------- boucle principale VNS ---------
    while True:
        if time_limit is not None and (time.time() - start_time) >= time_limit:
            break

        improved_outer = False
        k = 1

        while k <= k_max:
            if time_limit is not None and (time.time() - start_time) >= time_limit:
                break

            # 1) shaking à intensité k
            W_sh, H_sh = _shake(bestW, bestH, LW, UW, LH, UH, k, rng)

            # 2) descente locale depuis la solution secouée
            W_loc, H_loc, F_loc = _local_descent(
                X, W_sh, H_sh, LW, UW, LH, UH,
                max_ls_iters=max_ls_iters,
                moves_per_iter=moves_per_iter,
                rng=rng,
                time_limit=time_limit,
                start_time=start_time,
            )

            # 3) mise à jour du meilleur
            if F_loc < bestF:
                bestF = F_loc
                bestW = W_loc.copy()
                bestH = H_loc.copy()
                improved_outer = True
                k = 1  # on repart du plus petit voisinage
            else:
                k += 1

        if not improved_outer:
            no_improv += 1
            if (max_no_improv is not None) and (no_improv >= max_no_improv):
                break
        else:
            no_improv = 0

    return bestW, bestH, bestF
