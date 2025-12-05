import numpy as np
import time

def fobj(X, W, H):
    # ou alors, sans NumPy : f = sum(sum((X-W@H)**2))
    f = np.linalg.norm(X - W @ H, 'fro')**2
    return f


def solutionIsFeasible(W, H, r, LW, UW, LH, UH):
    if W.shape[1] != r or H.shape[0] != r:
        return False

    Wi = np.issubdtype(W.dtype, np.integer) or (
        np.issubdtype(W.dtype, np.floating)
        and np.all(np.isfinite(W))
        and np.all(W == np.floor(W))
    )

    Hi = np.issubdtype(H.dtype, np.integer) or (
        np.issubdtype(H.dtype, np.floating)
        and np.all(np.isfinite(H))
        and np.all(H == np.floor(H))
    )

    if not (Wi and Hi):
        return False

    return np.all((W >= LW) & (W <= UW)) and np.all((H >= LH) & (H <= UH))


def metaheuristicVNS(X, r, LW, UW, LH, UH):
    """
    VNS pour la factorisation entière X ≈ W @ H
    """
    m, n = X.shape

    # ---------- 1) solution initiale ----------
    def initial_solution():
        W = np.random.randint(LW, UW + 1, size=(m, r))
        H = np.random.randint(LH, UH + 1, size=(r, n))
        return W, H

    # ---------- 2) shaking : voisinage N_k ----------
    def shake(W, H, k):
        W_new = W.copy()
        H_new = H.copy()

        for _ in range(k):
            if np.random.rand() < 0.5:
                i = np.random.randint(m)
                j = np.random.randint(r)
                delta = np.random.choice([-1, 1])
                W_new[i, j] = np.clip(W_new[i, j] + delta, LW, UW)
            else:
                i = np.random.randint(r)
                j = np.random.randint(n)
                delta = np.random.choice([-1, 1])
                H_new[i, j] = np.clip(H_new[i, j] + delta, LH, UH)

        return W_new, H_new

    # ---------- 3) recherche locale (voisinage de base N1) ----------
    def local_search(W, H, max_ls_iter=500):
        bestW = W.copy()
        bestH = H.copy()
        bestF = fobj(X, bestW, bestH)

        for _ in range(max_ls_iter):
            Wn, Hn = shake(bestW, bestH, k=1)
            Fn = fobj(X, Wn, Hn)

            if Fn < bestF:
                bestF = Fn
                bestW = Wn
                bestH = Hn

        return bestW, bestH, bestF

    # ---------- 4) paramètres VNS ----------
    kmax = 3          # nombre de voisinages différents, adaptable ici
    max_iter = 50     # nombre d'itérations VNS, adaptable ici
    max_ls_iter = 300 # intensité de la recherche locale, adaptable ici

    # ---------- 5) VNS principal ----------
    W0, H0 = initial_solution()
    bestW, bestH, bestF = local_search(W0, H0, max_ls_iter=max_ls_iter)

    for _ in range(max_iter):
        k = 1
        while k <= kmax:
            Wk, Hk = shake(bestW, bestH, k)

            Wloc, Hloc, Floc = local_search(Wk, Hk, max_ls_iter=max_ls_iter // 3)

            if Floc < bestF:
                bestF = Floc
                bestW = Wloc
                bestH = Hloc
                k = 1     
            else:
                k += 1    

    return bestW, bestH


def metaheuristicRS(X, r, LW, UW, LH, UH, time_limit=None):
    m, n = X.shape
    rng = np.random.default_rng()

    # paramètres du recuit interne
    T_init = 10.0
    Tmin = 1e-4
    alpha = 0.99

    bestF = float("inf")
    bestW = None
    bestH = None

    # timer uniquement si time_limit est fourni
    use_timer = time_limit is not None
    if use_timer:
        start_time = time.time()

    while True:

        # ---- arrêt par temps (si activé) ----
        if use_timer and (time.time() - start_time) >= time_limit:
            break

        # --- 1) solution initiale aléatoire ---
        W = rng.integers(LW, UW + 1, size=(m, r))
        H = rng.integers(LH, UH + 1, size=(r, n))

        # --- 2) recuit simulé local ---
        T = T_init

        WH = W @ H
        diff = X - WH
        currentF = np.sum(diff * diff)

        while T > Tmin:

            # ---- arrêt par temps si activé ----
            if use_timer and (time.time() - start_time) >= time_limit:
                return bestW, bestH

            modify_W = rng.random() < 0.5

            if modify_W:
                i = rng.integers(0, m)
                j = rng.integers(0, r)
                old_val = W[i, j]
                delta = rng.choice([-1, 1])
                new_val = int(np.clip(old_val + delta, LW, UW))
                d = new_val - old_val
                if d == 0:
                    T *= alpha
                    continue

                new_diff_row = diff[i, :] - d * H[j, :]
                old_row_sq = np.sum(diff[i, :] ** 2)
                new_row_sq = np.sum(new_diff_row ** 2)
                newF = currentF - old_row_sq + new_row_sq

            else:
                j = rng.integers(0, r)
                k = rng.integers(0, n)
                old_val = H[j, k]
                delta = rng.choice([-1, 1])
                new_val = int(np.clip(old_val + delta, LH, UH))
                d = new_val - old_val
                if d == 0:
                    T *= alpha
                    continue

                new_diff_col = diff[:, k] - d * W[:, j]
                old_col_sq = np.sum(diff[:, k] ** 2)
                new_col_sq = np.sum(new_diff_col ** 2)
                newF = currentF - old_col_sq + new_col_sq

            # ---- acceptation ----
            accept = False
            if newF < currentF:
                accept = True
            else:
                if rng.random() < np.exp(-(newF - currentF) / T):
                    accept = True

            if accept:
                currentF = newF
                if modify_W:
                    W[i, j] = new_val
                    diff[i, :] = new_diff_row
                else:
                    H[j, k] = new_val
                    diff[:, k] = new_diff_col

                # meilleur global
                if currentF < bestF:
                    bestF = currentF
                    bestW = W.copy()
                    bestH = H.copy()

            T *= alpha

        # fin d’un recuit pour ce point de départ

        # sans time_limit : un seul recuit, puis on sort
        if not use_timer:
            break

    return bestW, bestH


import numpy as np
import time

def metaheuristicRS2(
    X, r, LW, UW, LH, UH,
    time_limit=None,
    restart_guided_prob=0.5,  # proba de repartir de bestW/bestH
    noise_rate_W=0.05,        # % d'entrées de W à perturber
    noise_rate_H=0.05         # % d'entrées de H à perturber
):
    m, n = X.shape
    rng = np.random.default_rng()

    # paramètres du recuit interne (tu peux les ajuster si besoin)
    T_init = 5.0
    Tmin   = 1e-3
    alpha  = 0.99

    bestF = float("inf")
    bestW = None
    bestH = None

    use_timer = time_limit is not None
    if use_timer:
        start_time = time.time()

    while True:

        # ---- arrêt par temps (si activé) ----
        if use_timer and (time.time() - start_time) >= time_limit:
            break

        # ========= 1) INITIALISATION DE W, H =========
        if bestW is not None and rng.random() < restart_guided_prob:
            # --- restart guidé : on part du meilleur + bruit ---
            W = bestW.copy()
            H = bestH.copy()

            # nombre de perturbations
            num_changes_W = max(1, int(noise_rate_W * m * r))
            num_changes_H = max(1, int(noise_rate_H * r * n))

            # bruit sur W
            for _ in range(num_changes_W):
                i = rng.integers(0, m)
                j = rng.integers(0, r)
                delta = rng.choice([-1, 1])
                W[i, j] = int(np.clip(W[i, j] + delta, LW, UW))

            # bruit sur H
            for _ in range(num_changes_H):
                j = rng.integers(0, r)
                k = rng.integers(0, n)
                delta = rng.choice([-1, 1])
                H[j, k] = int(np.clip(H[j, k] + delta, LH, UH))

        else:
            # --- restart aléatoire classique ---
            W = rng.integers(LW, UW + 1, size=(m, r))
            H = rng.integers(LH, UH + 1, size=(r, n))

        # ========= 2) RECALCUL INITIAL =========
        WH = W @ H
        diff = X - WH
        currentF = np.sum(diff * diff)

        # ========= 3) RECUIT SIMULÉ LOCAL =========
        T = T_init

        while T > Tmin:

            if use_timer and (time.time() - start_time) >= time_limit:
                return bestW, bestH

            modify_W = rng.random() < 0.5

            if modify_W:
                i = rng.integers(0, m)
                j = rng.integers(0, r)
                old_val = W[i, j]
                delta = rng.choice([-1, 1])
                new_val = int(np.clip(old_val + delta, LW, UW))
                d = new_val - old_val
                if d == 0:
                    T *= alpha
                    continue

                # mise à jour locale de diff et du coût
                new_diff_row = diff[i, :] - d * H[j, :]
                old_row_sq = np.sum(diff[i, :] ** 2)
                new_row_sq = np.sum(new_diff_row ** 2)
                newF = currentF - old_row_sq + new_row_sq

            else:
                j = rng.integers(0, r)
                k = rng.integers(0, n)
                old_val = H[j, k]
                delta = rng.choice([-1, 1])
                new_val = int(np.clip(old_val + delta, LH, UH))
                d = new_val - old_val
                if d == 0:
                    T *= alpha
                    continue

                new_diff_col = diff[:, k] - d * W[:, j]
                old_col_sq = np.sum(diff[:, k] ** 2)
                new_col_sq = np.sum(new_diff_col ** 2)
                newF = currentF - old_col_sq + new_col_sq

            # ---- acceptation ----
            accept = False
            if newF < currentF:
                accept = True
            else:
                if rng.random() < np.exp(-(newF - currentF) / T):
                    accept = True

            if accept:
                currentF = newF
                if modify_W:
                    W[i, j] = new_val
                    diff[i, :] = new_diff_row
                else:
                    H[j, k] = new_val
                    diff[:, k] = new_diff_col

                # mise à jour du meilleur global
                if currentF < bestF:
                    bestF = currentF
                    bestW = W.copy()
                    bestH = H.copy()

            T *= alpha

        # fin du recuit pour ce restart

        # sans time_limit : on arrête après un seul recuit
        if not use_timer:
            break

    return bestW, bestH








