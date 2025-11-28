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


import numpy as np
import time

def metaheuristicRS(X, r, LW, UW, LH, UH, time_limit=None, targetF=None):
    m, n = X.shape
    rng = np.random.default_rng()

    # paramètres du petit recuit interne
    T_init = 5.0
    Tmin = 1e-3
    alpha = 0.99

    bestF = float("inf")
    bestW = None
    bestH = None

    # point de départ du timer (si utilisé)
    start_time = time.time()

    # boucle multi-start
    while True:

        # ---- condition d'arrêt par temps ----
        if time_limit is not None and (time.time() - start_time) >= time_limit:
            break

        # --- 1) solution initiale aléatoire ---
        W = rng.integers(LW, UW + 1, size=(m, r))
        H = rng.integers(LH, UH + 1, size=(r, n))

        # --- 2) recuit simulé local depuis cette solution ---
        T = T_init

        WH = W @ H
        diff = X - WH
        currentF = np.sum(diff * diff)

        # boucle interne du recuit
        while T > Tmin:

            # ---- arrêt temps si time_limit existe ----
            if time_limit is not None and (time.time() - start_time) >= time_limit:
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

                # ---- mise à jour du meilleur global ----
                if currentF < bestF:
                    bestF = currentF
                    bestW = W.copy()
                    bestH = H.copy()

                    # ---- arrêt anticipé si optimum connu ----
                    if targetF is not None and bestF <= targetF:
                        print("Arrêt anticipé : optimum atteint.")
                        return bestW, bestH

            T *= alpha

        # fin du recuit interne : multi-start relance

        # ---- sans time_limit : on sort après UN seul recuit ----
        if time_limit is None:
            break

    return bestW, bestH







def metaheuristicRSRAPID(X, r, LW, UW, LH, UH):
    """
    Recuit simulé ULTRA RAPIDE avec fobj incrémentale.
    """
    m, n = X.shape

    # ---------------- Paramètres ----------------
    T0 = 10.0
    Tmin = 0.01
    alpha = 0.9      # Refroidissement agressif
    max_iter = 800   # Beaucoup moins nécessaire grâce à l’incrémental

    rng = np.random.default_rng()

    # ---------------- Solution initiale ----------------
    W = rng.integers(LW, UW + 1, size=(m, r))
    H = rng.integers(LH, UH + 1, size=(r, n))

    # Produit WH initial
    WH = W @ H

    # Erreur initiale
    diff = X - WH
    currentF = np.sum(diff * diff)
    bestF = currentF
    bestW = W.copy()
    bestH = H.copy()

    T = T0

    # ---------------- Boucle principale ----------------
    for it in range(max_iter):
        
        # ------ Choisir la variable à modifier ------
        if rng.random() < 0.5:
            # Modifier W[i,j]
            i = rng.integers(0, m)
            j = rng.integers(0, r)
            delta = rng.choice([-1, 1])

            old_val = W[i, j]
            new_val = np.clip(old_val + delta, LW, UW)
            delta_val = new_val - old_val

            if delta_val != 0:
                # Mise à jour temporaire du WH
                row_update = delta_val * H[j, :]

                new_diff_row = diff[i, :] - row_update
                newF = currentF - np.sum(diff[i, :]**2) + np.sum(new_diff_row**2)
        else:
            # Modifier H[j,k]
            j = rng.integers(0, r)
            k = rng.integers(0, n)
            delta = rng.choice([-1, 1])

            old_val = H[j, k]
            new_val = np.clip(old_val + delta, LH, UH)
            delta_val = new_val - old_val

            if delta_val != 0:
                col_update = delta_val * W[:, j]

                new_diff_col = diff[:, k] - col_update
                newF = currentF - np.sum(diff[:, k]**2) + np.sum(new_diff_col**2)

        # ------ Critère d'acceptation ------
        if newF < currentF or np.exp(-(newF - currentF) / T) > rng.random():

            currentF = newF

            # Mise à jour réelle
            if rng.random() < 0.5:
                W[i, j] = new_val
                diff[i, :] = new_diff_row
            else:
                H[j, k] = new_val
                diff[:, k] = new_diff_col

            if currentF < bestF:
                bestF = currentF
                bestW = W.copy()
                bestH = H.copy()

        # ------ Refroidissement ------
        T *= alpha
        if T < Tmin:
            break

    return bestW, bestH



