import numpy as np
import time

def local_descent(X, W, H, LW, UW, LH, UH):
    """
    Descente locale stricte :
    - parcourt toutes les entrées de W et H
    - teste +/-1
    - garde tous les mouvements qui améliorent F
    - stop quand plus aucune amélioration
    """
    m, r = W.shape
    r2, n = H.shape
    assert r == r2

    WH = W @ H
    diff = X - WH
    currentF = np.sum(diff * diff)

    improved = True
    while improved:
        improved = False

        # --- phase sur W ---
        for i in range(m):
            for j in range(r):
                old_val = W[i, j]
                best_val = old_val
                bestF_ij = currentF
                best_diff_row = diff[i, :]

                # tester -1 puis +1 (ou l'inverse)
                for delta in (-1, 1):
                    new_val = int(np.clip(old_val + delta, LW, UW))
                    if new_val == old_val:
                        continue
                    d = new_val - old_val

                    new_diff_row = diff[i, :] - d * H[j, :]
                    old_row_sq = np.sum(diff[i, :] ** 2)
                    new_row_sq = np.sum(new_diff_row ** 2)
                    newF = currentF - old_row_sq + new_row_sq

                    if newF < bestF_ij:
                        bestF_ij = newF
                        best_val = new_val
                        best_diff_row = new_diff_row

                if best_val != old_val:
                    # appliquer la meilleure modif trouvée pour (i,j)
                    W[i, j] = best_val
                    diff[i, :] = best_diff_row
                    currentF = bestF_ij
                    improved = True

        # --- phase sur H ---
        for j in range(r):
            for k in range(n):
                old_val = H[j, k]
                best_val = old_val
                bestF_jk = currentF
                best_diff_col = diff[:, k]

                for delta in (-1, 1):
                    new_val = int(np.clip(old_val + delta, LH, UH))
                    if new_val == old_val:
                        continue
                    d = new_val - old_val

                    new_diff_col = diff[:, k] - d * W[:, j]
                    old_col_sq = np.sum(diff[:, k] ** 2)
                    new_col_sq = np.sum(new_diff_col ** 2)
                    newF = currentF - old_col_sq + new_col_sq

                    if newF < bestF_jk:
                        bestF_jk = newF
                        best_val = new_val
                        best_diff_col = new_diff_col

                if best_val != old_val:
                    H[j, k] = best_val
                    diff[:, k] = best_diff_col
                    currentF = bestF_jk
                    improved = True

    return W, H, currentF


def metaheuristicRStest(
    X, r, LW, UW, LH, UH,
    time_limit=None,
    restart_guided_prob=0.5,
    noise_rate_W=0.05,
    noise_rate_H=0.05
):
    m, n = X.shape
    rng = np.random.default_rng()

    # paramètres du recuit
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

        if use_timer and (time.time() - start_time) >= time_limit:
            break

        # ===== 1) init W, H (guided ou aléatoire) =====
        if bestW is not None and rng.random() < restart_guided_prob:
            W = bestW.copy()
            H = bestH.copy()

            num_changes_W = max(1, int(noise_rate_W * m * r))
            num_changes_H = max(1, int(noise_rate_H * r * n))

            for _ in range(num_changes_W):
                i = rng.integers(0, m)
                j = rng.integers(0, r)
                delta = rng.choice([-1, 1])
                W[i, j] = int(np.clip(W[i, j] + delta, LW, UW))

            for _ in range(num_changes_H):
                j = rng.integers(0, r)
                k = rng.integers(0, n)
                delta = rng.choice([-1, 1])
                H[j, k] = int(np.clip(H[j, k] + delta, LH, UH))
        else:
            W = rng.integers(LW, UW + 1, size=(m, r))
            H = rng.integers(LH, UH + 1, size=(r, n))

        WH = W @ H
        diff = X - WH
        currentF = np.sum(diff * diff)
        T = T_init

        # ===== 2) recuit simulé =====
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

            T *= alpha

        # ===== 3) descente locale stricte (type VNS niveau 1) =====
        if use_timer and (time.time() - start_time) >= time_limit:
            return bestW, bestH

        W, H, currentF = local_descent(X, W, H, LW, UW, LH, UH)

        # ===== 4) mise à jour du meilleur global =====
        if currentF < bestF:
            bestF = currentF
            bestW = W.copy()
            bestH = H.copy()

        if not use_timer:
            break

    return bestW, bestH
