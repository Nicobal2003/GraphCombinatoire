# file: VNS_tabou.py
import numpy as np
import time
from typing import Optional, Tuple


# ---------- objectif + faisabilité ----------

def fobj(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    return float(np.linalg.norm(X - W @ H, ord="fro") ** 2)


def solutionIsFeasible(
    W: np.ndarray, H: np.ndarray, r: int, LW: int, UW: int, LH: int, UH: int
) -> bool:
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


# ---------- shaking inchangé (±1, borné) ----------

def _shake(
    W: np.ndarray,
    H: np.ndarray,
    LW: int,
    UW: int,
    LH: int,
    UH: int,
    k: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    m, r = W.shape
    r2, n = H.shape
    assert r == r2
    W2 = W.copy()
    H2 = H.copy()
    for _ in range(k):
        if rng.random() < 0.5:
            i = rng.integers(0, m)
            j = rng.integers(0, r)
            W2[i, j] = int(np.clip(W2[i, j] + rng.choice([-1, 1]), LW, UW))
        else:
            j = rng.integers(0, r)
            k_idx = rng.integers(0, n)
            H2[j, k_idx] = int(np.clip(H2[j, k_idx] + rng.choice([-1, 1]), LH, UH))
    assert solutionIsFeasible(W2, H2, r, LW, UW, LH, UH)
    return W2, H2


# ---------- descente locale avec tabou léger ----------

def _local_descent_tabou(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    LW: int,
    UW: int,
    LH: int,
    UH: int,
    *,
    r: int,
    max_ls_iters: int = 50,
    moves_per_iter: int = 2000,
    tabu_tenure: int = 200,
    aspiration_bestF: float = float("inf"),
    rng: Optional[np.random.Generator] = None,
    time_limit: Optional[float] = None,
    start_time: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Descente locale "première amélioration" + tabou léger (par position).
    Pourquoi: éviter les va-et-vient récents; aspiration si on bat le meilleur global.
    """
    if rng is None:
        rng = np.random.default_rng()
    if start_time is None:
        start_time = time.time()

    m, rr = W.shape
    r2, n = H.shape
    assert r == rr == r2

    # état courant
    WH = W @ H
    diff = X - WH
    currentF = float(np.sum(diff * diff))

    # mémoire tabou (expiration en nombre d'acceptations)
    tabu_W = {}  # key: (i,j) -> expire_at
    tabu_H = {}  # key: (j,k) -> expire_at
    accept_count = 0  # horloge pour la durée tabou

    for _ in range(max_ls_iters):
        improved = False
        if (time_limit is not None) and ((time.time() - start_time) >= time_limit):
            break

        for _ in range(moves_per_iter):
            if (time_limit is not None) and ((time.time() - start_time) >= time_limit):
                break

            if rng.random() < 0.5:
                # move sur W[i,j]
                i = rng.integers(0, m)
                j = rng.integers(0, r)
                old_val = W[i, j]
                new_val = int(np.clip(old_val + rng.choice([-1, 1]), LW, UW))
                d = new_val - old_val
                if d == 0:
                    continue

                # tabou (position W[i,j])
                is_tabu = (i, j) in tabu_W and tabu_W[(i, j)] > accept_count

                # delta local
                new_diff_row = diff[i, :] - d * H[j, :]
                old_row_sq = float(np.sum(diff[i, :] ** 2))
                new_row_sq = float(np.sum(new_diff_row ** 2))
                newF = currentF - old_row_sq + new_row_sq

                # aspiration: autoriser tabou si on bat le meilleur connu
                if is_tabu and not (newF < aspiration_bestF):
                    continue

                if newF < currentF:
                    # acceptation
                    W[i, j] = new_val
                    diff[i, :] = new_diff_row
                    # verrouiller et fiabiliser l'objectif (pour éviter la dérive)
                    currentF = fobj(X, W, H)
                    if not solutionIsFeasible(W, H, r, LW, UW, LH, UH):
                        # sécurité: revert si jamais
                        W[i, j] = old_val
                        diff[i, :] = diff[i, :] + d * H[j, :]
                        currentF = fobj(X, W, H)
                        continue
                    # mise tabou
                    accept_count += 1
                    tabu_W[(i, j)] = accept_count + tabu_tenure
                    improved = True
                    break

            else:
                # move sur H[j,k]
                j = rng.integers(0, r)
                k = rng.integers(0, n)
                old_val = H[j, k]
                new_val = int(np.clip(old_val + rng.choice([-1, 1]), LH, UH))
                d = new_val - old_val
                if d == 0:
                    continue

                # tabou (position H[j,k])
                is_tabu = (j, k) in tabu_H and tabu_H[(j, k)] > accept_count

                # delta local
                new_diff_col = diff[:, k] - d * W[:, j]
                old_col_sq = float(np.sum(diff[:, k] ** 2))
                new_col_sq = float(np.sum(new_diff_col ** 2))
                newF = currentF - old_col_sq + new_col_sq

                # aspiration
                if is_tabu and not (newF < aspiration_bestF):
                    continue

                if newF < currentF:
                    H[j, k] = new_val
                    diff[:, k] = new_diff_col
                    currentF = fobj(X, W, H)
                    if not solutionIsFeasible(W, H, r, LW, UW, LH, UH):
                        H[j, k] = old_val
                        diff[:, k] = diff[:, k] + d * W[:, j]
                        currentF = fobj(X, W, H)
                        continue
                    accept_count += 1
                    tabu_H[(j, k)] = accept_count + tabu_tenure
                    improved = True
                    break

        if not improved:
            break

    # final
    currentF = fobj(X, W, H)
    assert solutionIsFeasible(W, H, r, LW, UW, LH, UH)
    return W, H, currentF


# ---------- VNS + Tabou léger ----------

def vns_factorization(
    X: np.ndarray,
    r: int,
    LW: int,
    UW: int,
    LH: int,
    UH: int,
    *,
    k_max: int = 5,
    max_ls_iters: int = 50,
    moves_per_iter: int = 2000,
    tabu_tenure: int = 200,
    reset_tabu_on_shake: bool = True,  # pourquoi: repartir propre à chaque voisinage
    max_no_improv: int = 20,
    time_limit: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    m, n = X.shape
    start_time = time.time()

    # init faisable
    W = rng.integers(LW, UW + 1, size=(m, r))
    H = rng.integers(LH, UH + 1, size=(r, n))
    assert solutionIsFeasible(W, H, r, LW, UW, LH, UH)

    # première descente (aspire au +∞ → aucun blocage)
    W, H, F = _local_descent_tabou(
        X, W, H, LW, UW, LH, UH,
        r=r,
        max_ls_iters=max_ls_iters,
        moves_per_iter=moves_per_iter,
        tabu_tenure=tabu_tenure,
        aspiration_bestF=float("inf"),
        rng=rng,
        time_limit=time_limit,
        start_time=start_time,
    )

    bestW = W.copy()
    bestH = H.copy()
    bestF = F
    no_improv = 0

    while True:
        if (time_limit is not None) and ((time.time() - start_time) >= time_limit):
            break

        improved_outer = False
        k = 1

        while k <= k_max:
            if (time_limit is not None) and ((time.time() - start_time) >= time_limit):
                break

            W_sh, H_sh = _shake(bestW, bestH, LW, UW, LH, UH, k, rng)

            # à chaque voisinage, on repart propre ou on conserve ? (par défaut: reset)
            W_loc, H_loc, F_loc = _local_descent_tabou(
                X, W_sh, H_sh, LW, UW, LH, UH,
                r=r,
                max_ls_iters=max_ls_iters,
                moves_per_iter=moves_per_iter,
                tabu_tenure=tabu_tenure,
                aspiration_bestF=bestF,  # aspiration si on bat le global
                rng=rng,
                time_limit=time_limit,
                start_time=start_time,
            )

            if F_loc < bestF:
                bestF = F_loc
                bestW = W_loc.copy()
                bestH = H_loc.copy()
                improved_outer = True
                k = 1
            else:
                k += 1

        if not improved_outer:
            no_improv += 1
            if (max_no_improv is not None) and (no_improv >= max_no_improv):
                break
        else:
            no_improv = 0

    assert solutionIsFeasible(bestW, bestH, r, LW, UW, LH, UH)
    bestF = fobj(X, bestW, bestH)
    return bestW, bestH, bestF
