import numpy as np
import time
from typing import Optional, Tuple


# ---------- Objective + Feasibility ----------

def fobj(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    return float(np.linalg.norm(X - W @ H, ord="fro") ** 2)


def solutionIsFeasible(
    W: np.ndarray, H: np.ndarray, r: int, LW: int, UW: int, LH: int, UH: int
) -> bool:
    if W.shape[1] != r or H.shape[0] != r:
        return False
    Wi = np.issubdtype(W.dtype, np.integer) or (
        np.issubdtype(W.dtype, np.floating)
        and np.all(np.isfinite(W)) and np.all(W == np.floor(W))
    )
    Hi = np.issubdtype(H.dtype, np.integer) or (
        np.issubdtype(H.dtype, np.floating)
        and np.all(np.isfinite(H)) and np.all(H == np.floor(H))
    )
    if not (Wi and Hi):
        return False
    return np.all((W >= LW) & (W <= UW)) and np.all((H >= LH) & (H <= UH))


# ---------- Greedy (biclique rank-1) on integer residual ----------

def _grow_biclique(R: np.ndarray, i0: int, j0: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Construit un biclique u vᵀ inclus dans {R >= 1} à partir du pivot (i0,j0).
    Gain = |U| * |V|. Ne couvre que des cellules avec R>=1.
    """
    m, n = R.shape
    if R[i0, j0] < 1:
        return np.zeros(m, dtype=np.int8), np.zeros(n, dtype=np.int8), 0

    u = np.zeros(m, dtype=np.int8)
    v = (R[i0] >= 1).astype(np.int8)
    u[i0] = 1
    if v.sum() == 0:
        v[j0] = 1

    changed = True
    while changed:
        changed = False
        if v.sum() > 0:
            need = int(v.sum())
            mask_rows = (R[:, v == 1] >= 1).sum(axis=1) == need
            new_u = u.copy()
            new_u[mask_rows] = 1
            if new_u.sum() > u.sum():
                u = new_u
                changed = True
        if u.sum() > 0:
            need = int(u.sum())
            mask_cols = (R[u == 1, :] >= 1).sum(axis=0) == need
            new_v = np.zeros_like(v)
            new_v[mask_cols] = 1
            if new_v.sum() > v.sum():
                v = new_v
                changed = True

    gain = int(u.sum() * v.sum())
    if gain == 0:
        return u * 0, v * 0, 0
    if np.any(R[np.ix_(u == 1, v == 1)] < 1):
        return u * 0, v * 0, 0
    return u, v, gain


def greedy_biclique_build(
    X: np.ndarray,
    r: int,
    *,
    max_pivots_per_factor: int = 8,
    min_gain: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    """
    Construit jusqu'à r facteurs binaires (u_t v_tᵀ) sur résidu entier R.
    À chaque facteur accepté: R[U,V] -= 1. Exact=True si R==0 (donc f=0).
    Retourne (W,H,R,f,exact).
    """
    if rng is None:
        rng = np.random.default_rng()

    m, n = X.shape
    W = np.zeros((m, r), dtype=np.int64)
    H = np.zeros((r, n), dtype=np.int64)
    R = X.astype(np.int64).copy()
    used = 0

    for t in range(r):
        pos = np.argwhere(R >= 1)
        if pos.size == 0:
            break

        idx = rng.choice(len(pos), size=min(max_pivots_per_factor, len(pos)), replace=False)
        best_u = best_v = None
        best_gain = 0
        for p in idx:
            i0, j0 = map(int, pos[p])
            u, v, gain = _grow_biclique(R, i0, j0)
            if gain > best_gain:
                best_gain, best_u, best_v = gain, u, v

        if best_gain >= min_gain:
            W[:, t] = best_u
            H[t, :] = best_v
            R[np.ix_(best_u == 1, best_v == 1)] -= 1
            used += 1
        else:
            break

        if np.all(R == 0):
            return W, H, R, 0, True

    f = int(np.sum(R ** 2))
    return W, H, R, f, np.all(R == 0)


# ---------- Shake (±1) ----------

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
    return W2, H2


# ---------- Local descent + light Tabu ----------

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
    if rng is None:
        rng = np.random.default_rng()
    if start_time is None:
        start_time = time.time()

    m, rr = W.shape
    r2, n = H.shape
    assert r == rr == r2

    diff = X - W @ H
    currentF = float(np.sum(diff * diff))
    tabu_W, tabu_H = {}, {}
    accept_count = 0

    for _ in range(max_ls_iters):
        if time_limit and (time.time() - start_time) >= time_limit:
            break
        improved = False

        for _ in range(moves_per_iter):
            if time_limit and (time.time() - start_time) >= time_limit:
                break

            if rng.random() < 0.5:
                i = rng.integers(0, m)
                j = rng.integers(0, r)
                old = W[i, j]
                new = int(np.clip(old + rng.choice([-1, 1]), LW, UW))
                if new == old:
                    continue
                is_tabu = (i, j) in tabu_W and tabu_W[(i, j)] > accept_count
                d = new - old
                new_row = diff[i, :] - d * H[j, :]
                newF = currentF - float(np.sum(diff[i, :] ** 2)) + float(np.sum(new_row ** 2))
                if is_tabu and not (newF < aspiration_bestF):
                    continue
                if newF < currentF:
                    W[i, j] = new
                    diff[i, :] = new_row
                    currentF = fobj(X, W, H)  # why: verrouiller objectif
                    if not solutionIsFeasible(W, H, r, LW, UW, LH, UH):
                        W[i, j] = old
                        diff[i, :] = diff[i, :] + d * H[j, :]
                        currentF = fobj(X, W, H)
                        continue
                    accept_count += 1
                    tabu_W[(i, j)] = accept_count + tabu_tenure
                    improved = True
                    break
            else:
                j = rng.integers(0, r)
                k = rng.integers(0, n)
                old = H[j, k]
                new = int(np.clip(old + rng.choice([-1, 1]), LH, UH))
                if new == old:
                    continue
                is_tabu = (j, k) in tabu_H and tabu_H[(j, k)] > accept_count
                d = new - old
                new_col = diff[:, k] - d * W[:, j]
                newF = currentF - float(np.sum(diff[:, k] ** 2)) + float(np.sum(new_col ** 2))
                if is_tabu and not (newF < aspiration_bestF):
                    continue
                if newF < currentF:
                    H[j, k] = new
                    diff[:, k] = new_col
                    currentF = fobj(X, W, H)
                    if not solutionIsFeasible(W, H, r, LW, UW, LH, UH):
                        H[j, k] = old
                        diff[:, k] = diff[:, k] + d * W[:, j]
                        currentF = fobj(X, W, H)
                        continue
                    accept_count += 1
                    tabu_H[(j, k)] = accept_count + tabu_tenure
                    improved = True
                    break

        if not improved:
            break

    return W, H, currentF


# ---------- VNS + (optional) Greedy front-end ----------

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
    max_no_improv: int = 20,
    time_limit: Optional[float] = None,
    seed: Optional[int] = None,
    use_greedy: bool = True,
    greedy_pivots: int = 8,
    greedy_min_gain: int = 2,
) -> Tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    m, n = X.shape
    start_time = time.time()

    if use_greedy:
        Wg, Hg, R, fR, exact = greedy_biclique_build(
            X, r,
            max_pivots_per_factor=greedy_pivots,
            min_gain=greedy_min_gain,
            rng=rng,
        )
        if exact and np.array_equal(X, (Wg @ Hg)):
            return Wg, Hg, 0.0
        used = (Hg.sum(axis=1) > 0)
        W0 = rng.integers(LW, UW + 1, size=(m, r))
        H0 = rng.integers(LH, UH + 1, size=(r, n))
        W0[:, used] = Wg[:, used]
        H0[used, :] = Hg[used, :]
    else:
        W0 = rng.integers(LW, UW + 1, size=(m, r))
        H0 = rng.integers(LH, UH + 1, size=(r, n))

    assert solutionIsFeasible(W0, H0, r, LW, UW, LH, UH)

    W, H, F = _local_descent_tabou(
        X, W0, H0, LW, UW, LH, UH,
        r=r,
        max_ls_iters=max_ls_iters,
        moves_per_iter=moves_per_iter,
        tabu_tenure=tabu_tenure,
        aspiration_bestF=float("inf"),
        rng=rng,
        time_limit=time_limit,
        start_time=start_time,
    )

    bestW, bestH, bestF = W.copy(), H.copy(), F
    no_improv = 0

    while True:
        if time_limit and (time.time() - start_time) >= time_limit:
            break
        improved_outer = False
        k = 1
        while k <= k_max:
            if time_limit and (time.time() - start_time) >= time_limit:
                break
            W_sh, H_sh = _shake(bestW, bestH, LW, UW, LH, UH, k, rng)
            W_loc, H_loc, F_loc = _local_descent_tabou(
                X, W_sh, H_sh, LW, UW, LH, UH,
                r=r,
                max_ls_iters=max_ls_iters,
                moves_per_iter=moves_per_iter,
                tabu_tenure=tabu_tenure,
                aspiration_bestF=bestF,
                rng=rng,
                time_limit=time_limit,
                start_time=start_time,
            )
            if F_loc < bestF:
                bestW, bestH, bestF = W_loc.copy(), H_loc.copy(), F_loc
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
    return bestW, bestH, float(fobj(X, bestW, bestH))