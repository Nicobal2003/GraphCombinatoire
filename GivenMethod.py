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


