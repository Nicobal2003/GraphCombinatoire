import numpy as np


def fobj(X, W, H):
    f = np.linalg.norm(X - W@H, 'fro')**2
    return f