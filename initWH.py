import numpy as np

def initial_solution(m, n, r, LW, UW, LH, UH):
    # W : m x r, entiers dans [LW, UW]
    W = np.random.randint(LW, UW + 1, size=(m, r))

    # H : r x n, entiers dans [LH, UH]
    H = np.random.randint(LH, UH + 1, size=(r, n))

    return W, H
