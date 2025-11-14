import numpy as np
import random
import os

def read_input(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    m, n, r, LW, UW, LH, UH = map(int, lines[0].split())
    X = np.array([list(map(int, l.split())) for l in lines[1:]])
    return X, m, n, r, LW, UW, LH, UH