import numpy as np
import sympy as sp

def find_SVD(A: np.ndarray):
    assert(np.ndim(A) == 2)
    assert(A.shape == (2,2))
    