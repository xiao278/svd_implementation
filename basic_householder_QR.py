import numpy as np


def calculate_householder(mat: np.ndarray, iter: int):
    # iter 1 output is mxm
    # iter 2 output is (m-1)x(m-1)
    n = mat.shape[0]
    x = mat[iter:,iter]
    z = np.zeros(x.shape)
    z[0] = 1
    v = x + np.linalg.norm(x, 2) * z
    v = v.reshape(-1, 1)
    I = np.identity(n - iter)
    H = I - 2 * np.matmul(v, v.transpose()) / (np.linalg.norm(v) ** 2)
    H = np.pad(H, pad_width=((iter, 0), (iter, 0)), mode='constant', constant_values=0)
    for i in range(iter):
        H[i,i] = 1
    return H

def calculate_QR(A: np.ndarray):
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    Q = np.identity(n)
    R = A
    for iter in range(0, n-1):
        H = calculate_householder(R, iter)
        Q = np.matmul(Q, H)
        R = np.matmul(H, R)
    return Q, R



if __name__ == "__main__":
    matrix2 = np.array([
        [1,2,3,4],
        [5,6,70,8],
        [9,101,11,12],
        [13,-5,15,16]    
    ])
    # H1 = calculate_householder(matrix2, 0)
    # R1 = np.matmul(H1, matrix2)
    # H2 = calculate_householder(R1, 1)
    # R2 = np.matmul(H2, R1)
    # H3 = calculate_householder(R2, 2)
    # R3 = np.matmul(H3, R2)
    Q, R = calculate_QR(matrix2)
    print(np.matmul(Q, R))

    