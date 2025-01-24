import numpy as np
import sympy as sp

def find_SVD(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    AT = np.transpose(A)
    A_AT = np.matmul(A, AT)
    AT_A = np.matmul(AT, A)
    eigvals, eigvecs = np.linalg.eigh(AT_A)
    sorted_eigvals_indexes = sorted(range(len(eigvals)), key=lambda i: eigvals[i], reverse=True)
    sorted_eigvals = eigvals[sorted_eigvals_indexes]
    sorted_eigvecs = eigvecs[:, sorted_eigvals_indexes]
    # all eigenvalues of symmetric matrices like AT_A are positive so any negative value is an error that can be corrected with an abs
    sigmas = np.sqrt(np.abs(sorted_eigvals))
    V = sorted_eigvecs
    U = compute_U_matrix(A, V, sigmas)
    Sigma_Matrix = np.zeros(A.shape)
    np.fill_diagonal(Sigma_Matrix, sigmas)
    
    print(np.matmul(np.matmul(U, Sigma_Matrix), V.transpose()))

def compute_U_matrix(A: np.ndarray, V: np.ndarray, sigmas: np.ndarray):
    assert(np.ndim(A) == 2)
    m = A.shape[0]
    n = A.shape[1]
    r = min(m, n)
    vecs = []
    for i in range(r):
        vecs.append(np.matmul(A, V[:, i]) / sigmas[i])
    U = np.column_stack(vecs)
    if (m > r):
        # Find the null space basis of A^T using orthogonalization
        identity = np.eye(m)
        null_space_basis = identity - U @ U.T  # Project orthogonal to U_partial
        Q_additional, _ = np.linalg.qr(null_space_basis)  # Orthonormalize
        
        # Take only the additional columns needed
        Q_additional = Q_additional[:, r:]
        
        # Combine U_partial with the additional orthonormal vectors
        U = np.hstack((U, Q_additional))
    return U
    
def vec_length(vector: np.ndarray):
    assert(vector.ndim == 1)
    return np.sqrt(np.sum(np.square(vector)))

def compute_eigenvalues(mat: np.ndarray):
    assert(np.ndim(mat) == 2)
    assert(mat.shape[0] == mat.shape[1])
    results = np.linalg.eigvals(mat)
    return results
    
def find_determinant(mat: np.ndarray):
    return np.linalg.det(mat)

def compute_matrix(mat: np.ndarray, sigmas_squared: list[float]):
    assert(mat.shape[0] == mat.shape[1])
    vectors = []
    for sigma_squared in sigmas_squared:
        assert (sigma_squared != 0)
        size = mat.shape[0]
        new_mat = np.copy(mat)
        for i in range (size):
            new_mat[i,i] -= sigma_squared
        unit_vector = np.zeros(new_mat.shape[1])
        unit_vector[0] = 1
        M = np.vstack([new_mat, unit_vector])  # Add normalization row
        b = np.zeros(new_mat.shape[0] + 1)            # Adjust b vector
        b[-1] = 1  # Normalization value
        x, residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
        vectors.append()
    print(vectors)
    return None

if __name__ == "__main__":
    matrix = np.array([
        [3,2,2],
        [2,3,-2]
    ])
    find_SVD(matrix)
    matrix1 = np.array([
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [13,14,15,16],
        [17,18,19,20]
    ])
    find_SVD(matrix1)
    matrix2 = np.array([
        [2,1],
        [1,-1]    
    ])
    find_SVD(matrix2)
    
    
