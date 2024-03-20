import numpy as np
import scipy
import torch
from numba import njit

def euclidean_distances(X, Y, squared=False):
    a2 = np.einsum('ij,ij->i', X, X)
    b2 = np.einsum('ij,ij->i', Y, Y)

    c = -2 * np.dot(X, Y.T)
    c += a2[:, None]
    c += b2[None, :]

    c = np.maximum(c, 0)

    if not squared:
        c = np.sqrt(c)

    if X is Y:
        c = c * (1 - np.eye(X.shape[0], dtype=c.dtype))

    return c

def pcc_distances(v1, v2):
    if v1.shape[1] != v2.shape[1]:
        raise ValueError("The two matrices v1 and v2 must have equal dimensions; two slice data must have the same genes")

    n = v1.shape[1]
    sums = np.multiply.outer(v1.sum(1), v2.sum(1))
    stds = np.multiply.outer(v1.std(1), v2.std(1))
    correlation = (v1.dot(v2.T) - sums / n) / stds / n
    distances=1-correlation
    return distances

def kl_divergence_backend(X, Y):
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    X = X/np.sum(X,axis=1, keepdims=True)
    Y = Y/np.sum(Y,axis=1, keepdims=True)
    log_X = np.log(X)
    log_Y = np.log(Y)
    X_log_X = np.einsum('ij,ij->i',X,log_X)
    X_log_X = np.reshape(X_log_X,(1,X_log_X.shape[0]))
    D = X_log_X.T - np.dot(X,log_Y.T)
    return D

## Covert a sparse matrix into a dense np array
to_dense_array = lambda X: X.toarray() if isinstance(X,scipy.sparse.csr.spmatrix) else np.array(X)

## Returns the data matrix or representation
extract_data_matrix = lambda adata,rep: adata.X if rep is None else adata.obsm[rep]

@njit
def calculate_squared_differences(matrix_A, matrix_B):
    n = matrix_A.shape[0]
    m = matrix_B.shape[0]
    matrix_A = matrix_A.reshape(n, 1, n, 1)
    matrix_B = matrix_B.reshape(1, m, 1, m)

    result = (matrix_A - matrix_B) ** 2
    print("over")
    return result


# def calculate_squared_differences(matrix_A, matrix_B, block_size=50):
#     n = matrix_A.shape[0]
#     m = matrix_B.shape[0]
#     result = np.zeros((n, n, m, m))
    
#     for i in range(0, n, block_size):
#         i_end = min(i + block_size, n)
#         for k in range(0, n, block_size):
#             k_end = min(k + block_size, n)
#             for j in range(0, m, block_size):
#                 j_end = min(j + block_size, m)
#                 for l in range(0, m, block_size):
#                     l_end = min(l + block_size, m)
#                     result[i:i_end, k:k_end, j:j_end, l:l_end] = (
#                         (matrix_A[i:i_end, k:k_end].reshape(-1, 1, 1) - matrix_B[j:j_end, l:l_end]) ** 2
#                     )
#     print("over")
#     return result

def init_matrix(C1, C2, p, q, loss_fun='square_loss'):
    p=p/len(p)
    if loss_fun == 'square_loss':
        def f1(a):
            return (a**2)

        def f2(b):
            return (b**2)

        def h1(a):
            return a

        def h2(b):
            return 2 * b

    constC1 = np.dot(
        np.dot(f1(C1), np.reshape(p, (-1, 1))),
        np.ones((1, len(q)), dtype=q.dtype)
    )

    constC2 = np.dot(
        np.ones((len(p), 1), dtype=p.dtype),
        np.dot(np.reshape(q, (1, -1)), f2(C2).T)
    )

    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2