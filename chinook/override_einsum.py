import numpy as np
import numba

__all__ = ['_einsum_ij_kjl_kil', '_einsum_ijk_ijk_k', '_einsum_ij_ijkl_ikl', '_einsum_ij_ijk_k']

@numba.njit(nogil=True)
def _einsum_ij_kjl_kil(a, b):
    ii, jj = a.shape
    kk, _, ll = b.shape
    out = np.empty((kk, ii, ll), np.complex128)
    for k in numba.prange(kk):
        for i in range(ii):
            for l in range(ll):
                out[k,i,l] = 0.
                for j in range(jj):
                    out[k,i,l] += a[i,j] * b[k,j,l]
    return out

@numba.njit(nogil=True)
def _einsum_ijk_ijk_k(a, b):
    ii, jj, kk = a.shape
    out = np.zeros(kk, np.complex128)
    for i in range(ii):
        for j in range(jj):
            for k in range(kk):
                out[k] += a[i,j,k] * b[i,j,k]
    return out

@numba.njit(nogil=True)
def _einsum_i_ij_ij(a, b):
    return np.multiply(np.reshape(a,(-1, 1)), b)

@numba.njit(nogil=True)
def _einsum_i_i_i(a, b):
    return np.multiply(a, b)

def _einsum_ij_j_i(a, b):
    return a @ b
def _einsum_ijk_k_ij(a, b):
    return a @ b

@numba.njit(nogil=True)
def _einsum_ij_ij_i(a, b):
    return np.multiply(a, b).sum(axis=1)

def _einsum_ij_kj_ik(a, b):
    return a @ b.T

@numba.njit(nogil=True)
def _einsum_ij_ijkl_ikl(a, b):
    ii, jj, kk, ll = b.shape
    val = np.empty((ii, kk, ll), np.complex128)
    for i in range(ii):
        for k in range(kk):
            for l in range(ll):
                val[i,k,l] = 0
                for j in range(jj):
                    val[i,k,l] += a[i,j] * b[i,j,k,l]
    return val

@numba.njit(nogil=True)
def _einsum_ij_ijk_k(a, b):
    ii, jj, kk = b.shape
    val = np.zeros(kk, np.complex128)
    for i in range(ii):
        for j in range(jj):
            for k in range(kk):
                val[k] += a[i,j] * b[i,j,k]
    return val
