# cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.math cimport INFINITY, NAN, sqrt
from libc.stdlib cimport free, malloc
from cython.parallel cimport prange
cimport numpy as cnp
import numpy as np


cdef extern from "_exhaustive_sort.h" namespace "otlingam":
    void sort_values(double *values, size_t size) noexcept nogil


cdef inline int _popcount(int mask) noexcept nogil:
    cdef int result = 0
    while mask:
        result += mask & 1
        mask >>= 1
    return result


cdef inline int _build_system(
    const double *cov, int d, int target, int mask,
    double *A, int *parents,
) noexcept nogil:
    cdef int i, j, k = 0, kk
    for i in range(d):
        if (mask >> i) & 1:
            kk = k * (k + 1) // 2
            for j in range(k):
                A[kk + j] = cov[i * d + parents[j]]

            A[kk + k] = cov[i * d + i]
            parents[k] = i
            k += 1

    kk = k * (k + 1) // 2
    for j in range(k):
        A[kk + j] = cov[target * d + parents[j]]

    A[kk + k] = cov[target * d + target]
    return k


cdef inline double _cholesky(double *A, int k) noexcept nogil:
    cdef int i, j, l, ii, ij
    for i in range(k + 1):
        ii = i * (i + 1) // 2
        for j in range(i):
            ij = ii + j
            for l in range(j):
                A[ij] -= A[ii + l] * A[j * (j + 1) // 2 + l]

            A[ij] /= A[j * (j + 1) // 2 + j]

        ij = ii + i
        for l in range(i):
            A[ij] -= A[ii + l] * A[ii + l]

        if A[ij] <= 0.0:
            return NAN
        A[ij] = sqrt(A[ij])

    return A[k * (k + 3) // 2] ** 2


cdef inline void _solve_coefficients(
    const double *A, int k, double *coef,
) noexcept nogil:
    cdef int i, j
    for i in range(k):
        coef[i] = A[k * (k + 1) // 2 + i]

    for i in range(k - 1, -1, -1):
        for j in range(i + 1, k):
            coef[i] -= A[j * (j + 1) // 2 + i] * coef[j]

        coef[i] /= A[i * (i + 1) // 2 + i]


cdef inline void _compute_residuals(
    const double *X, int n, int d, int target, int mask,
    const double *coef, double *residuals,
) noexcept nogil:
    cdef int i, j, parent
    cdef double value
    for i in range(n):
        value = X[i * d + target]
        parent = 0
        for j in range(d):
            if (mask >> j) & 1:
                value -= coef[parent] * X[i * d + j]
                parent += 1

        residuals[i] = value


cdef inline double _score(
    const double *X, const double *cov, const double *quantiles,
    int n, int d, int target, int mask,
) noexcept nogil:
    cdef double A[528]
    cdef double coef[31]
    cdef int parents[31]
    cdef double *residuals = <double *>malloc(n * sizeof(double))
    cdef int k, i
    cdef double rss, scale, delta, result = 0.0

    if residuals == NULL:
        return NAN
    k = _build_system(cov, d, target, mask, A, parents)
    rss = _cholesky(A, k)
    if rss != rss:
        free(residuals)
        return NAN
    _solve_coefficients(A, k, coef)
    _compute_residuals(X, n, d, target, mask, coef, residuals)
    scale = sqrt(rss / n)
    for i in range(n):
        residuals[i] /= scale

    sort_values(residuals, n)
    for i in range(n):
        delta = residuals[i] - quantiles[i]
        result += delta * delta

    free(residuals)
    return result / n


cdef inline void _process_mask(
    const double *X, const double *cov, const double *quantiles,
    float *H, cnp.int32_t *sink_out, int mask, int n, int d,
) noexcept nogil:
    cdef int bits = mask
    cdef int sink = 0
    cdef int previous
    cdef float best_score = -INFINITY
    cdef float candidate
    cdef int best_sink = -1
    while bits:
        if bits & 1:
            previous = mask ^ (1 << sink)
            candidate = H[previous] + <float>_score(
                X, cov, quantiles, n, d, sink, previous)
            if candidate > best_score:
                best_score = candidate
                best_sink = sink
        bits >>= 1
        sink += 1

    H[mask] = best_score
    sink_out[mask] = best_sink


cpdef tuple _sink_dp(
    const double[:, ::1] X,
    const double[:, ::1] cov_matrix,
    const double[::1] quantiles,
    int d,
):
    cdef int n_states = 1 << d
    cdef int mask, size, idx, start, end
    cdef int n = X.shape[0]
    cdef cnp.ndarray[cnp.int32_t, ndim=1] masks = np.empty(n_states - 1, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] offsets = np.zeros(d + 2, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] counts = np.zeros(d + 1, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] positions
    cdef cnp.ndarray[cnp.float32_t, ndim=1] scores = np.zeros(n_states, dtype=np.float32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] sinks = np.full(n_states, -1, dtype=np.int32)
    cdef float *H = &scores[0]
    cdef cnp.int32_t *sink_out = &sinks[0]
    cdef const double *x_ptr = &X[0, 0]
    cdef const double *cov_ptr = &cov_matrix[0, 0]
    cdef const double *q_ptr = &quantiles[0]

    for mask in range(1, n_states):
        counts[_popcount(mask)] += 1

    for size in range(1, d + 1):
        offsets[size + 1] = offsets[size] + counts[size]

    positions = offsets.copy()
    for mask in range(1, n_states):
        size = _popcount(mask)
        masks[positions[size]] = mask
        positions[size] += 1

    for size in range(1, d + 1):
        start = offsets[size]
        end = offsets[size + 1]
        for idx in prange(start, end, nogil=True, schedule='static'):
            _process_mask(x_ptr, cov_ptr, q_ptr, H, sink_out, masks[idx], n, d)

    return sinks, float(H[n_states - 1])
