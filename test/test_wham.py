
from rewcore import wham_fi, wham_fk, wham_normalize
import numpy as np
from numpy.testing import assert_allclose

def tower_sample(distribution):
    """draws random integers from the given distribution"""
    cdf = np.cumsum(distribution)
    rnd = np.random.rand() * cdf[-1]
    ind = (cdf > rnd)
    idx = np.where(ind == True)
    return np.min(idx)

def get_data(pi_K_i, n_samples):
    N_K_i = np.zeros(shape=pi_K_i.shape, dtype=np.intc)
    for K in range(pi_K_i.shape[0]):
        for s in range(n_samples):
            N_K_i[K, tower_sample(pi_K_i[K, :])] += 1
    return N_K_i

def run_wham(log_N_K, log_N_i, f_K, f_i, b_K_i, maxiter, ftol):
    old_f_K = f_K.copy()
    scratch_K = np.zeros(shape=f_K.shape, dtype=np.float64)
    scratch_i = np.zeros(shape=f_i.shape, dtype=np.float64)
    stop = False
    for m in range(maxiter):
        wham_fk(f_i, b_K_i, scratch_i, f_K)
        nz = (old_f_K != 0.0)
        if (nz.sum() > 0) and (np.max(np.abs((f_K[nz] - old_f_K[nz])/old_f_K[nz])) < ftol):
            stop = True
        else:
            old_f_K[:] = f_K[:]
        wham_fi(log_N_K, log_N_i, f_K, b_K_i, scratch_K, f_i)
        wham_normalize(f_i, scratch_i)
        if stop:
            break

def get_2therm_parameters(A_i):
    pi_i = np.exp(-A_i)
    pi_i /= pi_i.sum()
    f_i = -np.log(pi_i)
    b_K_i = np.array([np.zeros(shape=f_i.shape), f_i.max()-f_i], dtype=np.float64)
    pi_K_i = np.exp(-(b_K_i + f_i[np.newaxis, :]))
    f_K = 1.0/pi_K_i.sum(axis=1)
    pi_K_i = pi_K_i / f_K[:, np.newaxis]
    return f_i, b_K_i, np.log(f_K), pi_K_i

def test_wham_fk_with_zeros():
    T = 5
    M = 10
    f_i = np.zeros(shape=(M,), dtype=np.float64)
    b_K_i = np.zeros(shape=(T, M), dtype=np.float64)
    scratch = np.zeros(shape=(M,), dtype=np.float64)
    f_K = np.zeros(shape=(T,), dtype=np.float64)
    ref = -np.log(M)
    wham_fk(f_i, b_K_i, scratch, f_K)
    assert_allclose(f_K, ref, atol=1.0E-15)

def test_wham_fk_with_ascending_K():
    T = 5
    M = 10
    f_i = np.zeros(shape=(M,), dtype=np.float64)
    b_K_i = -np.log(np.array([[K]*M for K in range(1, T + 1)], dtype=np.float64)/float(M))
    scratch = np.zeros(shape=(M,), dtype=np.float64)
    f_K = np.zeros(shape=(T,), dtype=np.float64)
    ref = -np.log(np.arange(T).astype(np.float64) + 1)
    wham_fk(f_i, b_K_i, scratch, f_K)
    assert_allclose(f_K, ref, atol=1.0E-15)

def test_wham_fi_with_zeros():
    T = 5
    M = 10
    N_K_i = np.ones(shape=(T, M), dtype=np.float64)
    log_N_K = np.log(N_K_i.sum(axis=1))
    log_N_i = np.log(N_K_i.sum(axis=0))
    f_i = np.zeros(shape=(M,), dtype=np.float64)
    b_K_i = np.zeros(shape=(T, M), dtype=np.float64)
    scratch = np.zeros(shape=(T,), dtype=np.float64)
    f_K = np.zeros(shape=(T,), dtype=np.float64)
    ref = np.log(M)
    wham_fi(log_N_K, log_N_i, f_K, b_K_i, scratch, f_i)
    assert_allclose(f_i, ref, atol=1.0E-15)

def test_wham_fi_with_ascending_K():
    T = 5
    M = 10
    N_K_i = np.ones(shape=(T, M), dtype=np.float64)
    log_N_K = np.log(N_K_i.sum(axis=1))
    log_N_i = np.log(N_K_i.sum(axis=0))
    f_i = np.zeros(shape=(M,), dtype=np.float64)
    b_K_i = np.log(np.array([[i*M for i in range(1, M + 1)] for K in range(T)], dtype=np.float64))
    scratch = np.zeros(shape=(T,), dtype=np.float64)
    f_K = np.zeros(shape=(T,), dtype=np.float64)
    ref = -np.log(np.arange(M).astype(np.float64) + 1)
    wham_fi(log_N_K, log_N_i, f_K, b_K_i, scratch, f_i)
    assert_allclose(f_i, ref, atol=1.0E-15)

def test_run_wham_from_uniform_guess():
    n_samples = 10000
    maxiter = 10000
    ftol = 1.0E-15
    A_i = np.array([-2.0, 0.0, -4.0], dtype=np.float64)
    f_i, b_K_i, f_K, pi_K_i = get_2therm_parameters(A_i)
    N_K_i = get_data(pi_K_i, n_samples)
    log_N_K = np.log(N_K_i.sum(axis=1).astype(np.float64))
    log_N_i = np.log(N_K_i.sum(axis=0).astype(np.float64))
    F_K = np.zeros(shape=f_K.shape, dtype=np.float64)
    F_i = np.zeros(shape=f_i.shape, dtype=np.float64)
    run_wham(log_N_K, log_N_i, F_K, F_i, b_K_i, maxiter, ftol)
    assert_allclose(F_K, f_K, atol=0.05)
    assert_allclose(F_i, f_i, atol=0.05)
