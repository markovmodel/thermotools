
from rewcore import log_nu_K_i_setter, log_nu_K_i_equation, f_i_equation, p_K_ij_equation, f_K_equation
import numpy as np
from numpy.testing import assert_allclose

def run_dtram(C_K_ij, b_K_i, log_nu_K_i, f_K, f_i, maxiter, ftol):
    log_nu_K_i_setter(log_nu_K_i, C_K_ij)
    scratch_K_i = np.zeros(shape=b_K_i.shape, dtype=np.float64)
    scratch_i = np.zeros(shape=f_i.shape, dtype=np.float64)
    old_f_K = f_K.copy()
    for m in xrange(maxiter):
        tmp_log_nu_K_i = np.copy(log_nu_K_i)
        log_nu_K_i_equation(tmp_log_nu_K_i, b_K_i, f_i, C_K_ij, scratch_i, log_nu_K_i)
        tmp_f_i = np.copy(f_i)
        f_i_equation(log_nu_K_i, b_K_i, tmp_f_i, C_K_ij, scratch_K_i, scratch_i, f_i)
        f_K_equation(b_K_i, f_i, scratch_i, f_K)
        nz = (old_f_K != 0.0)
        if (nz.sum() > 0) and (np.max(np.abs((f_K[nz] - old_f_K[nz])/old_f_K[nz])) < ftol):
            break
        else:
            old_f_K[:] = f_K[:]

def test_dtram_with_toy_model():
    C_K_ij = np.array([
        [[2358, 29, 0], [29, 0, 32], [0, 32, 197518]],
        [[16818, 16763, 0], [16763, 0, 16510], [0, 16510, 16635]]], dtype=np.intc)
    b_K_i = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 8.0]], dtype=np.float64)
    f_i = np.zeros(shape=(b_K_i.shape[1],), dtype=np.float64)
    f_K = np.zeros(shape=(b_K_i.shape[0],), dtype=np.float64)
    log_nu_K_i = np.zeros(shape=b_K_i.shape, dtype=np.float64)
    run_dtram(C_K_ij, b_K_i, log_nu_K_i, f_K, f_i, 10000, 1.0E-15)
    pi = np.array([1.82026887e-02, 3.30458960e-04, 9.81466852e-01], dtype=np.float64)
    assert_allclose(np.exp(-f_i), pi, atol=1.0E-8)
    T = np.array([
        [9.90504397e-01, 9.49560284e-03, 0.0],
        [5.23046803e-01, 0.0, 4.76953197e-01],
        [0.0, 1.60589690e-04, 9.99839410e-01]], dtype=np.float64)
    scratch_i = np.zeros(shape=f_i.shape, dtype=np.float64)
    p_K_ij = np.zeros(shape=C_K_ij.shape, dtype=np.float64)
    p_K_ij_equation(log_nu_K_i, b_K_i, f_i, C_K_ij, scratch_i, p_K_ij)
    assert_allclose(p_K_ij[0, :, :], T, atol=1.0E-8)