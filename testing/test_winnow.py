from ..winnow_modw1 import Winnow
import pytest
import numpy as np

winnow = Winnow((60,60),0,"99flip0.95_noisy1.0")

def test_psi_rows():
    #Test PD
    psi = winnow.psi_rows
    eigvals = np.linalg.eigvals(psi)
    assert np.all(eigvals>0)
    assert np.allclose(psi,psi.T,atol = 1e-8)

def test_psi_cols():
    #Test PD
    psi = winnow.psi_rows
    eigvals = np.linalg.eigvals(psi)
    assert np.all(eigvals>0)
    assert np.allclose(psi,psi.T,atol = 1e-8)

def test_X():
    winnow._one_step()
    X = winnow.X
    eigvals = np.linalg.eigvals(X)
    print(eigvals)
    assert np.sum(eigvals)<1
    assert np.all(eigvals>0)
    assert np.allclose(X,X.T,atol = 1e-8)

def test_sqrt():
    X = np.eye(4,4)
    out = winnow._sqrt(X)
    assert np.allclose(np.diag(out),0.5)

def test_sqrt():
    X = np.eye(4,4)
    out = winnow._exponentiate(X)
    assert np.allclose(np.diag(out),np.e)
