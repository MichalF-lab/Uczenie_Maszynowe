#!/usr/bin/env python3

import numpy as np
import pytest
from scipy.stats import alpha
from sklearn.linear_model import Ridge

class RidgeRegr:
    def __init__(self, alpha = 0.0):
        self.alpha = alpha

    def fit(self, X, Y):
        # wejscie:
        #  X = np.array, shape = (n, m)
        #  Y = np.array, shape = (n)
        # Znajduje theta (w przyblizeniu) minimalizujace kwadratowa funkcje kosztu L uzywajac metody iteracyjnej.
        n, m = X.shape
        for _ in range(1000):
            X_extended = np.c_[np.ones((n, 1)), X]
            self.theta = np.linalg.inv(X_extended.T @ X_extended + (self.alpha * np.ones(n))) @ X_extended.T @ Y
        return self
    
    def predict(self, X):
        # wejscie
        #  X = np.array, shape = (k, m)
        # zwraca
        k, m = X.shape
        X_extended = np.c_[np.ones((k, 1)), X]
        return X_extended @ self.theta


def test_RidgeRegressionInOneDim():
    X = np.array([1,3,2,5]).reshape((4,1))
    Y = np.array([2,5, 3, 8])
    X_test = np.array([1,2,10]).reshape((3,1))
    alpha = 0.3
    expected = Ridge(alpha).fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    assert list(actual) == pytest.approx(list(expected), rel=1e-5)

def test_RidgeRegressionInThreeDim():
    X = np.array([1,2,3,5,4,5,4,3,3,3,2,5]).reshape((4,3))
    Y = np.array([2,5, 3, 8])
    X_test = np.array([1,0,0, 0,1,0, 0,0,1, 2,5,7, -2,0,3]).reshape((5,3))
    alpha = 0.4
    expected = Ridge(alpha).fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    assert list(actual) == pytest.approx(list(expected), rel=1e-3)
    
