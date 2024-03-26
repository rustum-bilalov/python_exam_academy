#!/usr/bin/env python
# coding: utf-8
import numpy as np
from sklearn.metrics import mean_squared_error
class LinearRegression:
    def __init__(self, fit_intercept=True, lr=0.01, max_iter=100, sgd=False, n_sample=16, regularization=None, alpha=0.1):
        self.fit_intercept = fit_intercept
        self.w = None
        self.lr = lr
        self.max_iter = max_iter
        self.sgd = sgd
        self.n_sample = n_sample
        self.regularization = regularization
        self.alpha = alpha

    def fit(self, X, y):
        n, k = X.shape

        if self.w is None:
            self.w = np.random.randn(k + 1 if self.fit_intercept else k)

        X_train = np.hstack((X, np.ones((n, 1)))) if self.fit_intercept else X

        self.losses = []

        for iter_num in range(self.max_iter):
            y_pred = self.predict(X)
            self.losses.append(mean_squared_error(y_pred, y))

            grad = self._calc_gradient(X_train, y, y_pred)

            assert grad.shape == self.w.shape, f"gradient shape {grad.shape} is not equal weight shape {self.w.shape}"
            self.w -= self.lr * grad

        return self

    def _calc_gradient(self, X, y, y_pred):
        if self.sgd:
            inds = np.random.choice(np.arange(X.shape[0]), size=self.n_sample, replace=False)
            grad = 2 * (y_pred[inds] - y[inds])[:, np.newaxis] * X[inds]
            grad = grad.mean(axis=0)
        else:
            grad = 2 * (y_pred - y)[:, np.newaxis] * X
            grad = grad.mean(axis=0)
        
        if self.regularization == 'l1':
            grad += self.alpha * np.sign(self.w)
        elif self.regularization == 'l2':
            grad += 2 * self.alpha * self.w
        elif self.regularization is None:
            pass
        else:
            raise ValueError("Invalid regularization type. Choose from 'l1', 'l2' or None.")

        return grad

    def get_losses(self):
        return self.losses

    def predict(self, X):
        n, k = X.shape
        X_train = np.hstack((X, np.ones((n, 1)))) if self.fit_intercept else X
        y_pred = X_train @ self.w
        return y_pred

    def get_weights(self):
        return self.w



