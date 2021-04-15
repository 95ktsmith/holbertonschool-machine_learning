#!/usr/bin/env python3
""" Gaussian Process """
import numpy as np


class GaussianProcess:
    """ Noiseless 1D Gaussian Process """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        X_init is a numpy.ndarray of shape (t, 1) representing the inputs
            already sampled with the black-box function
        Y_init is a numpy.ndarray of shape (t, 1) representing the outputs of
            the black-box function for each input in X_init
        t is the number of initial samples
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output of the black-box
            function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        """
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) -\
            2 * np.dot(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """
        Predicts mean and covariance of points in a Gaussian Process
        Returns mean and covariance, respectively
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s) + 1e-8 * np.eye(len(X_s))
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T @ K_inv @ self.Y
        sigma = K_ss - K_s.T @ K_inv @ K_s

        return mu.reshape(len(mu)), np.diag(sigma)
