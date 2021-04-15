#!/usr/bin/env python3
""" Bayesian Optimization model """
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        f is the black-box function to be optimized
        X_init is a numpy.ndarray of shape (t, 1) representing the inputs
            already sampled with the black-box function
        Y_init is a numpy.ndarray of shape (t, 1) representing the outputs of
            the black-box function for each input in X_init
        t is the number of initial samples
        bounds is a tuple of (min, max) representing the bounds of the space in
            which to look for the optimal point
        ac_samples is the number of samples that should be analyzed during
            acquisition
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output of the black-box
            function
        xsi is the exploration-exploitation factor for acquisition
        minimize is a bool determining whether optimization should be performed
            for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(
            bounds[0],
            bounds[1],
            ac_samples
        ).reshape((ac_samples, 1))
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location
        Uses the Expected Improvement acquisition function
        Returns X_next, EI
            X_next is a np.ndarray representing the next best sample point
            EI is a np.ndarray containing the expected improvement of each
                potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize is True:
            mu_sample_opt = 2 * mu - np.min(self.gp.Y)
        else:
            mu_sample_opt = np.max(self.gp.Y)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - self.xsi
            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI
