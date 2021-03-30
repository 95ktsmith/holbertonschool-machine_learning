#!/usr/bin/env python3
""" K-Means """
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset
    X is a numpy.ndarray of shape (n, d) containing the dataset
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containiner the number of clusters
    iterations is a positive integer containing the maxiumum number of
        iterations that should be performed
    Returns C, clss or None, None on failure
        C is a numpy.ndarray of shape (k, d) containing the centroid means
            for each cluster
        clss is a numpy.ndarray of shape (n, ) containing the index of the
            cluster in C that each data point belongs to
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    n, d = X.shape
    if type(k) is not int or k < 1 or k > n:
        return None, None
    if type(iterations) is not int or iterations < 1:
        return None, None

    maxes = np.amax(X, axis=0)
    mins = np.amin(X, axis=0)
    C = np.random.uniform(mins, maxes, (k, d))
    C_prev = np.ones(C.shape)
    clss = np.negative(np.ones((n)))

    i = 0
    while i < iterations and np.any(C != C_prev):
        i += 1
        # Save copy of C before iteration
        C_prev = np.copy(C)

        # Assign data points to centroids
        points = np.repeat(X, k, axis=0).reshape((n, k, d))
        dist = np.linalg.norm(points[:] - C, axis=2)
        clss = np.argmin(dist, axis=1)

        # Reassign centroids if no related data points
        replace = np.isin(np.array(range(len(C))), clss)
        replace = np.argwhere(replace == False)
        if len(replace != 0):
            C[replace[0]] = np.random.uniform(
                mins,
                maxes,
                (len(replace[0]), d)
            )

        # Move centroids to means of clusters
        for centroid in range(len(C)):
            points = np.argwhere(clss == centroid)
            if len(points) != 0:
                points = points.reshape((points.shape[0]))
                C[centroid] = np.mean(X[points], axis=0)

    return C, clss
