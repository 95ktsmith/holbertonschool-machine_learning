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
    try:
        maxes = np.amax(X, axis=0)
        mins = np.amin(X, axis=0)
        C = np.random.uniform(mins, maxes, (k, X.shape[1]))
        C_prev = np.ones(C.shape)
        clss = np.negative(np.ones((X.shape[0])))

        i = 0
        while i < iterations and np.any(C != C_prev):
            i += 1
            # Save copy of C before iteration
            C_prev = np.copy(C)

            # Assign data points to centroids
            for point in range(len(X)):
                dist = np.linalg.norm(C[:] - X[point], axis=1)
                clss[point] = np.argmin(dist)

            # Reassign centroids if no related data points
            replace = np.isin(np.array(range(len(C))), clss)
            replace = np.argwhere(replace == False)
            if len(replace != 0):
                C[replace[0]] = np.random.uniform(
                    mins,
                    maxes,
                    (len(replace[0]), 2)
                )

            # Move centroids to means of clusters
            for centroid in range(len(C)):
                points = np.argwhere(clss == centroid)
                if len(points) != 0:
                    points = points.reshape((points.shape[0]))
                    C[centroid] = np.mean(X[points], axis=0)

        return C, clss

    except Exception as e:
        return None, None
