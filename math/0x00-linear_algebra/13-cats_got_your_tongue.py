#!/usr/bin/env python3
""" Concat numpy ndarrays """
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ Returns a new matrix of mat2 concatenated to mat1
        If axis is 0, mat2 is concatenated as new rows
        If axis is 1, mat2 is concatenated as new columns
    """
    return np.concatenate((mat1, mat2), axis)
